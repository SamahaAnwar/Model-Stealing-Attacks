import argparse
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def print_versions():
    print("=== Versions ===")
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("NumPy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("================\n")


def build_logreg(C, max_iter, seed):
    """StandardScaler + multiclass Logistic Regression."""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            random_state=seed
        ))
    ])


def count_lr_params(pipeline):
    """Number of Logistic Regression parameters = coef + intercept."""
    lr = pipeline.named_steps["lr"]
    return int(lr.coef_.size + lr.intercept_.size)


def make_uniform_queries(X_reference, n_queries, seed):
    """
    UNIFORM query strategy (Runif-style):
    Sample each feature independently from Uniform(min_feature, max_feature),
    where min/max are computed from X_reference (typically the training set).
    """
    rng = np.random.default_rng(seed)
    mins = X_reference.min(axis=0).to_numpy()
    maxs = X_reference.max(axis=0).to_numpy()
    Xq = rng.uniform(low=mins, high=maxs, size=(n_queries, X_reference.shape[1]))
    return pd.DataFrame(Xq, columns=X_reference.columns)

def make_equation_solving_queries(X_train, scaler, n_queries, seed):
    """
    For equation-solving, it's easier to sample in *standardized* space z
    and then map back to x so the API can be queried with raw features.

    z ~ Uniform(min(z_train), max(z_train)) feature-wise
    x = z * scale + mean
    """
    rng = np.random.default_rng(seed)

    Z_train = scaler.transform(X_train)
    z_mins = Z_train.min(axis=0)
    z_maxs = Z_train.max(axis=0)

    Zq = rng.uniform(low=z_mins, high=z_maxs, size=(n_queries, Z_train.shape[1]))

    # Map back to x-space: x = z*scale + mean
    Xq = Zq * scaler.scale_ + scaler.mean_
    return pd.DataFrame(Xq, columns=X_train.columns)


def solve_multinomial_lr_via_log_odds(Z, P, ref_class=None, eps=1e-12):
    """
    Recover softmax-linear parameters from probabilities using:
      log(P_k / P_ref) = (w_k - w_ref)^T Z + (b_k - b_ref)

    We set w_ref=0 and b_ref=0 (valid because softmax is invariant to adding the same
    linear function to all class scores).

    Inputs:
      Z: (n, d) standardized features
      P: (n, K) probabilities from victim
    Returns:
      W: (K, d), b: (K,)
    """
    Z = np.asarray(Z)
    P = np.asarray(P)

    n, d = Z.shape
    K = P.shape[1]
    if ref_class is None:
        ref_class = K - 1

    # clip probs for numerical stability
    P = np.clip(P, eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    # Design matrix A = [Z | 1]
    A = np.hstack([Z, np.ones((n, 1))])  # (n, d+1)

    W = np.zeros((K, d), dtype=float)
    b = np.zeros((K,), dtype=float)

    for k in range(K):
        if k == ref_class:
            continue
        y = np.log(P[:, k] / P[:, ref_class])  # (n,)

        # Solve A * theta = y in least-squares sense
        theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        W[k, :] = theta[:d]
        b[k] = theta[d]

    # ref_class remains zeros
    return W, b, ref_class


class StolenSoftmaxModel:
    """
    A lightweight substitute model:
    - applies the victim's scaler (assumed known in this setup)
    - then uses recovered W,b to compute softmax probabilities and labels
    """
    def __init__(self, scaler, W, b):
        self.scaler = scaler
        self.W = np.asarray(W)
        self.b = np.asarray(b)

    def predict_proba(self, X):
        Z = self.scaler.transform(X)
        scores = Z @ self.W.T + self.b  # (n, K)
        scores = scores - scores.max(axis=1, keepdims=True)  # stability
        expS = np.exp(scores)
        P = expS / expS.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        P = self.predict_proba(X)
        return np.argmax(P, axis=1)

class VictimAPI:
    """
    Black-box wrapper:
    - attacker can call predict_label / predict_proba
    - we count number of queried inputs
    """
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        self.query_count = 0

    def predict_label(self, X_query):
        self.query_count += len(X_query)
        return self.model.predict(X_query)

    def predict_proba(self, X_query):
        self.query_count += len(X_query)
        return self.model.predict_proba(X_query)

    def reset(self):
        self.query_count = 0


def load_dataset(name: str):
    """Return (X_df, y_series, class_names, dataset_name)."""
    name = name.lower().strip()
    if name == "wine":
        ds = load_wine(as_frame=True)
    elif name == "iris":
        ds = load_iris(as_frame=True)
    else:
        raise ValueError("Dataset must be 'wine' or 'iris'.")
    return ds.data, ds.target, list(ds.target_names), name


# -----------------------------
# Core experiment: one dataset, one query_size
# -----------------------------
def run_uniform_retraining_attack(
    dataset_name: str,
    seed: int,
    test_size: float,
    val_size: float,
    C: float,
    max_iter: int,
    query_size: int,
    verbose: bool = True
):
    """
    Runs:
      1) train victim on train
      2) generate uniform synthetic queries using train feature ranges
      3) query victim for labels
      4) train stolen on (X_query, victim_labels)
      5) evaluate on test
    """
    X, y, class_names, ds_name = load_dataset(dataset_name)

    # Split: train/val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )

    # Victim
    victim = build_logreg(C, max_iter, seed)
    victim.fit(X_train, y_train)
    victim_params = count_lr_params(victim)

    victim_val_pred = victim.predict(X_val)
    victim_val_acc = float(accuracy_score(y_val, victim_val_pred))

    # Black-box API
    api = VictimAPI(victim)
    api.reset()

    # Uniform synthetic queries
    q = max(1, int(query_size))
    X_query = make_uniform_queries(X_train, n_queries=q, seed=seed)

    # Query victim labels
    y_victim = api.predict_label(X_query)
    queries_used = int(api.query_count)

    # Stolen
    stolen = build_logreg(C, max_iter, seed)
    stolen.fit(X_query, y_victim)

    # Evaluate
    stolen_test_pred = stolen.predict(X_test)
    victim_test_pred = victim.predict(X_test)

    stolen_acc = float(accuracy_score(y_test, stolen_test_pred))
    fidelity = float(np.mean(stolen_test_pred == victim_test_pred))

    victim_correct_mask = (victim_test_pred == y_test)
    transferability = float(np.mean(stolen_test_pred[victim_correct_mask] == y_test[victim_correct_mask])) \
        if int(np.sum(victim_correct_mask)) > 0 else 0.0

    cm_true_vs_stolen = confusion_matrix(y_test, stolen_test_pred)
    cm_victim_vs_stolen = confusion_matrix(victim_test_pred, stolen_test_pred)

    if verbose:
        print(f"\n=== Dataset: {ds_name.upper()} | query_size={q} ===")
        print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        print("Victim params:", victim_params)
        print(f"Victim val acc: {victim_val_acc:.4f}")
        print("Queries used:", queries_used)
        print(f"Stolen acc (true labels): {stolen_acc:.4f}")
        print(f"Fidelity (stolen==victim): {fidelity:.4f}")
        print(f"Transferability (victim-correct): {transferability:.4f}")
        print("\nClassification report (Stolen vs True):")
        print(classification_report(y_test, stolen_test_pred, target_names=class_names))

    return {
        "dataset": ds_name,
        "victim_model": "logistic_regression",
        "attack": "retraining (uniform queries + victim labels)",
        "query_strategy": "uniform_per_feature_range",
        "seed": seed,
        "splits": {"test_size": test_size, "val_size": val_size},
        "hyperparams": {"C": C, "max_iter": max_iter, "query_size": q},
        "efficiency": {
            "victim_params": int(victim_params),
            "queries_used": int(queries_used),
        },
        "victim_val_acc": float(victim_val_acc),
        "effectiveness_test": {
            "accuracy": float(stolen_acc),
            "fidelity": float(fidelity),
            "transferability": float(transferability),
            "class_names": class_names,
            "confusion_matrix_true_vs_stolen": cm_true_vs_stolen.tolist(),
            "confusion_matrix_victim_vs_stolen": cm_victim_vs_stolen.tolist(),
        },
    }

def run_equation_solving_attack(
    dataset_name: str,
    seed: int,
    test_size: float,
    val_size: float,
    C: float,
    max_iter: int,
    query_size: int,
    verbose: bool = True,
    round_probs: int | None = None,   # optional: simulate limited API precision
):
    """
    Equation-solving attack for multinomial Logistic Regression:
      1) train victim
      2) query victim for probabilities on crafted inputs
      3) solve linear systems using log-odds to recover parameters
      4) evaluate stolen model
    """
    X, y, class_names, ds_name = load_dataset(dataset_name)

    # Split: train/val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )

    # Victim
    victim = build_logreg(C, max_iter, seed)
    victim.fit(X_train, y_train)
    victim_params = count_lr_params(victim)

    victim_val_pred = victim.predict(X_val)
    victim_val_acc = float(accuracy_score(y_val, victim_val_pred))

    # Black-box API
    api = VictimAPI(victim)
    api.reset()

    # --- Queries crafted for equation-solving ---
    scaler = victim.named_steps["scaler"]
    q = max(1, int(query_size))
    X_query = make_equation_solving_queries(X_train, scaler=scaler, n_queries=q, seed=seed)

    # Query victim probabilities
    P = api.predict_proba(X_query)
    if round_probs is not None:
        P = np.round(P, round_probs)

        # renormalize in case rounding broke sums
        P = np.clip(P, 1e-12, 1.0)
        P = P / P.sum(axis=1, keepdims=True)

    queries_used = int(api.query_count)

    # Solve for parameters in standardized space
    Z_query = scaler.transform(X_query)
    W_rec, b_rec, ref_class = solve_multinomial_lr_via_log_odds(Z_query, P, ref_class=None)

    stolen = StolenSoftmaxModel(scaler=scaler, W=W_rec, b=b_rec)

    # Evaluate
    stolen_test_pred = stolen.predict(X_test)
    victim_test_pred = victim.predict(X_test)

    stolen_acc = float(accuracy_score(y_test, stolen_test_pred))
    fidelity = float(np.mean(stolen_test_pred == victim_test_pred))

    victim_correct_mask = (victim_test_pred == y_test)
    transferability = float(np.mean(stolen_test_pred[victim_correct_mask] == y_test[victim_correct_mask])) \
        if int(np.sum(victim_correct_mask)) > 0 else 0.0

    cm_true_vs_stolen = confusion_matrix(y_test, stolen_test_pred)
    cm_victim_vs_stolen = confusion_matrix(victim_test_pred, stolen_test_pred)

    if verbose:
        print(f"\n=== Equation-solving | Dataset: {ds_name.upper()} | query_size={q} ===")
        print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        print("Victim params:", victim_params)
        print(f"Victim val acc: {victim_val_acc:.4f}")
        print("Queries used:", queries_used)
        print(f"Stolen acc (true labels): {stolen_acc:.4f}")
        print(f"Fidelity (stolen==victim): {fidelity:.4f}")
        print(f"Transferability (victim-correct): {transferability:.4f}")
        print(f"Reference class used in log-odds: {ref_class} ({class_names[ref_class]})")
        if round_probs is not None:
            print(f"API probs rounding: {round_probs} decimals")
        print("\nClassification report (Stolen vs True):")
        print(classification_report(y_test, stolen_test_pred, target_names=class_names))

    return {
        "dataset": ds_name,
        "victim_model": "logistic_regression",
        "attack": "equation_solving (log-odds + least-squares)",
        "query_strategy": "uniform_in_standardized_space_then_inverse_scale",
        "seed": seed,
        "splits": {"test_size": test_size, "val_size": val_size},
        "hyperparams": {"C": C, "max_iter": max_iter, "query_size": q, "round_probs": round_probs},
        "efficiency": {
            "victim_params": int(victim_params),
            "queries_used": int(queries_used),
        },
        "victim_val_acc": float(victim_val_acc),
        "effectiveness_test": {
            "accuracy": float(stolen_acc),
            "fidelity": float(fidelity),
            "transferability": float(transferability),
            "class_names": class_names,
            "confusion_matrix_true_vs_stolen": cm_true_vs_stolen.tolist(),
            "confusion_matrix_victim_vs_stolen": cm_victim_vs_stolen.tolist(),
        },
    }



def run_query_sweep(datasets, query_sizes, seed, test_size, val_size, C, max_iter, round_probs=None):
    """
    Returns a DataFrame with one row per (dataset, query_size).
    SAME COLUMN FORMAT as run_query_sweep (uniform retraining).
    """
    rows = []
    for ds in datasets:
        for q in query_sizes:
            res = run_equation_solving_attack(
                dataset_name=ds,
                seed=seed,
                test_size=test_size,
                val_size=val_size,
                C=C,
                max_iter=max_iter,
                query_size=int(q),
                verbose=False,
                round_probs=round_probs
            )

            met = res["effectiveness_test"]
            eff = res["efficiency"]

            rows.append({
                "dataset": res["dataset"],
                "query_size": res["hyperparams"]["query_size"],
                "queries_used": eff["queries_used"],
                "victim_params": eff["victim_params"],
                "victim_val_acc": res["victim_val_acc"],
                "stolen_test_acc": met["accuracy"],
                "fidelity": met["fidelity"],
                "transferability": met["transferability"],
                "seed": res["seed"],
                "test_size": res["splits"]["test_size"],
                "val_size": res["splits"]["val_size"],
            })

            print(f"SWEEP | EQSOL | {ds.upper():4s} q={q:4d} fidelity={met['fidelity']:.4f}")

    return pd.DataFrame(rows).sort_values(["dataset", "query_size"]).reset_index(drop=True)

def main_equation_solving():
    parser = argparse.ArgumentParser(
        description="Equation-solving model stealing on BOTH wine and iris + query-size sweep + Excel export."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)

    # One main run
    parser.add_argument("--query-size", type=int, default=200)

    # Sweep options
    parser.add_argument("--do-sweep", action="store_true")
    parser.add_argument("--sweep-sizes", type=str, default="25,50,100,200,500,1000")

    # Optional: simulate low-precision API
    parser.add_argument("--round-probs", type=int, default=-1,
                        help="If >=0, round predict_proba to this many decimals.")

    # Outputs
    parser.add_argument("--out-xlsx", type=str, default="equation_solving_results.xlsx")
    parser.add_argument("--out-plot", type=str, default="")

    args = parser.parse_args(["--do-sweep", "--sweep-sizes", "25,50,100,200,500,1000"])


    print_versions()

    datasets = ["wine", "iris"]
    round_probs = None if args.round_probs < 0 else args.round_probs

    # 1) Main run at query_size
    results = []
    for ds in datasets:
        res = run_equation_solving_attack(
            dataset_name=ds,
            seed=args.seed,
            test_size=args.test_size,
            val_size=args.val_size,
            C=args.C,
            max_iter=args.max_iter,
            query_size=args.query_size,
            round_probs=round_probs,
            verbose=True
        )
        results.append(res)

    df_summary = results_to_summary_df(results)


def save_summary_only_to_excel(xlsx_path, df_summary):
    os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"\n[OK] Saved final Excel to: {xlsx_path}")

def results_to_summary_df(results_list):
    rows = []
    for res in results_list:
        eff = res["efficiency"]
        met = res["effectiveness_test"]
        hyp = res["hyperparams"]

        # Map to your requested labels
        dataset = res["dataset"]

        attack = "Retraining" if "retraining" in res["attack"].lower() else "Equation-solving"
        substitute = "LR (stolen)"  # because both attacks produce a stolen LR substitute here

        rows.append({
            "Dataset": dataset,
            "Attack": attack,
            "Substitute": substitute,
            "C": hyp.get("C"),
            "Stolen Accuracy": met.get("accuracy"),
            "Victim Accuracy": res.get("victim_val_acc"),   # if you prefer victim TEST accuracy, tell me
            "Fidelity": met.get("fidelity"),
            "Transferability": met.get("transferability"),
            "Queries": eff.get("queries_used"),
            "Model_Params": eff.get("victim_params"),
        })

    cols = [
        "Dataset", "Attack", "Substitute", "C",
        "Stolen Accuracy", "Victim Accuracy", "Fidelity",
        "Transferability", "Queries", "Model_Params"
    ]
    return pd.DataFrame(rows)[cols]

def adapt_results_for_plots(results):
    """
    Convert experiment result dicts into the flat structure
    expected by the plotting utilities, with distinct labels
    for each attack strategy.
    """
    adapted = []
    for res in results:
        met = res["effectiveness_test"]
        eff = res["efficiency"]

        # Distinguish attack strategies
        if "retraining" in res["attack"].lower():
            label = "Uniform retraining (LR)"
        elif "equation" in res["attack"].lower():
            label = "Equation-solving (LR)"
        else:
            label = "Stolen LR"

        adapted.append({
            "Substitute": label,
            "Accuracy": met["accuracy"],
            "Fidelity": met["fidelity"],
            "Transferability": met["transferability"],
            "Queries": eff["queries_used"],
        })
    return adapted


def plot_effectiveness(dataset, results, attack_type=None):
    attacks = [r["Substitute"] for r in results]

    accuracy = [r["Accuracy"] for r in results]
    fidelity = [r["Fidelity"] for r in results]
    transferability = [r.get("Transferability", 0) for r in results]

    x = np.arange(len(attacks))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, accuracy, width, label="Accuracy")
    plt.bar(x, fidelity, width, label="Fidelity")
    plt.bar(x + width, transferability, width, label="Transferability")

    plt.xticks(x, attacks)
    plt.ylabel("Score")
    plt.title(f"Effectiveness of {attack_type} Attack ({dataset.capitalize()})")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_efficiency_tradeoff(dataset, results, victim_model_specs):
    queries = [r["Queries"] for r in results]
    params = [victim_model_specs['n_parameters']] * len(results)
    labels = [r["Substitute"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.scatter(params, queries)

    for i, label in enumerate(labels):
        plt.annotate(label, (params[i], queries[i]))

    plt.xlabel("Target Model Parameters")
    plt.ylabel("Number of Queries")
    plt.title(f"Efficiency of Model Extraction Attacks ({dataset.capitalize()})")
    plt.tight_layout()
    plt.show()

def main_final_lr():
    print_versions()

    # Keep these consistent with your final settings
    seed = 42
    test_size = 0.25
    val_size = 0.25
    C = 1.0
    max_iter = 2000

    datasets = ["iris", "wine"]

    sweep_sizes = [25, 50, 100, 200, 500, 1000]

    print("\n==============================")
    print("SWEEP (Uniform retraining)")
    print("==============================")
    df_sweep_uniform = run_query_sweep(
        datasets=datasets,
        query_sizes=sweep_sizes,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        C=C,
        max_iter=max_iter
    )
    print(df_sweep_uniform.to_string(index=False))

    print("\n==============================")
    print("SWEEP (Equation-solving)")
    print("==============================")
    df_sweep_eq = run_query_sweep(
        datasets=datasets,
        query_sizes=sweep_sizes,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        C=C,
        max_iter=max_iter,
        round_probs=None
    )
    print(df_sweep_eq.to_string(index=False))

    FINAL_Q_UNIFORM = 200
    FINAL_Q_EQSOL = 25

    results = []

    print("\n==============================")
    print("FINAL RUNS: Uniform retraining")
    print("==============================")
    for ds in datasets:
        results.append(run_uniform_retraining_attack(
            dataset_name=ds,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
            C=C,
            max_iter=max_iter,
            query_size=FINAL_Q_UNIFORM,
            verbose=True      # prints to console
        ))

    print("\n==============================")
    print("FINAL RUNS: Equation-solving")
    print("==============================")
    for ds in datasets:
        results.append(run_equation_solving_attack(
            dataset_name=ds,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
            C=C,
            max_iter=max_iter,
            query_size=FINAL_Q_EQSOL,
            round_probs=None,
            verbose=True      # prints to console
        ))

    # Build final summary (4 rows)
    df_summary = results_to_summary_df(results)

    # Optional: make it pretty in Excel
    df_summary = df_summary.sort_values(["Dataset", "Attack"]).reset_index(drop=True)

    # Save only the summary to one Excel
    save_summary_only_to_excel("Final_LR.xlsx", df_summary)
        # ==============================
    # VISUALISATIONS
    # ==============================
    print("\nGenerating plots...")

    # Split results by dataset
    for dataset in datasets:
        ds_results = [r for r in results if r["dataset"] == dataset]

        plot_ready = adapt_results_for_plots(ds_results)

        victim_params = ds_results[0]["efficiency"]["victim_params"]

        plot_effectiveness(
            dataset=dataset,
            results=plot_ready,
            attack_type="Model Stealing"
        )

        plot_efficiency_tradeoff(
            dataset=dataset,
            results=plot_ready,
            victim_model_specs={"n_parameters": victim_params}
        )


    print("\nDone.")

if __name__ == "__main__":
    main_final_lr()

