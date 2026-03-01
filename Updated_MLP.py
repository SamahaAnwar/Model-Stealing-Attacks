import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine

#============Defining File Paths for Datasets============#
from sklearn.datasets import load_iris, load_wine

def load_dataset(name: str):
    """Return (X_df, y_series_or_array, class_names, dataset_name)."""
    name = name.lower().strip()

    if name == "wine":
        ds = load_wine(as_frame=True)
    elif name == "iris":
        ds = load_iris(as_frame=True)
    else:
        raise ValueError("Dataset must be 'wine' or 'iris'.")

    X = ds.data                # pandas DataFrame
    y = ds.target              # pandas Series (because as_frame=True)
    class_names = list(ds.target_names)

    return X, y, class_names, name


def train_MLP(X, y, test_size=0.3, seed=42, verbose=True):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="logistic",
        solver="adam",
        max_iter=5000,
        #early_stopping=True,
        #n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = mlp.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    if verbose:
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Number of model parameters:", n_params)

    model_specs = {
        'model_type': 'MLPClassifier',
        'model': mlp,
        'scaler': scaler,
        'accuracy': accuracy,
        'n_parameters': n_params, 
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test, 
        'y_test': y_test
    }
    return model_specs

def mlp_parameters(mlp):
    n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    return n_params

class VictimAPI:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
        self.query_count = 0

    def query(self, X):
        """
        Black-box query interface.
        X: np.ndarray of shape (n_samples, n_features)
        Returns predicted labels.
        """
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        self.query_count += len(X)
        return self.model.predict(X)

    def reset_queries(self):
        self.query_count = 0

    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    #============Implementing Line Search Retraining Attack============#

def line_search(x1, x2, target, max_iter=20):
    #max_iter : query budget for line search 
    y1 = target.query(x1)[0]
    y2 = target.query(x2)[0]

    if y1 == y2:
        return None

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        x_mid = mid * x1 + (1 - mid) * x2
        y_mid = target.query(x_mid)[0]

        if y_mid == y1:
            lo = mid
        else:
            hi = mid

    return x_mid


def line_search_retraining(trainer, X_pool, n_pairs=500):
    #X_pool is the unlabeled input data available to the attacker, from which he can sample data points and query from the target model
    boundary_points = []

    for _ in range(n_pairs):
        x1, x2 = X_pool[np.random.choice(len(X_pool), 2, replace=False)]
        x_b = line_search(x1, x2, trainer.target)

        if x_b is not None:
            boundary_points.append(x_b)

    if boundary_points:
        X_b = np.array(boundary_points)
        trainer.add_samples(X_b)
        trainer.train()
    else:
        print("Warning: No boundary points found. Substitute not trained.")

    return trainer

def uniform_retraining(trainer, X_pool, init_size=200, n_rounds=5, n_samples_per_round=500, seed=42):
    """
    Uniform retraining attack:
    - Start with some random points from X_pool (seed set)
    - Then, for multiple rounds:
        sample uniformly in the feature-wise min/max box,
        query victim, add to substitute dataset, retrain.
    """
    rng = np.random.default_rng(seed)

    X_pool = np.asarray(X_pool)
    n_features = X_pool.shape[1]

    # Feature-wise min/max box (uniform hyper-rectangle)
    x_min = X_pool.min(axis=0)
    x_max = X_pool.max(axis=0)

    # --- Initial seed set (random real points from pool) ---
    init_idx = rng.choice(len(X_pool), size=min(init_size, len(X_pool)), replace=False)
    X_init = X_pool[init_idx]
    trainer.add_samples(X_init)
    trainer.train()

    # --- Uniform augmentation rounds ---
    for r in range(n_rounds):
        X_new = rng.uniform(low=x_min, high=x_max, size=(n_samples_per_round, n_features))
        trainer.add_samples(X_new)
        trainer.train()

    return trainer

#============Plotting Decision Boundary for 2D Data Visualization============#
def make_2d_slice_grid(X_train_full, feature_indices, grid_size=300):
    f1, f2 = feature_indices

    x_min, x_max = X_train_full[:, f1].min() - 1, X_train_full[:, f1].max() + 1
    y_min, y_max = X_train_full[:, f2].min() - 1, X_train_full[:, f2].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    return xx, yy

def predict_2d_slice(model, scaler, xx, yy, feature_indices, feature_means):
    f1, f2 = feature_indices
    n_points = xx.size
    n_features = feature_means.shape[0]

    X_full = np.tile(feature_means, (n_points, 1))
    X_full[:, f1] = xx.ravel()
    X_full[:, f2] = yy.ravel()

    X_full = scaler.transform(X_full)
    Z = model.predict(X_full)

    return Z.reshape(xx.shape)


def plot_decision_slice(
    model,
    scaler,
    X_train_full,
    y_train,
    feature_indices,
    title
):
    xx, yy = make_2d_slice_grid(X_train_full, feature_indices)
    feature_means = X_train_full.mean(axis=0)

    Z = predict_2d_slice(
        model, scaler, xx, yy, feature_indices, feature_means
    )

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(
        X_train_full[:, feature_indices[0]],
        X_train_full[:, feature_indices[1]],
        c=y_train,
        edgecolor="k",
        s=40
    )
    plt.xlabel(f"Feature {feature_indices[0]}")
    plt.ylabel(f"Feature {feature_indices[1]}")
    plt.title(title)
    plt.tight_layout()
    plt.show()




#============Defining Substitute Trainer Class (Attacker Class)============#

class SubstituteTrainer:
    def __init__(self, target_model, substitute_model):
        """
        target_model: VictimAPI
        substitute_model: sklearn classifier (MLP or LogisticRegression)
        """
        self.target = target_model
        self.substitute = substitute_model
        self.X_sub = []
        self.y_sub = []

    def add_samples(self, X):
        """
        Query target model on RAW X (VictimAPI scales internally),
        but store SCALED X for training the substitute (same feature space as victim model).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Query victim on raw inputs
        y = self.target.query(X)

        # Store scaled inputs for training substitute (prevents mismatch)
        if self.target.scaler is not None:
            X_store = self.target.scaler.transform(X)
        else:
            X_store = X

        self.X_sub.append(X_store)
        self.y_sub.append(y)

    def get_dataset(self):
        """Return stacked (X, y) dataset for substitute training."""
        if len(self.X_sub) == 0:
            return None, None
        X = np.vstack(self.X_sub)
        y = np.hstack(self.y_sub)
        return X, y

    def train(self):
        X, y = self.get_dataset()
        if X is None or y is None:
            raise ValueError("No samples collected for substitute training.")
        self.substitute.fit(X, y)

    def predict(self, X):
        """Predict with substitute; scale X to match training space."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.target.scaler is not None:
            X = self.target.scaler.transform(X)

        return self.substitute.predict(X)

    def score(self, X, y):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.target.scaler is not None:
            X = self.target.scaler.transform(X)

        return self.substitute.score(X, y)


#============Implementing Adversarial Example Generation to Calculate Transferability============#
def generate_adversarial_examples(substitute, X, eps=0.05, max_iter=50):
    X_adv = X.copy()

    for i in range(len(X)):
        x = X[i].copy()
        y = substitute.predict(x.reshape(1, -1))[0]

        for _ in range(max_iter):
            noise = np.random.normal(scale=eps, size=x.shape)
            x_perturbed = x + noise

            y_new = substitute.predict(x_perturbed.reshape(1, -1))[0]
            if y_new != y:
                X_adv[i] = x_perturbed
                break

    return X_adv

def calculate_transferability(substitute, target, X_test, y_test, eps=0.05):
    # Generate adversarial examples on substitute
    X_adv = generate_adversarial_examples(substitute, X_test, eps=eps)

    # Predictions
    sub_clean = substitute.predict(X_test)
    sub_adv = substitute.predict(X_adv)
    tgt_adv = target.query(X_adv)

    # Only count examples that fooled the substitute
    fooled_sub = sub_adv != sub_clean

    if np.sum(fooled_sub) == 0:
        return 0.0  # no adversarial success on substitute

    # Transferability: fraction that also fool the target
    transferability = np.mean(
        tgt_adv[fooled_sub] != sub_clean[fooled_sub]
    )

    return transferability

 
def evaluate_substitutes(substitutes, target, X_train, X_test, y_test, attack_type):
    results = []
    for name, sub_model in substitutes.items():
        target.reset_queries()

        trainer = SubstituteTrainer(target, sub_model)
        
        # Ideally the attacker would use public data or generate synthetic data from a distribution similar to the target's training data 
        X_pool = X_train.copy()

        # Choose retraining strategy:
        # Choose retraining strategy
        if attack_type == "uniform":
            trainer = uniform_retraining(
                trainer,
                X_pool,
                init_size=200,
                n_rounds=5,
                n_samples_per_round=500,
                seed=42
            )
        elif attack_type == "line_search":
            trainer = line_search_retraining(trainer, X_pool, n_pairs=1000)
        else:
            raise ValueError("attack_type must be 'uniform' or 'line_search'")

        if trainer.X_sub == []:
            print(f"Substitute model {name} could not be trained due to lack of boundary points.")
            continue
       
        fidelity = np.mean(
            trainer.predict(X_test) == target.query(X_test)
        )

        accuracy = sub_model.score(X_test, y_test)
        queries = target.query_count
        n_params = mlp_parameters(sub_model) if isinstance(sub_model, MLPClassifier) else sum(p.size for p in sub_model.coef_) + sub_model.intercept_.size
        transferability = calculate_transferability(sub_model, target, X_test, y_test)

        results.append({
            "Dataset": None,  # we'll fill later (optional)
            "Attack": attack_type,
            "Substitute": name,
            "Accuracy": accuracy,
            "Fidelity": fidelity,
            "Queries": queries,
            "Model_Params": n_params,
            "Transferability": transferability
        })

    return results

def attack_pipeline(dataset):
    # Load from sklearn (no file paths)
    X_df, y_ser, class_names, name = load_dataset(dataset)

    # Convert to numpy arrays (your code expects numpy later)
    X = X_df.values
    y = y_ser.values

    victim_model_specs = train_MLP(X, y, test_size=0.3, seed=42, verbose=True)

    mlp = victim_model_specs['model']
    scaler = victim_model_specs['scaler']
    X_train = victim_model_specs['X_train']
    y_train = victim_model_specs['y_train']


    target = VictimAPI(mlp, scaler=scaler)

    # Plot 2D slice (use raw sklearn model + scaler)
    plot_decision_slice(
        model=target.model,
        scaler=target.scaler,
        X_train_full=X_train,
        y_train=y_train,
        feature_indices=(0, 1),
        title=f"{name} – 2D Slice of Full-Dimensional Decision Surface"
    )

    # Substitutes
    sub_lr = LogisticRegression(
        max_iter=5000,
        random_state=42
    )

    sub_mlp_small = MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=5000,
        random_state=42
    )

    sub_mlp_large = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=5000,
        random_state=42
    )

    substitutes = {
        "Logistic Regression": sub_lr,
        "MLP Small": sub_mlp_small,
        "MLP Large": sub_mlp_large
    }

    X_test = victim_model_specs['X_test']
    y_test = victim_model_specs['y_test']

    all_results = []

    for attack_type in ["uniform", "line_search"]:
        print("\n" + "=" * 60)
        print(f"DATASET: {dataset.upper()} | ATTACK: {attack_type.upper()}")
        print("=" * 60)

        results = evaluate_substitutes(
            substitutes, target, X_train, X_test, y_test, attack_type=attack_type
        )

        '''results = results.append({
            "Victim Parameters": victim_model_specs['n_parameters'],
            "Victim Accuracy": victim_model_specs['accuracy']
        })'''

        if attack_type == "uniform":
            attack_name = "Uniform Retraining"
        else:
            attack_name = "Line Search Retraining"    

        plot_effectiveness(dataset, results, attack_type=attack_name)

        #Plot efficiency trade-off only for line search (since uniform has same #queries for all substitutes)
        if attack_type == "line_search":
            plot_efficiency_tradeoff(dataset, results, victim_model_specs)
        # (Optional) fill dataset name into dicts
        for r in results:
            r["Dataset"] = dataset

        for r in results:
            print(r)

        all_results.extend(results)

    return all_results

def run_uniform_retraining_cli(
    dataset_name: str,
    seed: int,
    test_size: float,
    max_iter: int,
    init_size: int = 200,
    n_rounds: int = 5,
    n_samples_per_round: int = 500,
    verbose: bool = True,
):
    """
    CLI wrapper for the UNIFORM retraining attack on an MLP victim.
    """

    # -----------------------------
    # Bookkeeping
    # -----------------------------
    total_queries = init_size + n_rounds * n_samples_per_round

    if verbose:
        print("\n==============================")
        print("Running UNIFORM retraining attack")
        print("==============================")
        print(f"Dataset: {dataset_name}")
        print(f"Seed: {seed}")
        print(f"Test size: {test_size}")
        print(f"Max iter: {max_iter}")
        print(f"Initial seed size: {init_size}")
        print(f"Rounds: {n_rounds}")
        print(f"Samples per round: {n_samples_per_round}")
        print(f"TOTAL QUERY BUDGET: {total_queries}")
        print("==============================\n")

    # -----------------------------
    # Load dataset
    # -----------------------------
    X_df, y_ser, class_names, name = load_dataset(dataset_name)
    X = X_df.values
    y = y_ser.values

    # -----------------------------
    # Train victim
    # -----------------------------
    victim_model_specs = train_MLP(
        X,
        y,
        seed=seed,
        test_size=test_size
    )

    mlp = victim_model_specs["model"]
    scaler = victim_model_specs["scaler"]
    X_train = victim_model_specs["X_train"]
    y_train = victim_model_specs["y_train"]
    X_test = victim_model_specs["X_test"]
    y_test = victim_model_specs["y_test"]

    target = VictimAPI(mlp, scaler=scaler)

    # -----------------------------
    # Define substitute model
    # -----------------------------
    substitute = {"MLP Small": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=max_iter,
            random_state=seed
        )    }

  
    results = evaluate_substitutes(
        substitutes=substitute,
        target=target,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        attack_type="uniform"
    )

    plot_effectiveness(dataset_name,results,attack_type="Uniform Retraining")
    for r in results:
        r["Dataset"] = dataset_name
        r["Attack"] = "Uniform Retraining"

        if verbose:
            print(r)

    return results


def run_line_search_cli(
    dataset_name: str,
    seed: int,
    test_size: float,
    max_iter: int = 5000,
    verbose: bool = True,
    
):
    """
    CLI wrapper for running ONLY the line-search retraining attack
    on a trained MLP victim.
    """

    if verbose:
        print("\n==============================")
        print("Running LINE-SEARCH retraining attack")
        print("==============================")
        print(f"Dataset: {dataset_name}")
        print(f"Seed: {seed}")
        print(f"Test size: {test_size}")
        #print(f"Max iter (substitutes): {max_iter}")
        print("==============================\n")

    # ---- Load dataset ----
    X_df, y_ser, class_names, name = load_dataset(dataset_name)
    X = X_df.values
    y = y_ser.values

    # ---- Train victim ----
    victim_model_specs = train_MLP(
        X,
        y,
        test_size=test_size,
        seed=seed,
        verbose=verbose
    )

    mlp = victim_model_specs["model"]
    scaler = victim_model_specs["scaler"]
    X_train = victim_model_specs["X_train"]
    y_train = victim_model_specs["y_train"]
    X_test = victim_model_specs["X_test"]
    y_test = victim_model_specs["y_test"]

    target = VictimAPI(mlp, scaler=scaler)

    # ---- Define substitutes ----
    substitutes = {
        
        "MLP Small": MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=max_iter,
            random_state=seed
        )    
    }

    # ---- Run LINE SEARCH attack only ----
    results = evaluate_substitutes(
        substitutes=substitutes,
        target=target,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        attack_type="line_search"
    )

    # ---- Plotting ----
    plot_effectiveness(
        dataset_name,
        results,
        attack_type="Line Search Retraining"
    )

    plot_efficiency_tradeoff(
        dataset_name,
        results,
        victim_model_specs
    )

    # ---- Annotate results ----
    for r in results:
        r["Dataset"] = dataset_name
        r["Attack"] = "Line Search Retraining"

        if verbose:
            print(r)

    return results



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
    plt.title("Efficiency of Model Extraction Attacks")
    plt.tight_layout()
    plt.show()


def save_results_to_excel(all_results, filename="attack_results.xlsx"):
    """
    all_results: list of dicts, e.g. output of attack_pipeline('iris') + attack_pipeline('wine')
    Writes a clean, sorted Excel file.
    """
    df = pd.DataFrame(all_results)

    # Convert numpy types (np.float64 etc.) to normal Python floats for cleaner Excel output
    for col in df.columns:
        df[col] = df[col].apply(lambda v: float(v) if isinstance(v, (np.floating,)) else v)

    # Choose column order (only keep ones that exist)
    desired_cols = ["Dataset", "Attack", "Substitute", ''' "Victim Parameters", "Victim Accuracy", ''' "Accuracy", "Fidelity", "Transferability", "Queries", "Model_Params"]
    cols = [c for c in desired_cols if c in df.columns]
    df = df[cols]

    # Sort nicely
    df = df.sort_values(by=["Dataset", "Attack", "Substitute"], ascending=[True, True, True]).reset_index(drop=True)

    # Save to Excel
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)

    print(f"✅ Saved results to: {filename}")


if __name__ == "__main__":
    all_results = []

    # ---------- IRIS ----------
    results_iris = attack_pipeline("iris")   # should include BOTH uniform + line_search
    all_results.extend(results_iris)

    # ---------- WINE ----------
    results_wine = attack_pipeline("wine")   # should include BOTH uniform + line_search
    all_results.extend(results_wine)

    # ---------- SAVE EXCEL ----------
    save_results_to_excel(all_results, filename="attack_results.xlsx")
