import argparse as args
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#============Defining File Paths for Datasets============#
IRIS_DATA_PATH = 'Data/iris.data'
WINE_DATA_PATH = 'Data/wine.data'

def read_data_iris(DATA_PATH):
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(DATA_PATH, header=None, names=columns)

    # Drop empty rows if present
    df = df.dropna()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype("category").cat.codes.values
    return X, y

def read_data_wine(DATA_PATH):
    columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
               'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
               'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
    df = pd.read_csv(DATA_PATH, header=None, names=columns)

    # Drop empty rows if present
    df = df.dropna()
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].astype("category").cat.codes.values
    return X, y

def train_MLP(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(50, ), activation='logistic', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = mlp.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
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
        """Query target model and store labeled samples"""
        y = self.target.query(X)
        self.X_sub.append(X)
        self.y_sub.append(y)

    def get_dataset(self):
        X = np.vstack(self.X_sub)
        y = np.hstack(self.y_sub)
        return X, y

    def train(self):
        X, y = self.get_dataset()
        self.substitute.fit(X, y)

    def predict(self, X):
        return self.substitute.predict(X)
    
    def score(self, X, y):
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

def save_results_to_excel(dataset, results, victim_model_specs):
    filename = f"{dataset}_mlp_line_search_retraining_results.xlsx"
    final_results = []
    for i, r in enumerate(results):
        res = {
            "Dataset": dataset, 
            "Seed": args.seed,
            "Attack_Type": "Line Search Retraining",
            "Target_Model": "MLPClassifier",
            "Target_Accuracy": victim_model_specs['accuracy'],
            "Target_Model_Params": victim_model_specs['n_parameters'],
            "Substitute Model": r['Substitute'] if results else 'N/A',
            "Substitute Accuracy": r['Accuracy'] if results else 'N/A',
            "Substitute Fidelity": r['Fidelity'] if results else 'N/A',
            "Substitute Queries": r['Queries'] if results else 'N/A',
            "Substitute Transferability": r['Transferability'] if results else 'N/A',
            "Substitue Model Params": r['Model_Params'] if results else 'N/A',
        }
        final_results.append(res)

    df = pd.DataFrame(final_results)
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

def evaluate_substitutes(substitutes, target, X_train, X_test, y_test):
    results = []
    for name, sub_model in substitutes.items():
        target.reset_queries()

        trainer = SubstituteTrainer(target, sub_model)
        
        # Ideally the attacker would use public data or generate synthetic data from a distribution similar to the target's training data 
        X_pool = X_train.copy()

        #ADD RANDOM RETRAINING CODE FROM JHONA HERE
        trainer = line_search_retraining(trainer, X_pool, n_pairs=1000)
        
        if trainer.X_sub == []:
            print(f"Substitute model {name} could not be trained due to lack of boundary points.")
            continue
       
        fidelity = np.mean(
            trainer.predict(X_test) == target.query(X_test)
        )

        accuracy = sub_model.score(X_test, y_test)
        queries = target.query_count
        n_params = mlp_parameters(sub_model) if isinstance(sub_model, MLPClassifier) else sum(p.size for p in sub_model.coef_) + sub_model.intercept_.size
        transferability = calculate_transferability(sub_model, target, X_test, X_test)

        results.append({
            "Target": "MLP",
            "Substitute": name,
            "Accuracy": accuracy,
            "Fidelity": fidelity,
            "Queries": queries, 
            "Model_Params": n_params, 
            "Transferability": transferability
        })          
    return results

def attack_pipeline(dataset):
    if dataset == 'iris':
        X, y = read_data_iris(IRIS_DATA_PATH)
    elif dataset == 'wine':
        X, y = read_data_wine(WINE_DATA_PATH)
    else:
        raise ValueError("Unsupported dataset")     

    victim_model_specs = train_MLP(X, y)
    mlp = victim_model_specs['model']
    scaler = victim_model_specs['scaler']
    X_train = victim_model_specs['X_train']
    target = VictimAPI(mlp, scaler=scaler)
    y_train = victim_model_specs['y_train'] 

    plot_decision_slice(
            model=target,
            scaler=target.scaler,
            X_train_full=X_train,
            y_train=y_train,
            feature_indices=(0, 1),
            title=f"{dataset} – 2D Slice of Full-Dimensional Decision Surface"
        )
    #==============ATTACK PIPELINE==============#
    ##Assuming the attacker thinks the target model is an MLP, he can use an MLP as substitute model 
    sub_lr = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        random_state=42
    )

    sub_mlp_small = MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=1000,
        random_state=42
    )

    sub_mlp_large = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42
    )

    substitutes = {
        "Logistic Regression": sub_lr,
        "MLP Small": sub_mlp_small,
        "MLP Large": sub_mlp_large
    }

    results = evaluate_substitutes(substitutes, target, X_train, victim_model_specs['X_test'], victim_model_specs['y_test'])
    for res in results:
        print(res)

    #save to excel  
    save_results_to_excel(dataset, results, victim_model_specs)
    return results

def plot_effectiveness(dataset, results):
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
    plt.title(f"Effectiveness of Line Search Retraining Attack {dataset.capitalize()}")
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


#============Argument Parsing for Configurability============#
def parse_args():
    parser = args.ArgumentParser(
        description="Model Extraction Attack Experiments - MLP Classifier Victim"
    )

    # Dataset & model
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=["iris", "wine"],
                        help="Dataset to use")
    
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "logreg"],
                        help="Target model type")

    # Attack configuration
    parser.add_argument("--attack", type=str, default="random",
                        choices=["random", "line_search"],
                        help="Attack type")
    parser.add_argument("--queries", type=int, default=500,
                        help="Query budget or number of samples")
    parser.add_argument("--pairs", type=int, default=300,
                        help="Number of pairs for line-search retraining")

    # Evaluation
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Adversarial perturbation strength")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()

if __name__ == "__main__":
   results_iris = attack_pipeline('iris')
   plot_effectiveness('iris', results_iris)
   plot_efficiency_tradeoff('iris', results_iris, train_MLP(*read_data_iris(IRIS_DATA_PATH)))
   results_wine = attack_pipeline('wine')
   plot_effectiveness('wine', results_wine)
   plot_efficiency_tradeoff('wine', results_wine, train_MLP(*read_data_wine(WINE_DATA_PATH)))