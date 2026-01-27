import argparse
from LR_EX3_final import print_versions, run_uniform_retraining_attack, run_equation_solving_attack,save_summary_only_to_excel,results_to_summary_df, print_versions
from Updated_MLP import run_line_search_cli, run_uniform_retraining_cli, save_results_to_excel
import os

def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Logistic Regression and MLP experiments."
    )

    # General parameters
    parser.add_argument("--model", type=str, choices=["lr", "mlp"], required=True,
                        help="Choose the model to evaluate: 'lr' for Logistic Regression, 'mlp' for MLP.")
    parser.add_argument("--dataset", type=str, choices=["iris", "wine"], required=True,
                        help="Dataset to use: 'iris' or 'wine'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Proportion of data for testing.")
    parser.add_argument("--val-size", type=float, default=0.25, help="Proportion of data for validation.")

    # Logistic Regression-specific parameters
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for Logistic Regression.")
    parser.add_argument("--max-iter", type=int, default=2000, help="Maximum iterations for Logistic Regression.")
    parser.add_argument("--query-size", type=int, default=200, help="Number of queries for model stealing attacks.")
    parser.add_argument("--attack-type", type=str, choices=["uniform", "equation"], default="uniform",
                        help="Attack type for Logistic Regression: 'uniform' or 'equation'.")

    # MLP-specific parameters
    parser.add_argument("--mlp-attack-type", type=str, choices=["uniform", "line_search"], default="uniform",
                        help="Attack type for MLP: 'uniform' or 'line_search'.")

    # Output options
    parser.add_argument("--save-results", type=str, default="results.xlsx",
                        help="Path to save the results as an Excel file.")

    args = parser.parse_args()

    print_versions()

    if args.model == "lr":
        if args.attack_type == "uniform":
            print(f"Running Logistic Regression Uniform Retraining Attack on {args.dataset}...")
            result = run_uniform_retraining_attack(
                dataset_name=args.dataset,
                seed=args.seed,
                test_size=args.test_size,
                val_size=args.val_size,
                C=args.C,
                max_iter=args.max_iter,
                query_size=args.query_size,
                verbose=True
            )
            if isinstance(result, dict):  # Ensure it's a list of results
                result = [result]
            df_summary = results_to_summary_df(result)
            # Optional: make it pretty in Excel
            df_summary = df_summary.sort_values(["Dataset", "Attack"]).reset_index(drop=True)

            save_summary_only_to_excel(args.save_results, df_summary)
            print(f"\nResults saved to: {os.path.abspath(args.save_results)}")

        elif args.attack_type == "equation":
            print(f"Running Logistic Regression Equation-Solving Attack on {args.dataset}...")
            result = run_equation_solving_attack(
                dataset_name=args.dataset,
                seed=args.seed,
                test_size=args.test_size,
                val_size=args.val_size,
                C=args.C,
                max_iter=args.max_iter,
                query_size=args.query_size,
                verbose=True
            )
            if isinstance(result, dict):  # Ensure it's a list of results
                result = [result]
            df_summary = results_to_summary_df(result)
            # Optional: make it pretty in Excel
            df_summary = df_summary.sort_values(["Dataset", "Attack"]).reset_index(drop=True)

            save_summary_only_to_excel(args.save_results, df_summary)
            print(f"\nResults saved to: {os.path.abspath(args.save_results)}")
        ''' print("\nResults:")
        print(result)'''

    elif args.model == "mlp":
        if args.mlp_attack_type == "uniform":
            print(f"Running MLP Uniform Retraining Attack on {args.dataset}...")
            results = run_uniform_retraining_cli(
                dataset_name=args.dataset,
                seed=args.seed,
                test_size=args.test_size,
                max_iter=args.max_iter,
                init_size=args.query_size,
                n_rounds=5,
                n_samples_per_round=500,
                verbose=True
            )
            save_results_to_excel(results, filename=args.save_results)
            print(f"\nResults saved to: {os.path.abspath(args.save_results)}")

        if args.mlp_attack_type == "line_search":
            print(f"Running MLP {args.mlp_attack_type.capitalize()} Attack on {args.dataset}...")
            results = run_line_search_cli(
                dataset_name=args.dataset,
                seed=args.seed,
                test_size=args.test_size,
                verbose=True
            )
            save_results_to_excel(results, filename=args.save_results)
            print(f"\nResults saved to: {os.path.abspath(args.save_results)}")


if __name__ == "__main__":
    main()
