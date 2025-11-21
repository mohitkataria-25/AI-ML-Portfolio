import os
import argparse
import pandas as pd
from joblib import load as joblib_load
from tensorflow.keras.models import load_model

from .preprocessing import load_and_preprocess
from .evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Load and evaluate one or more saved models")
    parser.add_argument(
        "--model_path",
        type=str,
        nargs="+",
        help="List of model paths to load and evaluate (e.g. .pkl, .h5)",
    )
    args = parser.parse_args()

    model_paths = args.model_path
    if not model_paths:
        print("No model paths provided. Use --model_path to specify one or more saved models.")
        return

    # We only need test data for inference-time evaluation
    _, _, x_test, _, _, y_test, _, _ = load_and_preprocess(
        data_path="data/used_cars_data.csv"
    )

    results = []

    for path in model_paths:
        # Load model based on file extension
        if path.endswith(".pkl"):
            # scikit-learn / XGBoost
            model = joblib_load(path)
        elif path.endswith(".h5") or path.endswith(".keras"):
            # Keras / TensorFlow
            model = load_model(path)
        else:
            print(f"Skipping unsupported model type for path: {path}")
            continue

        # Evaluate model on test set
        metrics = evaluate_model(x_eval=x_test, y_eval=y_test, model=model)

        # Store results as a single row dict
        model_name = os.path.basename(path)
        row = {"model": model_name}
        row.update(metrics)
        results.append(row)

    if not results:
        print("No models were successfully evaluated.")
        return

    results_df = pd.DataFrame(results)

    print("\n=== Inference-time Test Metrics ===")
    print(results_df)


if __name__ == "__main__":
    main()