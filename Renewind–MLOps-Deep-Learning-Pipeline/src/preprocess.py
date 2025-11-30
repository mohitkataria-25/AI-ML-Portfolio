import os
import argparse
import yaml
import joblib

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


def load_df(path: str, fmt: str) -> pd.DataFrame:
    """Load a DataFrame from parquet or csv."""
    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")


def save_df(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Save a DataFrame to parquet or csv."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")


def compute_class_weights(y_train: pd.Series) -> dict:
    """Compute balanced class weights from y_train."""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {cls: weight for cls, weight in zip(classes, weights)}


def scale_data(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
):
    """Fit a StandardScaler on x_train and transform all splits."""
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = pd.DataFrame(
        scaler.transform(x_train),
        columns=x_train.columns,
    )
    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val),
        columns=x_val.columns,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        columns=x_test.columns,
    )

    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


def main():
    parser = argparse.ArgumentParser(description="Run feature scaling module")
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        type=str,
        help="Path to the configurations for the current run",
    )
    args = parser.parse_args()

    print("[PREPROCESS] scale training, validation & test datasets")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_cfg = config["preprocess"]["input_paths"]
    output_cfg = config["preprocess"]["output_paths"]

    input_fmt = input_cfg["input_fmt"]
    output_fmt = output_cfg["output_fmt"]

    # === 1. Load separate X and y splits from data_ingest output ===
    print("[PREPROCESS] Loading split data from interim…")
    #interim_dir = input_cfg["input_paths"]

    x_train = load_df(
        path=input_cfg["x_train"],
        fmt=input_fmt,
    )
    x_val = load_df(
        path=input_cfg["x_val"],
        fmt=input_fmt,
    )
    x_test = load_df(
        path=input_cfg["x_test"],
        fmt=input_fmt,
    )

    y_train = load_df(
        path=input_cfg["y_train"],
        fmt=input_fmt,
    ).squeeze()
    y_val = load_df(
        path=input_cfg["y_val"],
        fmt=input_fmt,
    ).squeeze()
    y_test = load_df(
        path=input_cfg["y_test"],
        fmt=input_fmt,
    ).squeeze()

    # === 2. Class weights from y_train ===
    print("[PREPROCESS] Computing class weights…")
    weights = compute_class_weights(y_train=y_train)
    print(f"[PREPROCESS] Class weights: {weights}")

    # === 3. Scale only X ===
    print("[PREPROCESS] Scaling features…")
    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_data(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
    )

    # === 4. Save scaler ===
    print("[PREPROCESS] Saving scaler for inference…")
    scaler_path = output_cfg["scaler_path"]
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # === 5. Save scaled X and copy y to processed folder ===
    print("[PREPROCESS] Saving scaled splits and labels…")

    # features
    save_df(x_train_scaled, output_cfg["x_train"], output_fmt)
    save_df(x_val_scaled,   output_cfg["x_val"],   output_fmt)
    save_df(x_test_scaled,  output_cfg["x_test"],  output_fmt)

    # labels (unscaled)
    save_df(y_train.to_frame(name="TARGET"), output_cfg["y_train"], output_fmt)
    save_df(y_val.to_frame(name="TARGET"),   output_cfg["y_val"],   output_fmt)
    save_df(y_test.to_frame(name="TARGET"),  output_cfg["y_test"],  output_fmt)

    print("[PREPROCESS] Done.")


if __name__ == "__main__":
    main()