import os
import json
import shutil
import pandas as pd
from datetime import datetime
from src.preprocess import compute_class_weights

import yaml
import joblib
import mlflow

from src.model_utils import (
    build_nn_model,
    compile_model,
)


def load_data (config):

    fmt = config["output_fmt"]
 
    if fmt == "parquet":
        x_train = pd.read_parquet(config["x_train"])
        x_val = pd.read_parquet(config["x_val"])
        y_train = pd.read_parquet(config["y_train"])
        y_val = pd.read_parquet(config["y_val"])
        

    elif fmt == "csv":
        x_train = pd.read_csv(config["x_train"])
        x_val = pd.read_csv(config["x_val"])
        y_train = pd.read_csv(config["y_train"])
        y_val = pd.read_csv(config["y_val"])
    else:
        raise ValueError ("Unsupported file format...")
    
    return (x_train, x_val, y_train.squeeze(), y_val.squeeze())


# --------------------------------------------------------------------
# Artifact / output paths
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CURRENT_DIR = os.path.join(ARTIFACTS_DIR, "current")
MODEL_DIR = os.path.join(CURRENT_DIR, "models")
PLOT_DIR = os.path.join(CURRENT_DIR, "plots")
SCALER_DIR = os.path.join(CURRENT_DIR, "scalers")
METRICS_DIR = os.path.join(CURRENT_DIR, "metrics")
CONFIG_USED_DIR = os.path.join(CURRENT_DIR, "config")

#directories exist check
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CONFIG_USED_DIR, exist_ok=True)


def main():
    # load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # set mlflow params
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config["training"]["experiment_name"])

    with mlflow.start_run():
        # load data
        x_train, x_val, y_train, y_val = load_data(config=config["preprocess"]["output_paths"])

        # set class weights
        cw = compute_class_weights(y_train=y_train)

        # define input dimension
        input_dim = x_train.shape[1]

        # build model
        model = build_nn_model(input_dim=input_dim, config=config)

        # compile model
        model = compile_model(model=model, config=config)

        # training params
        epochs = config["training"]["epochs"]
        batch_size = config["training"]["batch_size"]

        # start training
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=cw,
            verbose=1,
        )

        #save history
        with open(os.path.join(METRICS_DIR, f"history.json"), "w") as f:
            json.dump(history.history, f, indent=2)


        # save metrics in mlflow
        mlflow.log_params(
            {
                "optimizer": config["training"]["optimizer"],
                "epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "hidden_layers": config["training"]["hidden_layers"],
                "activations": config["training"]["activation"],
                "dropout": config["training"]["dropout_rate"],
                "l2": config["training"]["l2"],
            }
        )
       

        # save model
        model.save(os.path.join(MODEL_DIR, f"model.h5"))
        mlflow.keras.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
