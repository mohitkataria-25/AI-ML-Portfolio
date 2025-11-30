import pandas as pd
import numpy as np
import seaborn as sns
import mlflow
import json
import shutil
import yaml
import os

import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from src.preprocess import compute_class_weights


def load_data (config):

    fmt = config["output_fmt"]

    if fmt == "parquet":
        x_train = pd.read_parquet(config["x_train"])
        x_val = pd.read_parquet(config["x_val"])
        x_test = pd.read_parquet(config["x_test"])
        y_train = pd.read_parquet(config["y_train"])
        y_val = pd.read_parquet(config["y_val"])
        y_test = pd.read_parquet(config["y_test"])
        

    elif fmt == "csv":
        x_train = pd.read_csv(config["x_train"])
        x_val = pd.read_csv(config["x_val"])
        x_test = pd.read_csv(config["x_test"])
        y_train = pd.read_csv(config["y_train"])
        y_val = pd.read_csv(config["y_val"])
        y_test = pd.read_csv(config["y_test"])
    else:
        raise ValueError ("Unsupported file format...")
    
    return (x_train, x_val, x_test, y_train.squeeze(), y_val.squeeze(), y_test.squeeze())

def evaluate_model (model, predictors, target, threshold=0.5):

    #generate prediction
    prediction_prob = model.predict(predictors)
    prediction_prob = prediction_prob.ravel()
    prediction = (prediction_prob >= threshold).astype(int)

    #evaluate performance
    recall = recall_score(y_true=target, y_pred=prediction)
    precision = precision_score(y_true=target, y_pred=prediction)
    accuracy = accuracy_score(y_true=target, y_pred=prediction)
    f1 = f1_score(y_true=target, y_pred=prediction)

    return {"recall":recall, "accuracy":accuracy,  "precision":precision, "f1":f1}

def plot_history(history, save_path):

    plt.figure(figsize=(8, 4))

    loss = history.get("loss")
    val_loss = history.get("val_loss")
    recall = history.get("recall")
    val_recall = history.get("val_recall")

    if loss is not None:
        plt.plot(loss, label="training_loss")
    if val_loss is not None:
        plt.plot(val_loss, label="val_loss")
    if recall is not None:
        plt.plot(recall, label="training_recall")
    if val_recall is not None:
        plt.plot(val_recall, label="val_recall")

    plt.xlabel("Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(model, predictors, target, save_path, threshhold=0.5):

    prediction_prob = model.predict(predictors)
    prediction = (prediction_prob >= threshhold).astype(int)

    cm = confusion_matrix(y_true=target, y_pred=prediction)
    total = cm.sum()

    labels = np.array([
        [
            f"{cm[i,j]}\n{cm[i,j]/total:.2%}"
            for j in range(2)
        ]
        for i in range(2)
    ])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels,fmt="")
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main(): 

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts/current")
    CONFIG_USED_DIR = os.path.join(ARTIFACTS_DIR, "config")
    MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
    PLOT_DIR = os.path.join(ARTIFACTS_DIR, "plots")
    METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
    
    #Make sure directories exits
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(CONFIG_USED_DIR, exist_ok=True)

    #history = json.load (os.path.join(config["model"]["model_history"], "history.json"))
    model = load_model(os.path.join(MODEL_DIR, "model.h5"))
   
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(config=config["preprocess"]["output_paths"])
    print(x_train.shape)
    # Optional: compute class weights if needed later
    cw = compute_class_weights(y_train=y_train)

    # Load training history saved as JSON
    history_path = os.path.join(METRICS_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {}

    # save plots
    plot_history(
            history=history,
            save_path=os.path.join(PLOT_DIR, f"plot_lost_recall.png"),
        )
    save_confusion_matrix(
            model=model,
            predictors=x_test,
            target=y_test,
            save_path=os.path.join(PLOT_DIR, f"confusion_matrix_train.png"),
        )

        # evaluate model
    metrics_train = evaluate_model(model=model, predictors=x_train, target=y_train)
    metrics_val = evaluate_model(model=model, predictors=x_val, target=y_val)
    metrics_test = evaluate_model(model=model, predictors=x_test, target=y_test)

    mlflow.log_metrics(
            {
                "train_recall": float(metrics_train["recall"]),
                "val_recall": float(metrics_val["recall"]),
                "test_recall": float(metrics_test["recall"]),
            }
        )

    # Log config snapshot
    config_copy_path = os.path.join(CONFIG_USED_DIR, "config_used.yaml")
    shutil.copy("config.yaml", config_copy_path)
    #mlflow.log_artifact(config_copy_path, artifact_path="config")

    metrics_path = os.path.join(METRICS_DIR, f"metrics.json")
    metrics_summary = {
            "train": metrics_train,
            "validation": metrics_val,
            "test": metrics_test,
        }
    with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)

if __name__ == "__main__":
    main()