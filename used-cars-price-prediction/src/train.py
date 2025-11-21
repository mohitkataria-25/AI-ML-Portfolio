import os
import time

import argparse
import pandas as pd
from joblib import dump
from datetime import datetime



from .models import build_mlp_regressor, build_random_forest_regressor, build_xgboost_regressor
from .preprocessing import load_and_preprocess
from .evaluate import evaluate_model


def parse_args():

    parser = argparse.ArgumentParser(description="Train an MLP regressor on the used cars dataset")

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/used_cars_data.csv",
        help="Path to the used cars CSV"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch Size."
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=[128, 64, 32],
        nargs="*",
        help="nuerons per hidden layers."
    )
    parser.add_argument(
        "--activations",
        type=str,
        nargs="*",
        default=None,
        help="activation function for each hidden layer."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate for adam optimizor."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="used_cars_mlp.h5",
        help="path where model can be saved."
    )

    return parser.parse_args()

def save_sklearn_model(model, path: str) -> None:
    """
    Save a scikit-learn / XGBoost model using joblib.
    """
    model_dir = os.path.dirname(path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    dump(model, path)

def save_keras_model(model, path):
    """
    Save a Keras (TensorFlow) model.
    """
    args = parse_args()
    model_dir = os.path.dirname(path)
    if model_dir:
        os.makedirs(model_dir, exists_ok = True)
    model.save(path)

def add_results(model_name, split_name, metrics, working_list):
    working_list.append({
        "model": model_name,
        "split": split_name,
        "mse":metrics["mse"],
        "rmse":metrics["rmse"],
        "mae":metrics["mae"],
        "r2_score":metrics["r2"]
    })

def main():

    args = parse_args()
    results = []
    

    print("Loading and preprocessing data.....")
    x_train, x_val, x_test, y_train, y_val, y_test, scaler, feature_names = load_and_preprocess(
        data_path=args.data_path,
    )

    input_dim = x_train.shape[1]
    print(f"Number of features are pre processing: {input_dim}")

    #Run randomForest
    model_rf = build_random_forest_regressor()

    print("Start training on Random Forest......")

    start = time.time()
    model_rf.fit(x_train, y_train)
    end = time.time()
    print (f"Training completed on Random Forest, total time taken: {end - start} seconds.")

    print (f"Collecting Metrics......")
    #Extract train metrics
    train_metrics = evaluate_model(x_eval=x_train, y_eval=y_train, model=model_rf)
    add_results("Random Forest", "train", train_metrics, results)
    #Extract validation metrics
    val_metrics = evaluate_model(x_eval=x_val, y_eval=y_val, model=model_rf)
    add_results("Random Forest", "validation", val_metrics, results)
    #Extract testing metrics
    test_metrics = evaluate_model(x_eval=x_test, y_eval=y_test, model=model_rf)
    add_results("Random Forest", "test", test_metrics, results)
    #save model
    print (f"Saving Model......")
    save_sklearn_model(model=model_rf, path=f"models/random_forest/used_cars_rf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl")


    #Run XG Boost
    model_xg = build_xgboost_regressor()
    print("Start training on XG Boost......")
    start = time.time()
    model_xg.fit(x_train, y_train)
    end = time.time()
    print (f"Training completed on XG Boost, total time taken: {end - start} seconds")

    print (f"Collecting Metrics......")
    #Extract train metrics
    train_metrics = evaluate_model(x_eval=x_train, y_eval=y_train, model=model_xg)
    add_results("XG Boost", "train", train_metrics, results)
    #Extract validation metrics
    val_metrics = evaluate_model(x_eval=x_val, y_eval=y_val, model=model_xg)
    add_results("XG Boost", "validation", val_metrics, results)
    #Extract testing metrics
    test_metrics = evaluate_model(x_eval=x_test, y_eval=y_test, model=model_xg)
    add_results("XG Boost", "test", test_metrics, results)
    #save model
    print (f"Saving Model......")
    save_sklearn_model(model=model_xg, path=f"models/xgboost/used_cars_xg_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl")


    #Run Neural Network model
    model_nn = build_mlp_regressor(input_dim=input_dim, hidden_layers=args.hidden_layers, activations=args.activations, learning_rate=args.learning_rate)

    print("Start training on NN......")
    start = time.time()
    history = model_nn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs= args.epochs, batch_size=args.batch_size, verbose=2)
    end = time.time()
    print (f"Training completed on NN, total time taken: {end - start} seconds")

    print (f"Collecting Metrics......")
    #Extract train metrics
    train_metrics = evaluate_model(x_eval=x_train, y_eval=y_train, model=model_nn)
    add_results("Neural Network", "train", train_metrics, results)
    #Extract validation metrics
    val_metrics = evaluate_model(x_eval=x_val, y_eval=y_val, model=model_nn)
    add_results("Neural Network", "validation", val_metrics, results)
    #Extract testing metrics
    test_metrics = evaluate_model(x_eval=x_test, y_eval=y_test, model=model_nn)
    add_results("Neural Network", "test", test_metrics, results)
    #save model
    print (f"Saving Model......")
    save_keras_model(model=model_nn, path=f"models/neural_net/used_cars_mlp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5")

    #build results dataframe
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df)

    #save results
    os.makedirs("metrics", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"metrics/results_summary_runtime_{ts}", index=False)
    
if __name__ == "__main__":
    main()




