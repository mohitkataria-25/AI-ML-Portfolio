import argparse

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from tensorflow.keras.models import load_model

from .preprocessing import load_and_preprocess
"""
def parse_args():

    args = argparse.ArgumentParser(description="Evaluate a trained used car model on test dataset.")

    args.add_argument(
        "--data_path",
        type=str,
        default="data\used_cars_data.csv",
        help="Path to the dataset."
    ),
    args.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved keras model (.h5)"
    )

    return args.parse_args()

"""


def evaluate_model (model, x_eval, y_eval):
    
    
    print(f"Loading and preprocessing data...." )
    y_pred = model.predict(x_eval).flatten()
    mse = mean_squared_error(y_pred=y_pred, y_true=y_eval)
    rmse = mse ** 0.5
    mae = mean_squared_error(y_pred=y_pred, y_true=y_eval)
    r2 = r2_score(y_pred=y_pred, y_true=y_eval)

    return ({"mse":mse, "rmse":rmse, "mae":mae, "r2":r2})

"""

def main ():

    #get arguments
    args = parse_args()

    #Get test data for evaluation
    print("Loading and preprocessing data...")
    x_train, x_val, x_test, y_train, y_val, y_test, scalar, features = load_and_preprocess(data_path=args.data_path)

    #Load the model from models directory
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)

    #Get test predictions for evaluation
    y_pred = model.predict(x_test).flatten()

    #Evaluate Model and get metrics
    test_mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    test_rmse = test_mse ** 0.5
    test_mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    test_r2 = r2_score(y_true=y_test, y_pred=y_pred)

    print("\nTest set performance:")
    print (f"TEST MSE: {test_mse}")
    print (f"TEST RMSE: {test_rmse}")
    print (f"TEST MAE: {test_mae}")
    print (f"TEST R2 SCORE: {test_r2}")

if __name__ == "__main__":
    main()


"""