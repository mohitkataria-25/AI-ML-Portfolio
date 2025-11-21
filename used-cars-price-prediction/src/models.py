from typing import List, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def build_random_forest_regressor(
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
):
    """
    Build and return a RandomForestRegressor with configurable hyperparameters.

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int or None): Maximum depth of each tree.
        min_samples_split (int): Minimum samples required to split a node.
        min_samples_leaf (int): Minimum samples required at a leaf node.
        random_state (int): Seed for reproducibility.
        n_jobs (int): Number of parallel jobs.

    Returns:
        RandomForestRegressor: Configured Random Forest model.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state
    )

def build_xgboost_regressor(
        learning_rate : float = 0.05,
        n_estimators: int = 300,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state : int = 42
):
    """
    Build and return an XGBRegressor optimized for tabular regression tasks.

    Args:
        learning_rate (float): Step size shrinkage.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum depth of trees.
        subsample (float): Fraction of samples per tree.
        colsample_bytree (float): Fraction of features per tree.
        random_state (int): Seed for reproducibility.

    Returns:
        XGBRegressor: Configured XGBoost regression model.
    """
    return XGBRegressor(
        learning_rate = learning_rate,
        n_estimators = n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1
    )

def build_mlp_regressor(
        input_dim : int,
        hidden_layers : Optional[List[int]] = None,
        activations: Optional[List[str]] = None,
        learning_rate : float = 1e-3
):
    """
    Build and return a Keras Sequential MLP regressor.

    Args:
        input_dim (int): Number of input features.
        hidden_layers (List[int] or None): Units per hidden layer.
        activations (List[str] or None): Activations per hidden layer.
        learning_rate (float): Learning rate for Adam optimizer.

    Returns:
        Sequential: Compiled MLP regression model.
    """
    
    #Hidden_layers and activations defalut settings
    if hidden_layers is None or len(hidden_layers) == 0:
        hidden_layers = [128, 64, 32]
    
    if activations is None:
        activations = ["relu"] * len(hidden_layers)
    elif len(activations) != len(hidden_layers):
        raise ValueError("Length of activations must match length of hidden layers.")
    
    #Intilaize Model & 1st layer with input dim
    model = Sequential()

    model.add(Dense(hidden_layers[0], activation=activations[0], input_dim = input_dim))

    #intialize Hidden layers
    for units, act in zip(hidden_layers[1:], activations[1:]):
        model.add(Dense(units, activation=act))
    
    #intiliaze output layer
    model.add(Dense(1)) #linear activaltion by default

    #compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer = optimizer,
                  loss="mse",
                  metrics=["mae"]
    )

    return model