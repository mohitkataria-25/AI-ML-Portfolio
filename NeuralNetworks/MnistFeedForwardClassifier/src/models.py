"""
check if hidden layers is none or not and return a single a model with no hiddden layers
check if activation is none and assign relu as the default activation mechanishms
intizile a model and add the input layer
loop over the list of hidden layers and add each layer with the activation'
create output layer
retrun model

"""

from typing import List, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_mlp(
        input_dim: int,
        num_classes = 10,
        hidden_layers: Optional[List[int]] = None,
        activations: Optional[List[str]] = None,
):

    if hidden_layers is None or len(hidden_layers) == 0:
        model = Sequential()
        model.add(Dense(num_classes, activation = "softmax", input_dim = input_dim))
        return model
    
    if activations is None:
        activations = ["relu"] * len(hidden_layers)
    
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation = activations[0], input_dim = input_dim))

    for nuerons, act in zip(hidden_layers[0:], activations[0:]):
        model.add(Dense(nuerons, activation=act))
    
    model.add(Dense(num_classes, activation = "softmax"))

    return model
    