import tensorflow as tf
from tensorflow.keras import regularizers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout


def build_nn_model(input_dim: int, config):

    model_cfg = config["training"]
    l2 = model_cfg["l2"]
    dropout_rate = model_cfg["dropout_rate"]
    

    tf.keras.backend.clear_session()
    
    model = Sequential()

    #first hidden layer
    model.add(Dense(
        model_cfg["hidden_layers"][0],
        activation=model_cfg["activation"],
        input_dim=input_dim,
        kernel_regularizer=regularizers.l2(l2),
    ))
    #apply batch normalization if applicable
    if model_cfg["batch_norm"]:
        model.add(BatchNormalization())

    #apply drop if applicable
    if dropout_rate[0]:
        model.add(Dropout(dropout_rate[1]))

    #second hidden layer
    model.add(Dense(
        model_cfg["hidden_layers"][1],
        activation=model_cfg["activation"],
        kernel_regularizer=regularizers.l2(l2),
    ))

    #apply batch normalization if applicable
    if model_cfg["batch_norm"]:
        model.add(BatchNormalization())

    #apply drop if applicable
    if dropout_rate[0]:
        model.add(Dropout(dropout_rate[1]))


    #output layer
    model.add(Dense(1,
              activation=model_cfg["output_activation"]))
    

    return model

def compile_model(model, config):
    
    model_cfg = config["training"]
    lr = model_cfg["learning_rate"]
    optimizer_name = model_cfg["optimizer"]
    
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, 
                  metrics = [tf.keras.metrics.Recall(name="recall")]
                             )
    return model

 
