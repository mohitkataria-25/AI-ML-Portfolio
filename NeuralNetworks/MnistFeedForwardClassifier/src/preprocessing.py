"""
load data
flatten data
normalize data
train/validation split
one hot encoding
"""

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

def load_mnist_dataset(test_set_size: float = 1.0/6.0, random_state: int=42 ):
    
    #load dataset
    (x_train, y_train), (x_test, y_test) = keras.mnist.dataset.load_data()

    #flatten data
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    #normalize data
    x_train = x_train.astype("float32")/255.0
    x_test = x_test.astype("float32")/255.0

    #train/validate
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, 
        y_train,
        test_size=test_set_size,
        stratify=y_train,
        random_state=random_state

    )
    num_classes = 10
    #one hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    return x_train, x_val, x_test, y_train, y_test, y_val, num_classes