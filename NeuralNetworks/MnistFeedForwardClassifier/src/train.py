import argparse
import os
import time


import numpy
from tensorflow.keras.optimizer import SGD 

from .preprocessing import load_mnist_dataset
from .models import build_mlp

def parse_args():

    parser = argparse.ArgumentParser(description="Train an MLP on the MNIST dataset.")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training Epochs.")
    parser.add_argument("--batchsize", type=int, default=64, help="Number of batches.")
    parser.add_argument("--hiddenlayers", type=int, nargs= "*", default=[128, 64, 32], help="Number of hidden layers. e.g - 128 64 32") 
    parser.add_argument("--activations", type=str, nargs="*", default=None, help="Number of activators for hidden layers. e.g - relu tanh sigmoid")
    parser.add_argument("--learningrate", type=float, default=0.01, help="Learning rate for SGD.")
    parser.add_argument("--model_path", type=str, default="models/mnist_mlp.h5", help="Where do you want to save the model.")

    return parser.parse_args()

def main():

    args = parse_args()

    x_train, x_val, x_test, y_train, y_val, y_test, num_classes = load_mnist_dataset()

    model = build_mlp(
        input_dim=x_train.shape[1], 
        num_classes=num_classes, 
        hidden_layers=args.hiddenlayers,
        activations=args.activations
    )

    optimizer = SGD(learning_rate=args.learningrate)
    model.compile(loss = "categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    print ('Strating model training....')
    start = time.time()

    model.fit(x_train, 
              y_train, 
              validation_data = (x_val, y_val), 
              epochs = args.epochs, 
              batch_size = args.batchsize, 
              verbose= 2)
    end = time.time()
    print('Model traning complete, total time taken in seconds: ', round(end-start))

    # Evaluate on validation and test
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy:       {test_accuracy:.4f}")

    #save model
    model.save(args.model_path)
    print("Model saved at location: {args.model_path}")

if __name__ == "main":
    main()

