# MNIST Handwritten Digit Classifier (Feedforward Neural Network)

This project implements a **feedforward neural network** (Multi-Layer Perceptron) to classify handwritten digits from the classic **MNIST** dataset using **TensorFlow / Keras**.

It started as an exploratory Jupyter notebook and was refactored into a more **production-like, modular Python project** to better reflect real-world ML engineering practices:

- Clear **project structure** (`src/`, `notebooks/`, `models/`, `plots/`)
- Separate **data preprocessing**, **model definition**, **training**, and **evaluation**
- Easy to **reproduce experiments** and **extend architectures**

---

## 📊 Problem Overview

- **Task:** Multi-class classification (digits 0–9)
- **Dataset:** [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/)
- **Input:** 28 × 28 grayscale images (flattened to 784-dimensional vectors)
- **Output:** Probability distribution over 10 digit classes

The goal is to build and compare several fully-connected architectures:
- No hidden layer (softmax regression baseline)
- Single hidden layer
- Multiple hidden layers with different activation functions and hyperparameters

---

## 🧠 Approach

1. **Data Loading & Preprocessing**
   - Load MNIST via `keras.datasets.mnist`
   - Flatten `28×28` images to vectors of size `784`
   - Normalize pixel values to `[0, 1]`
   - Split the original training set into **train** and **validation** using `train_test_split`
   - One-hot encode labels using `keras.utils.to_categorical`

2. **Model Architecture**
   - Baseline: softmax regression (no hidden layer)
   - Multi-layer perceptron (MLP) variants, e.g.:

     ```python
     model = Sequential()
     model.add(Dense(128, activation="relu", input_dim=input_dim))
     model.add(Dense(64, activation="relu"))
     model.add(Dense(32, activation="relu"))
     model.add(Dense(num_classes, activation="softmax"))
     ```

   - Experiments with:
     - Number of hidden layers (0, 1, 2, 3)
     - Neurons per layer (e.g., 64, 128, 32...)
     - Activation functions (`relu`, `tanh`, `sigmoid`)
     - Batch size and number of epochs

3. **Training**
   - Loss: `categorical_crossentropy`
   - Optimizer: `sgd` (stochastic gradient descent)
   - Metrics: `accuracy`
   - Track:
     - Training loss & accuracy
     - Validation loss & accuracy
     - Total training time

4. **Evaluation**
   - Evaluate best model on the **test** set
   - Compare **train vs. validation performance**
   - Summarize experiments in a `pandas` DataFrame (e.g., number of layers, neurons, activations, accuracy, runtime)

> Note: A simple MLP with a few dense layers and ReLU activations typically reaches **high accuracy on MNIST** (often > 97% test accuracy, depending on hyperparameters and hardware).

---

## 🗂 Project Structure

```text
mnist-feedforward-classifier/
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── MNIST-Dataset-Practice.ipynb      # original exploration + EDA + experiments
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                     # data loading, flattening, normalization, splits
│   ├── model.py                          # model builder function(s)
│   ├── train.py                          # training script (CLI)
│   └── evaluate.py                       # evaluation utilities
│
├── models/
│   └── mnist_mlp.h5                      # saved Keras model(s)
│
└── plots/
    ├── training_curves.png               # loss/accuracy vs epochs
    └── confusion_matrix.png              # (optional)
