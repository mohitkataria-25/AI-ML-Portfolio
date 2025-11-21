# Used Car Price Prediction (Regression with Neural Networks)

This project predicts **used car prices** using a **feedforward neural network (MLP)** built with **TensorFlow / Keras**. It is refactored from an exploratory Jupyter notebook into a **modular, runnable ML project** suitable for showcasing to recruiters.

Key aspects:

- End-to-end ML workflow: data loading → cleaning → feature engineering → train/val/test split → scaling → model training → evaluation
- Clear code structure under `src/` (preprocessing, model, training, evaluation)
- Easily reproducible with CLI-based training scripts

---

## 📊 Problem Overview

- **Task:** Supervised regression – predict the selling price of a used car  
- **Input:** Tabular data with features such as brand, model, kilometers driven, fuel type, transmission, seats, mileage, engine, power, etc.  
- **Output:** Predicted price (continuous value)

---

## 🧠 Approach

### 1. Data Loading & Cleaning

- Load the dataset from `data/used_cars_data.csv`
- Drop rows with missing `Price`
- Handle duplicates
- Optionally engineer numeric features from string columns, if present:
  - `Mileage` → `mileage_num`
  - `Engine` → `engine_num`
  - `Power` → `power_num`

### 2. Feature Engineering

- Separate target: `Price`
- One-hot encode categorical variables (brand, model, fuel type, transmission, etc.)
- Standardize numeric features using `StandardScaler` (fit on training data only)

### 3. Data Splits

- Split data into **train / validation / test** sets using `train_test_split` with a fixed random seed for reproducibility.

### 4. Model

- Use a **Multi-Layer Perceptron (MLP)** regression model built with `tensorflow.keras`:

  - Several hidden layers with ReLU activation
  - Final layer: 1 neuron (linear) for regression

- Loss: **Mean Squared Error (MSE)**
- Metrics: **Mean Absolute Error (MAE)**, optional **R²**

### 5. Training & Evaluation

- Train the model on the training set, monitor validation performance
- Evaluate on the held-out test set
- Report metrics such as:
  - MAE
  - RMSE
  - R² score

---

## 🗂 Project Structure

```text
used-cars-price-prediction/
├── README.md
├── requirements.txt
│
├── data/
│   └── used_cars_data.csv
│
├── notebooks/
│   └── UsedCars.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py      # data loading, cleaning, encoding, scaling, splitting
│   ├── model.py           # build MLP regression model
│   ├── train.py           # CLI training script
│   └── evaluate.py        # CLI evaluation script
│
└── models/
    └── used_cars_mlp.h5   # saved Keras model
