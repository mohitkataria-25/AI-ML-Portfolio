# Used Cars Price Prediction — End-to-End ML Pipeline

An end-to-end machine learning system that predicts used car prices using classical tree-based models (primary) and an optional neural network baseline.  
This project demonstrates **modular ML architecture**, **clean engineering practices**, **hyperparameter tuning**, and a **production-like inference pipeline**.

---

## 🚀 Project Highlights

### ✔️ Modular ML Codebase  
Separated into `preprocessing`, `models`, `training`, `evaluation`, `tuning`, and `inference`.

### ✔️ Multi-Model Comparison  
- Random Forest (primary model)  
- XGBoost (primary model)  
- Neural Network (Keras Sequential, experimental baseline)

### ✔️ Hyperparameter Tuning  
Built using `RandomizedSearchCV` for Random Forest and XGBoost.

### ✔️ Model Persistence  
- Sklearn & XGBoost → Joblib (`.pkl`)  
- Neural Network → Keras Save (`.h5`)

### ✔️ Inference Pipeline  
Evaluate saved models or run predictions on new samples.

---

## 📂 Repository Structure

```
used-cars-price-prediction/
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── tuning.py
│
├── metrics/          
├── models/           
├── notebooks/        
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🧠 How It Works

### 1️⃣ Preprocessing
- Numeric extraction  
- One-hot encoding  
- Train/validation/test split  
- Standardization  

### 2️⃣ Model Training
```bash
python -m src.train
```

With tuning:
```bash
python -m src.train --tune
```

### 3️⃣ Model Selection Rationale
- The dataset is **tabular, structured data** with mixed numeric and categorical features.
- In this setting, **tree-based ensembles** (Random Forest, XGBoost) typically offer the best balance of performance, robustness, and training speed.
- A small feedforward **neural network** is included as an experimental baseline, but in practice the tree-based models performed as well or better while being faster and simpler to train on CPU-only environments.
- For any deployment or production-style use, this project treats **Random Forest and XGBoost as the primary candidate models**, with the neural network used mainly for comparison and learning purposes.

---

## 🔍 Inference

```bash
python -m src.inference --model_path     models/random_forest/used_cars_rf.pkl     models/xgboost/used_cars_xgb.pkl     models/neural_net/used_cars_mlp.h5
```

---

## 📊 Example Comparison Table

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Random Forest | … | … | … |
| XGBoost | … | … | … |
| Neural Network | … | … | … |

Actual results (with Random Forest and XGBoost usually outperforming the neural network on this tabular dataset) are saved to `/metrics/`.

---

## 🛠️ Technologies

Python, Pandas, NumPy, Scikit-Learn, XGBoost, TensorFlow/Keras, Joblib, Matplotlib.

---

## 👤 Author

**Mohit Kataria**  
Senior Software Engineer • Data & ML Engineering

---

## ⭐ Star the repo if you found it helpful!
