# Used Cars Price Prediction — End-to-End ML Pipeline

An end-to-end machine learning system that predicts used car prices using classical models and a neural network.  
This project demonstrates **modular ML architecture**, **clean engineering practices**, **hyperparameter tuning**, and a **production-like inference pipeline**.

---

## 🚀 Project Highlights

### ✔️ Modular ML Codebase  
Separated into `preprocessing`, `models`, `training`, `evaluation`, `tuning`, and `inference`.

### ✔️ Multi-Model Comparison  
- Random Forest  
- XGBoost  
- Neural Network (Keras Sequential)

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

Actual results saved to `/metrics/`.

---

## 🛠️ Technologies

Python, Pandas, NumPy, Scikit-Learn, XGBoost, TensorFlow/Keras, Joblib, Matplotlib.

---

## 👤 Author

**Mohit Kataria**  
Senior Software Engineer • Data & ML Engineering

---

## ⭐ Star the repo if you found it helpful!
