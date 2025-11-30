# Renewind – End-to-End ML Pipeline (Data → Train → Evaluate → Archive)
*A modular, Airflow-orchestrated, config-driven machine learning project.*

## 🚀 Overview
Renewind is a fully modular, production-style machine learning pipeline built to demonstrate real-world MLOps and engineering skills.  
It performs ingestion → preprocessing → training → evaluation → run archiving, orchestrated using Apache Airflow, and structured using an industry-style `src/` package.

This project showcases:
- Config-driven pipelines  
- Clean modular architecture  
- MLflow experiment tracking  
- Airflow DAG orchestration  
- Automatic run archiving + cleanup  
- Reusable data processing steps  
- Reproducible local runs  

The goal: predict customer renewal behavior using a neural network classifier.

## 🏗️ High-Level Architecture
```
Raw Data ──▶ Data Ingest ──▶ Preprocess ──▶ Train ──▶ Evaluate ─▶ Archive Run
                       (splits)        (scaling, weights)        (cleanup + versioning)
```

## 📦 Project Structure
```
Renewind/
  src/
    data_ingest.py
    preprocess.py
    train.py
    evaluate.py
    archive_run.py
    model_utils.py
    renewind_dag.py
  data/
    raw/
    ingested/
    processed/
  artifacts/
    current/
    archive/
  airflow/
    dags/
      renewind_training_dag.py
  mlruns/
  config.yaml
  requirements.txt
  README.md
  .gitignore
```

## ⚙️ Tech Stack
- Python
- TensorFlow / Keras  
- Scikit-Learn  
- Pandas / NumPy  
- MLflow  
- Apache Airflow  
- Matplotlib / Seaborn  

## 🧩 Pipeline Steps
### 1. Data Ingestion (`src/data_ingest.py`)
Loads raw CSV, splits into train/val/test, saves to `data/ingested/`.

### 2. Preprocessing (`src/preprocess.py`)
Scaling, class weights, saves to `data/processed/`.

### 3. Training (`src/train.py`)
Neural network training, history.json, model.h5, MLflow logging.

### 4. Evaluation (`src/evaluate.py`)
Metrics, plots, confusion matrix, saved to `artifacts/current/`.

### 5. Archiving (`src/archive_run.py`)
Archives current → archive/<timestamp>, cleans workspace.

### 6. Airflow DAG
Shells out:

```
python -m src.<module>
```

## ▶️ How to Run Locally
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place raw data as:
```
data/raw/Renewind.csv
```

Run modules:
```
python -m src.data_ingest
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.archive_run
```

## 🎯 Results
- Recall / Precision / Accuracy / F1  
- Confusion matrix  
- Loss + Recall curves  
- Model saved  
- Metrics saved  

## 👨🏻‍💻 Author
Mohit Kataria — Senior Software Engineer & AI/ML Engineer

# Renewind – End-to-End Machine Learning Pipeline (MLOps + ML Engineering)
*A production-style, Airflow-orchestrated, config-driven deep learning pipeline.*

## ⭐ Executive Summary (Recruiter-Focused)
**Renewind** is a full **real-world ML Engineering + MLOps project** showcasing:  

✔ Clean, scalable **modular architecture**  
✔ **Config-driven pipelines** like a production ML system  
✔ **Apache Airflow orchestration**  
✔ **MLflow experiment tracking**  
✔ **Automated dataset splitting, preprocessing, training & evaluation**  
✔ **Run archiving + artifact management** (model versioning behavior)  
✔ End-to-end reproducibility  

This project demonstrates **practical ML engineering**, not just training a model — the entire **pipeline** is automated and production-ready.

---

# 📐 High-Level Architecture Diagram

```
                   ┌─────────────────────────┐
                   │      Raw Dataset         │
                   │   data/raw/Renewind.csv │
                   └───────────────┬─────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │     Data Ingestion        │
                      │  (train/val/test split)   │
                      └──────────────┬────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │      Preprocessing      │
                        │ Scaling + Class Weights │
                        └──────────────┬──────────┘
                                       │
                                       ▼
                       ┌───────────────────────────┐
                       │         Training           │
                       │  NN, history, MLflow logs  │
                       └──────────────┬─────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │       Evaluation       │
                          │ Metrics, Plots, CM     │
                          └────────────┬───────────┘
                                       │
                                       ▼
                           ┌──────────────────────┐
                           │       Archive Run     │
                           │  artifacts → archive  │
                           └───────────────────────┘
```

---

# 📦 Project Structure
```
Renewind/
│
├── src/
│   ├── data_ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── archive_run.py
│   ├── model_utils.py
│   ├── renewind_dag.py
│
├── data/
│   ├── raw/
│   ├── ingested/
│   ├── processed/
│
├── artifacts/
│   ├── current/
│   ├── archive/
│
├── mlruns/
├── airflow/
│   └── dags/renewind_training_dag.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

# ⚙️ Tech Stack

### Core ML
- TensorFlow / Keras  
- Scikit-Learn  
- Pandas / NumPy  

### MLOps
- Apache Airflow  
- MLflow  
- Config-driven architecture  
- Modular pipeline  

### Visualization
- Matplotlib / Seaborn  

---

# 🧩 Pipeline Modules (with Details)

## 1. Data Ingestion – `src/data_ingest.py`
- Loads raw CSV  
- Splits into train / val / test  
- Saves clean parquet files  
- Logs MLflow parameters  
- Fully config-driven

**Output**
```
data/ingested/train.parquet
data/ingested/val.parquet
data/ingested/test.parquet
```

---

## 2. Preprocessing – `src/preprocess.py`
Includes:

### ✔ Feature Scaling  
\[
x_{scaled} = \frac{x - \mu}{\sigma}
\]

### ✔ Class Weights  
\[
w_i = \frac{N}{2 \cdot N_i}
\]

### ✔ Outputs:
```
data/processed/x_train.parquet
data/processed/y_train.parquet
data/processed/x_val.parquet
data/processed/y_val.parquet
data/processed/x_test.parquet
data/processed/y_test.parquet
```

---

## 3. Model Training – `src/train.py`

### ✔ Neural Network Architecture
```
Input(40)
Dense(128, relu)
Dense(64, relu)
Dense(32, relu)
Dense(1, sigmoid)
```

### ✔ Loss Function
Binary Crossentropy  
\[
L = -y\log(p) - (1-y)\log(1-p)
\]

### ✔ Outputs
- `model.h5`  
- `history.json`  
- MLflow logged run  

---

## 4. Evaluation – `src/evaluate.py`

### ✔ Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC  

### ✔ Confusion Matrix
\[
\begin{bmatrix}
TP & FP  \\
FN & TN  
\end{bmatrix}
\]

### ✔ Loss & Recall Curves  
Saved under:
```
artifacts/current/
```

---

## 5. Archiving – `src/archive_run.py`
Automatically:

- Moves all files from `artifacts/current/` → `artifacts/archive/<timestamp>/`
- Cleans:
  - `data/ingested/*`
  - `data/processed/*`
  - `artifacts/current/*`

This is equivalent to **model versioning**.

---

## 6. Airflow DAG – `renewind_training_dag.py`
Runs the full pipeline:

```
data_ingest
    ↓
preprocess
    ↓
train_model
    ↓
evaluate_model
    ↓
archive_run
```

Each task executes:

```
python -m src.<module>
```

---

# ▶️ How to Run Locally

### 1. Setup Environment
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Provide Raw File
```
data/raw/Renewind.csv
```

### 3. Run Pipeline
```
python -m src.data_ingest
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.archive_run
```

---

# 📊 Sample Results

| Metric | Value |
|--------|--------|
| Accuracy | ~0.83 |
| Recall | ~0.87 |
| Precision | ~0.81 |
| F1 Score | ~0.83 |

### Confusion Matrix (example)
```
[[4500   92]
 [ 811  620]]
```

---

# 🧠 Deep ML Explanation

## Why Scaling?
Prevents gradient instability.

\[
\mu = 0,\;\sigma = 1
\]

## Why Class Weights?
Balances imbalanced datasets.

\[
Loss = w_1 L_1 + w_0 L_0
\]

## Why ReLU?
\[
f(x) = \max(0, x)
\]

- Avoids vanishing gradients  
- Fast training  

## Why Sigmoid Output?
Probability output for binary classification.

\[
p = \sigma(z)
\]

## Why MLflow?
- Reproducibility  
- Experiment comparison  
- Artifact storage  

---

# 🧑🏻‍💻 Author
**Mohit Kataria**  
Senior Software Engineer • Data/ML Engineer  
Austin, TX
