"""
Heart Disease Prediction Model Training
=========================================
Dataset  : Cleveland Heart Disease Dataset
Algorithm: Logistic Regression (binary classification)

Target column:
  0 → Healthy heart
  1 → Defective heart (heart disease present)
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ── 1. Load Dataset ──────────────────────────────────────────────────────────

DATA_PATH  = "data/heart_disease_data.csv"
MODEL_PATH = "models/heart_disease_model.sav"

heart_data = pd.read_csv(DATA_PATH)

print("── Dataset Overview ──")
print(f"Shape : {heart_data.shape}")
print(f"\nFirst 5 rows:\n{heart_data.head()}")
print(f"\nNull values:\n{heart_data.isnull().sum()}")
print(f"\nStatistical Summary:\n{heart_data.describe()}")
print(f"\nTarget Distribution:\n{heart_data['target'].value_counts()}")
print(f"  0 → Healthy Heart\n  1 → Heart Disease\n")


# ── 2. Feature & Target Split ─────────────────────────────────────────────────

X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]


# ── 3. Train / Test Split ─────────────────────────────────────────────────────

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.1,
    stratify=Y,
    random_state=1,
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}\n")


# ── 4. Train Logistic Regression Model ───────────────────────────────────────

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc  = accuracy_score(Y_test,  model.predict(X_test))

print(f"Training Accuracy : {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"Test Accuracy     : {test_acc:.4f}  ({test_acc  * 100:.2f}%)\n")


# ── 6. Quick Prediction Demo ──────────────────────────────────────────────────

sample = np.array([48, 0, 2, 130, 275, 0, 1, 139, 0, 0.2, 2, 0, 2]).reshape(1, -1)
pred = model.predict(sample)
label = "Heart Disease Detected" if pred[0] == 1 else "Healthy Heart"
print(f"Sample Prediction → {label}")


# ── 7. Save Model ─────────────────────────────────────────────────────────────

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved to '{MODEL_PATH}'")
