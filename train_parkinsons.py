"""
Parkinson's Disease Prediction Model Training
===============================================
Dataset  : Parkinson's Disease Data Set (UCI / Kaggle)
Algorithm: Support Vector Machine (SVM) with linear kernel

Target column (status):
  0 → No Parkinson's
  1 → Parkinson's Disease
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ── 1. Load Dataset ──────────────────────────────────────────────────────────

DATA_PATH  = "data/parkinsons_data.csv"
MODEL_PATH = "models/parkinsons_model.sav"

parkinsons_data = pd.read_csv(DATA_PATH)

print("── Dataset Overview ──")
print(f"Shape : {parkinsons_data.shape}")
print(f"\nFirst 5 rows:\n{parkinsons_data.head()}")
print(f"\nDataset Info:")
parkinsons_data.info()
print(f"\nNull values:\n{parkinsons_data.isnull().sum()}")
print(f"\nStatistical Summary:\n{parkinsons_data.describe()}")
print(f"\nStatus Distribution:\n{parkinsons_data['status'].value_counts()}")
print(f"  0 → No Parkinson's\n  1 → Parkinson's Disease\n")


# ── 2. Feature & Target Split ─────────────────────────────────────────────────
# Drop 'name' (non-numeric identifier) and 'status' (target)

X = parkinsons_data.drop(columns=["name", "status"], axis=1)
Y = parkinsons_data["status"]


# ── 3. Data Standardisation ───────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── 4. Train / Test Split ─────────────────────────────────────────────────────

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2,
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}\n")


# ── 5. Train SVM Classifier ───────────────────────────────────────────────────

model = SVC(kernel="linear")
model.fit(X_train, Y_train)


# ── 6. Evaluate ───────────────────────────────────────────────────────────────

train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc  = accuracy_score(Y_test,  model.predict(X_test))

print(f"Training Accuracy : {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"Test Accuracy     : {test_acc:.4f}  ({test_acc  * 100:.2f}%)\n")


# ── 7. Quick Prediction Demo ──────────────────────────────────────────────────

sample = np.array([
    91.904, 115.871, 86.292, 0.0054, 0.00006, 0.00281, 0.00336,
    0.00844, 0.02752, 0.249, 0.01424, 0.01641, 0.02214, 0.04272,
    0.01141, 21.414, 0.58339, 0.79252, -4.960234, 0.363566,
    2.642476, 0.275931,
]).reshape(1, -1)

sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)
label = "Parkinson's Disease Detected" if pred[0] == 1 else "No Parkinson's Disease"
print(f"Sample Prediction → {label}")


# ── 8. Save Model & Scaler ────────────────────────────────────────────────────

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open("models/parkinsons_scaler.sav", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✅ Model saved to '{MODEL_PATH}'")
