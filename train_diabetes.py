"""
Diabetes Prediction Model Training
====================================
Dataset  : PIMA Indians Diabetes Dataset
Algorithm: Support Vector Machine (SVM) with linear kernel
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ── 1. Load Dataset ──────────────────────────────────────────────────────────

DATA_PATH  = "data/diabetes.csv"
MODEL_PATH = "models/trained_diabetes_model.sav"

diabetes_data = pd.read_csv(DATA_PATH)

print("── Dataset Overview ──")
print(f"Shape : {diabetes_data.shape}")
print(f"\nFirst 5 rows:\n{diabetes_data.head()}")
print(f"\nStatistical Summary:\n{diabetes_data.describe()}")
print(f"\nOutcome Distribution:\n{diabetes_data['Outcome'].value_counts()}")
print(f"  0 → Non-Diabetic\n  1 → Diabetic\n")


# ── 2. Feature & Target Split ─────────────────────────────────────────────────

X = diabetes_data.drop(columns="Outcome", axis=1)
Y = diabetes_data["Outcome"]


# ── 3. Data Standardisation ───────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── 4. Train / Test Split ─────────────────────────────────────────────────────

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y,
    test_size=0.2,
    stratify=Y,
    random_state=42,
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}\n")


# ── 5. Train SVM Classifier ───────────────────────────────────────────────────

classifier = SVC(kernel="linear")
classifier.fit(X_train, Y_train)


# ── 6. Evaluate ───────────────────────────────────────────────────────────────

train_acc = accuracy_score(Y_train, classifier.predict(X_train))
test_acc  = accuracy_score(Y_test,  classifier.predict(X_test))

print(f"Training Accuracy : {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"Test Accuracy     : {test_acc:.4f}  ({test_acc  * 100:.2f}%)\n")


# ── 7. Quick Prediction Demo ──────────────────────────────────────────────────

sample = np.array([4, 110, 92, 0, 0, 37.6, 0.191, 30]).reshape(1, -1)
sample_scaled = scaler.transform(sample)
pred = classifier.predict(sample_scaled)
label = "Diabetic" if pred[0] == 1 else "Not Diabetic"
print(f"Sample Prediction → {label}")


# ── 8. Save Model & Scaler ────────────────────────────────────────────────────

with open(MODEL_PATH, "wb") as f:
    pickle.dump(classifier, f)

with open("models/diabetes_scaler.sav", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✅ Model saved to '{MODEL_PATH}'")
