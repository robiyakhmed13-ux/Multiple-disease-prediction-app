"""
Multiple Disease Prediction System
=====================================
A Streamlit web app that predicts three diseases using trained ML models:
  • Diabetes          – SVM (Support Vector Machine)
  • Heart Disease     – Logistic Regression
  • Parkinson's       – SVM (Support Vector Machine)
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🏥",
    layout="wide",
)


# ── Load Trained Models ───────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    with open("models/trained_diabetes_model.sav", "rb") as f:
        diabetes_model = pickle.load(f)
    with open("models/diabetes_scaler.sav", "rb") as f:
        diabetes_scaler = pickle.load(f)

    with open("models/heart_disease_model.sav", "rb") as f:
        heart_model = pickle.load(f)

    with open("models/parkinsons_model.sav", "rb") as f:
        parkinsons_model = pickle.load(f)
    with open("models/parkinsons_scaler.sav", "rb") as f:
        parkinsons_scaler = pickle.load(f)

    return diabetes_model, diabetes_scaler, heart_model, parkinsons_model, parkinsons_scaler


diabetes_model, diabetes_scaler, heart_model, parkinsons_model, parkinsons_scaler = load_models()


# ── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    selected = option_menu(
        menu_title="Multiple Disease Prediction System",
        options=["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=["activity", "heart", "person"],
        default_index=0,
    )


# ── Helper: safe float conversion ─────────────────────────────────────────────

def to_float(value: str, field_name: str) -> float | None:
    """Convert a text-input string to float; return None on failure."""
    try:
        return float(value.strip())
    except ValueError:
        st.error(f"⚠️ Invalid value for **{field_name}** — please enter a number only.")
        return None


# ── Page 1: Diabetes Prediction ───────────────────────────────────────────────

if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction using ML")
    st.markdown("Fill in the patient details below and click **Diabetes Test Result**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies            = st.text_input("Number of Pregnancies")
    with col2:
        Glucose                = st.text_input("Glucose Level")
    with col3:
        BloodPressure          = st.text_input("Blood Pressure Value")
    with col1:
        SkinThickness          = st.text_input("Skin Thickness Value")
    with col2:
        Insulin                = st.text_input("Insulin Level")
    with col3:
        BMI                    = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    with col2:
        Age                    = st.text_input("Age of the Person")

    diab_diagnosis = ""

    if st.button("Diabetes Test Result"):
        fields = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "Blood Pressure": BloodPressure,
            "Skin Thickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "Diabetes Pedigree Function": DiabetesPedigreeFunction,
            "Age": Age,
        }
        values = {k: to_float(v, k) for k, v in fields.items()}

        if None not in values.values():
            input_array = np.array(list(values.values())).reshape(1, -1)
            input_scaled = diabetes_scaler.transform(input_array)
            prediction = diabetes_model.predict(input_scaled)
            diab_diagnosis = (
                "✅ The person is **NOT Diabetic**"
                if prediction[0] == 0
                else "⚠️ The person is **Diabetic**"
            )

    if diab_diagnosis:
        st.success(diab_diagnosis)


# ── Page 2: Heart Disease Prediction ─────────────────────────────────────────

elif selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction using ML")
    st.markdown("Fill in the patient details below and click **Heart Test Result**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age      = st.text_input("Age of the Person")
    with col2:
        sex      = st.text_input("Sex  (1 = Male, 0 = Female)")
    with col3:
        cp       = st.text_input("Chest Pain Type (0–3)")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol     = st.text_input("Serum Cholesterol (mg/dl)")
    with col3:
        fbs      = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)")
    with col1:
        restecg  = st.text_input("Resting ECG Results (0–2)")
    with col2:
        thalach  = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exang    = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)")
    with col1:
        oldpeak  = st.text_input("ST Depression Induced by Exercise")
    with col2:
        slope    = st.text_input("Slope of Peak Exercise ST Segment (0–2)")
    with col3:
        ca       = st.text_input("Major Vessels Coloured by Fluoroscopy (0–3)")
    with col1:
        thal     = st.text_input("Thal  (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect)")

    heart_diagnosis = ""

    if st.button("Heart Test Result"):
        fields = {
            "Age": age, "Sex": sex, "Chest Pain Type": cp,
            "Resting Blood Pressure": trestbps, "Cholesterol": chol,
            "Fasting Blood Sugar": fbs, "Resting ECG": restecg,
            "Max Heart Rate": thalach, "Exercise Angina": exang,
            "Oldpeak": oldpeak, "Slope": slope, "CA": ca, "Thal": thal,
        }
        values = {k: to_float(v, k) for k, v in fields.items()}

        if None not in values.values():
            input_array = np.array(list(values.values())).reshape(1, -1)
            prediction = heart_model.predict(input_array)
            heart_diagnosis = (
                "✅ The person has a **Healthy Heart**"
                if prediction[0] == 0
                else "⚠️ The person has **Heart Disease**"
            )

    if heart_diagnosis:
        st.success(heart_diagnosis)


# ── Page 3: Parkinson's Prediction ───────────────────────────────────────────

elif selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")
    st.markdown("Fill in the vocal frequency features below and click **Parkinsons Test Result**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        fo           = st.text_input("Average Vocal Fundamental Frequency")
    with col2:
        fhi          = st.text_input("Maximum Vocal Fundamental Frequency")
    with col3:
        flo          = st.text_input("Minimum Vocal Fundamental Frequency")
    with col1:
        Jitter_percent = st.text_input("MDVP:Jitter (%)")
    with col2:
        Jitter_Abs   = st.text_input("MDVP:Jitter (Abs)")
    with col3:
        RAP          = st.text_input("MDVP:RAP")
    with col1:
        PPQ          = st.text_input("MDVP:PPQ")
    with col2:
        DDP          = st.text_input("Jitter:DDP")
    with col3:
        Shimmer      = st.text_input("MDVP:Shimmer")
    with col1:
        Shimmer_dB   = st.text_input("MDVP:Shimmer (dB)")
    with col2:
        APQ3         = st.text_input("Shimmer:APQ3")
    with col3:
        APQ5         = st.text_input("Shimmer:APQ5")
    with col1:
        APQ          = st.text_input("MDVP:APQ")
    with col2:
        DDA          = st.text_input("Shimmer:DDA")
    with col3:
        NHR          = st.text_input("NHR")
    with col1:
        HNR          = st.text_input("HNR")
    with col2:
        RPDE         = st.text_input("RPDE")
    with col3:
        DFA          = st.text_input("DFA")
    with col1:
        spread1      = st.text_input("Spread1")
    with col2:
        spread2      = st.text_input("Spread2")
    with col3:
        D2           = st.text_input("D2")
    with col1:
        PPE          = st.text_input("PPE")

    parkinsons_diagnosis = ""

    if st.button("Parkinsons Test Result"):
        fields = {
            "fo": fo, "fhi": fhi, "flo": flo,
            "Jitter (%)": Jitter_percent, "Jitter Abs": Jitter_Abs,
            "RAP": RAP, "PPQ": PPQ, "DDP": DDP,
            "Shimmer": Shimmer, "Shimmer dB": Shimmer_dB,
            "APQ3": APQ3, "APQ5": APQ5, "APQ": APQ, "DDA": DDA,
            "NHR": NHR, "HNR": HNR, "RPDE": RPDE, "DFA": DFA,
            "Spread1": spread1, "Spread2": spread2, "D2": D2, "PPE": PPE,
        }
        values = {k: to_float(v, k) for k, v in fields.items()}

        if None not in values.values():
            input_array = np.array(list(values.values())).reshape(1, -1)
            input_scaled = parkinsons_scaler.transform(input_array)
            prediction = parkinsons_model.predict(input_scaled)
            parkinsons_diagnosis = (
                "✅ The person does **NOT** have Parkinson's Disease"
                if prediction[0] == 0
                else "⚠️ The person has **Parkinson's Disease**"
            )

    if parkinsons_diagnosis:
        st.success(parkinsons_diagnosis)
