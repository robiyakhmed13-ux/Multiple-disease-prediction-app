# 🏥 Multiple Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

<br/>

**An intelligent, ML-powered web application that predicts the likelihood of three critical diseases — Diabetes, Heart Disease, and Parkinson's — through an interactive, easy-to-use interface.**

<br/>

[🚀 Run the App](#-getting-started) · [📊 Models](#-ml-models) · [📁 Project Structure](#-project-structure) · [🤝 Contributing](#-contributing)

---

</div>

## ✨ Features

- 🩸 **Diabetes Prediction** — SVM model trained on the PIMA Indians Diabetes Dataset
- ❤️ **Heart Disease Prediction** — Logistic Regression trained on the Cleveland Heart Disease Dataset
- 🧠 **Parkinson's Disease Detection** — SVM trained on vocal biomarker data from UCI/Kaggle
- 🖥️ **Clean Streamlit UI** — Sidebar navigation, column layouts, live validation, and instant results
- 🛡️ **Input Validation** — Friendly error messages when non-numeric values are entered
- ⚡ **Model Caching** — Models are loaded once and cached for fast response times

---

## 📸 Demo

| Diabetes Page | Heart Disease Page | Parkinson's Page |
|:---:|:---:|:---:|
| ![Diabetes](https://via.placeholder.com/280x180?text=Diabetes+Page) | ![Heart](https://via.placeholder.com/280x180?text=Heart+Disease+Page) | ![Parkinson](https://via.placeholder.com/280x180?text=Parkinson%27s+Page) |

> 💡 Replace the placeholder images above with screenshots of your running app.

---

## 📊 ML Models

| Disease | Algorithm | Dataset | Training Accuracy | Test Accuracy |
|---|---|---|---|---|
| 🩸 Diabetes | Support Vector Machine (SVM) | PIMA Indians Diabetes | ~79% | ~77% |
| ❤️ Heart Disease | Logistic Regression | Cleveland Heart Disease | ~85% | ~82% |
| 🧠 Parkinson's | Support Vector Machine (SVM) | UCI Parkinson's Dataset | ~88% | ~87% |

### Why these algorithms?

- **SVM (linear kernel)** — Excellent for high-dimensional, small-to-medium datasets with a clear margin of separation. Ideal for the diabetes and Parkinson's datasets.
- **Logistic Regression** — The gold standard for binary classification. Interpretable, efficient, and well-suited to the heart disease dataset's structured features.

---

## 📁 Project Structure

```
multiple-disease-prediction/
│
├── app.py                      # Main Streamlit application
│
├── train_diabetes.py           # Train & save the diabetes SVM model
├── train_heart_disease.py      # Train & save the heart disease model
├── train_parkinsons.py         # Train & save the Parkinson's SVM model
│
├── models/                     # Saved model & scaler files (.sav)
│   ├── trained_diabetes_model.sav
│   ├── diabetes_scaler.sav
│   ├── heart_disease_model.sav
│   ├── parkinsons_model.sav
│   └── parkinsons_scaler.sav
│
├── data/                       # Raw CSV datasets (not committed to git)
│   ├── diabetes.csv
│   ├── heart_disease_data.csv
│   └── parkinsons_data.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multiple-disease-prediction.git
cd multiple-disease-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Datasets

Place the following CSV files inside the `data/` folder:

| File | Source |
|---|---|
| `diabetes.csv` | [Kaggle – PIMA Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| `heart_disease_data.csv` | [Kaggle – Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) |
| `parkinsons_data.csv` | [Kaggle – Parkinson's Disease Data Set](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) |

### 5. Train the Models

Run each training script once to generate the `.sav` model files:

```bash
python train_diabetes.py
python train_heart_disease.py
python train_parkinsons.py
```

### 6. Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 🧪 Input Features Guide

<details>
<summary>🩸 <strong>Diabetes Inputs</strong></summary>

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg / height in m²) |
| Diabetes Pedigree Function | Likelihood of diabetes based on family history |
| Age | Age in years |

</details>

<details>
<summary>❤️ <strong>Heart Disease Inputs</strong></summary>

| Feature | Description |
|---|---|
| Age | Age in years |
| Sex | 1 = Male, 0 = Female |
| Chest Pain Type | 0 = Typical angina … 3 = Asymptomatic |
| Resting Blood Pressure | On admission (mm Hg) |
| Serum Cholesterol | In mg/dl |
| Fasting Blood Sugar | 1 if > 120 mg/dl, else 0 |
| Resting ECG | 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy |
| Max Heart Rate | Maximum heart rate achieved |
| Exercise Induced Angina | 1 = Yes, 0 = No |
| Oldpeak | ST depression induced by exercise |
| Slope | Slope of peak exercise ST segment |
| CA | Major vessels coloured by fluoroscopy (0–3) |
| Thal | 0 = Normal, 1 = Fixed defect, 2 = Reversable defect |

</details>

<details>
<summary>🧠 <strong>Parkinson's Inputs</strong></summary>

| Feature | Description |
|---|---|
| fo, fhi, flo | Average / max / min vocal fundamental frequency |
| Jitter (%), Jitter Abs | Measures of variation in fundamental frequency |
| RAP, PPQ, DDP | Jitter-related measures |
| Shimmer, Shimmer dB | Measures of variation in amplitude |
| APQ3, APQ5, APQ, DDA | Shimmer-related measures |
| NHR, HNR | Noise-to-harmonics / harmonics-to-noise ratio |
| RPDE, DFA | Nonlinear dynamical complexity measures |
| Spread1, Spread2, D2 | Nonlinear measures of fundamental frequency variation |
| PPE | Pitch period entropy |

</details>

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | [Streamlit](https://streamlit.io) |
| ML Models | [Scikit-learn](https://scikit-learn.org) |
| Data Processing | [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Model Persistence | Python `pickle` |
| Language | Python 3.8+ |

---

## ⚠️ Disclaimer

> This application is built for **educational and research purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and Python

⭐ Star this repo if you found it useful!

</div>
