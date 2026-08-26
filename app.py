from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes_risk.csv"

NUMERIC_COLUMNS = [
    "age", "bmi", "hours_sleep_per_night", "stress_level",
    "fasting_blood_sugar", "hba1c_level", "blood_pressure_systolic",
    "blood_pressure_diastolic", "waist_circumference_cm",
    "pulse_pressure", "sugar_interaction",
]
ONE_HOT_COLUMNS = ["gender", "family_history_diabetes", "diet_type", "smoking_status"]
ORDINAL_COLUMNS = ["physical_activity_level"]


def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the two derived variables used during model training."""
    result = frame.copy()
    result["pulse_pressure"] = (
        result["blood_pressure_systolic"] - result["blood_pressure_diastolic"]
    )
    result["sugar_interaction"] = result["fasting_blood_sugar"] * result["hba1c_level"]
    return result


@st.cache_resource(show_spinner="Preparing the prediction model for the first time...")
def train_model():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    data = data.drop(columns=["patient_id", "city", "income_bracket", "alcohol_consumption"])
    data = enrich_features(data)

    features = data.drop(columns="diabetes_risk")
    target = data["diabetes_risk"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    activity_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, ONE_HOT_COLUMNS),
        ("activity", activity_pipeline, ORDINAL_COLUMNS),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(features, target)
    return model


def risk_style(label: str) -> tuple[str, str]:
    styles = {
        "Low": ("Low", "#15803d"),
        "Moderate": ("Moderate", "#d97706"),
        "High": ("High", "#dc2626"),
    }
    return styles.get(label, (label, "#334155"))


st.set_page_config(page_title="Diabetes Risk Assessment", page_icon="🩺", layout="centered")
st.markdown(
    """<style>
    .stApp { background: #f6fbfa; }
    #MainMenu, footer, header { visibility: hidden; }
    .hero { background: linear-gradient(135deg,#075e54,#0f766e); color:white;
      padding:2rem; border-radius:18px; margin-bottom:1.5rem; }
    .hero h1, .hero p { color:white; margin:0; }
    .result { padding:1.5rem; border-radius:16px; text-align:center; color:white;
      font-size:1.3rem; font-weight:700; margin:1rem 0; }
    </style>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div class="hero">
    <h1>🩺 Diabetes Risk Assessment</h1>
    <p>Enter your health information below to receive an instant risk estimate.</p>
    </div>""",
    unsafe_allow_html=True,
)

with st.form("risk_form", border=False):
    st.subheader("Personal Information & Lifestyle")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        family = st.selectbox("Family history of diabetes?", ["No", "Yes"])
        activity = st.selectbox("Physical activity level", ["Sedentary", "Moderate", "Active"])
        diet = st.selectbox("Diet type", ["Vegetarian", "Non-Vegetarian", "Vegan", "Pescatarian"])
    with col2:
        bmi = st.number_input("Body mass index (BMI)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        sleep = st.number_input("Average sleep per night (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        stress = st.slider("Stress level (1–10)", 1, 10, 5)
        smoking = st.selectbox("Smoking status", ["Never", "Former", "Current"])
        waist = st.number_input("Waist circumference (cm)", min_value=30.0, max_value=200.0, value=90.0, step=0.1)

    st.subheader("Measurements & Tests")
    col3, col4 = st.columns(2)
    with col3:
        fasting_sugar = st.number_input("Fasting blood sugar (mg/dL)", min_value=30.0, max_value=500.0, value=100.0, step=1.0)
        systolic = st.number_input("Systolic blood pressure", min_value=60.0, max_value=250.0, value=120.0, step=1.0)
    with col4:
        hba1c = st.number_input("HbA1c (%)", min_value=2.0, max_value=20.0, value=5.5, step=0.1)
        diastolic = st.number_input("Diastolic blood pressure", min_value=30.0, max_value=160.0, value=80.0, step=1.0)

    submitted = st.form_submit_button("Assess risk", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "age": age, "gender": gender, "bmi": bmi,
        "family_history_diabetes": family, "physical_activity_level": activity,
        "diet_type": diet, "smoking_status": smoking,
        "hours_sleep_per_night": sleep, "stress_level": stress,
        "fasting_blood_sugar": fasting_sugar, "hba1c_level": hba1c,
        "blood_pressure_systolic": systolic,
        "blood_pressure_diastolic": diastolic,
        "waist_circumference_cm": waist,
    }])
    model = train_model()
    prediction = model.predict(enrich_features(input_data))[0]
    probabilities = model.predict_proba(enrich_features(input_data))[0]
    confidence = max(probabilities) * 100
    english_label, color = risk_style(prediction)
    st.markdown(
        f'<div class="result" style="background:{color}">'
        f'Predicted risk level: {english_label}<br>'
        f'<span style="font-size:.85rem;font-weight:400">Model confidence: {confidence:.1f}%</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("This is an estimate generated by a machine-learning model and is not a medical diagnosis. Please consult a healthcare professional with any concerns.")

st.divider()
st.caption("Your information is not stored by this application.")
