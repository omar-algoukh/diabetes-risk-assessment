from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

# Base path setup
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes_risk.csv"

# Model features definitions
NUMERIC_COLUMNS = [
    "age", "bmi", "hours_sleep_per_night", "stress_level",
    "fasting_blood_sugar", "hba1c_level", "blood_pressure_systolic",
    "blood_pressure_diastolic", "waist_circumference_cm",
    "pulse_pressure", "sugar_interaction",
]
ONE_HOT_COLUMNS = ["gender", "family_history_diabetes", "diet_type", "smoking_status"]
ORDINAL_COLUMNS = ["physical_activity_level"]

def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived variables used during model training."""
    result = frame.copy()
    result["pulse_pressure"] = (
        result["blood_pressure_systolic"] - result["blood_pressure_diastolic"]
    )
    result["sugar_interaction"] = result["fasting_blood_sugar"] * result["hba1c_level"]
    return result

@st.cache_resource(show_spinner="Training predictive model...")
def train_model():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    
    # Drop non-predictive columns if present
    cols_to_drop = [c for c in ["patient_id", "city", "income_bracket", "alcohol_consumption"] if c in data.columns]
    data = data.drop(columns=cols_to_drop)
    
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

# 1. Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Modern Clean Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    /* Top Branding Header */
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
    }
    .brand-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }

    /* Section Cards */
    .card-box {
        background: #ffffff;
        padding: 1.8rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    .card-heading {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Input styling refinement */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 500 !important;
        color: #334155 !important;
        font-size: 0.9rem !important;
    }

    /* Primary Action Button */
    .stButton>button {
        background: #0284c7;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
    }
    .stButton>button:hover {
        background: #0369a1;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35);
    }

    /* Result Card Styling */
    .result-container {
        padding: 1.8rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    .res-low { background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); }
    .res-mod { background: linear-gradient(135deg, #d97706 0%, #b45309 100%); }
    .res-high { background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); }
</style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown("""
    <div class="brand-header">
        <h1 class="brand-title">🩺 Diabetes Risk Analytics</h1>
        <p class="brand-subtitle">Clinical prediction tool powered by machine learning and metabolic biomarkers</p>
    </div>
""", unsafe_allow_html=True)

# 4. Sidebar Information
with st.sidebar:
    st.markdown("### 📊 Model Architecture")
    st.write("""
    This platform utilizes a **Random Forest Classifier (300 estimators)** with automated data imputation, robust scaling, and derived biomarker interactions.
    """)
    st.divider()
    st.caption("🔒 Privacy Notice: Input data is processed in real-time and is not retained or stored.")
    st.caption("⚠️ Disclaimer: For screening purposes only. Consult a clinician for diagnostic testing.")

# 5. Data Entry Form
with st.form("risk_form", border=False):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">👤 Demographics & Lifestyle</div>', unsafe_allow_html=True)
        
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        family = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "Active"])
        diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan", "Pescatarian"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        sleep = st.number_input("Average Sleep per Night (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🔬 Clinical Biomarkers</div>', unsafe_allow_html=True)
        
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        waist = st.number_input("Waist Circumference (cm)", min_value=30.0, max_value=200.0, value=90.0, step=0.1)
        fasting_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=30.0, max_value=500.0, value=100.0, step=1.0)
        hba1c = st.number_input("HbA1c Level (%)", min_value=2.0, max_value=20.0, value=5.5, step=0.1)
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60.0, max_value=250.0, value=120.0, step=1.0)
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30.0, max_value=160.0, value=80.0, step=1.0)
        
        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Generate Risk Assessment", type="primary")

# 6. Model Execution & Presentation
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
    enriched_input = enrich_features(input_data)
    
    prediction = str(model.predict(enriched_input)[0])
    probabilities = model.predict_proba(enriched_input)[0]
    confidence = max(probabilities) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🎯 Estimated Risk Outcome</div>', unsafe_allow_html=True)
        
        risk_class = "res-low"
        if "mod" in prediction.lower(): risk_class = "res-mod"
        elif "high" in prediction.lower(): risk_class = "res-high"
        
        st.markdown(f"""
            <div class="result-container {risk_class}">
                <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Predicted Risk Level</div>
                <div style="font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0;">{prediction}</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">Model Certainty Score: {confidence:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">💡 Biomarker Observations</div>', unsafe_allow_html=True)
        
        if fasting_sugar >= 126 or hba1c >= 6.5:
            st.warning("⚠️ **Glycemic Indicators:** Glycemic markers are elevated above optimal thresholds.")
        else:
            st.success("✅ **Glycemic Indicators:** Fasting glucose and HbA1c are within standard ranges.")

        if bmi >= 25.0:
            st.info("ℹ️ **BMI Metrics:** Weight management strategies may help mitigate long-term risk.")
        else:
            st.success("✅ **BMI Metrics:** Body Mass Index is within standard bounds.")

        st.markdown('</div>', unsafe_allow_html=True)
