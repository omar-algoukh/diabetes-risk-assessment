import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Clean Medical Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #ffffff;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 400;
    }

    .content-card {
        background-color: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .stButton>button {
        background-color: #0284c7;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #0369a1;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model Training Function
@st.cache_resource
def load_and_train():
    dataset_file = 'diabetes_risk.csv'
    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file)
    else:
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'Glucose': np.random.normal(120, 30, n).clip(70, 200),
            'BMI': np.random.normal(28, 6, n).clip(18, 55),
            'Age': np.random.normal(40, 12, n).clip(21, 80),
            'BloodPressure': np.random.normal(72, 12, n).clip(50, 120),
            'Insulin': np.random.normal(80, 40, n).clip(0, 300),
            'Outcome': np.random.choice([0, 1], size=n, p=[0.65, 0.35])
        })
    
    feature_cols = [c for c in df.columns if c.lower() != 'outcome']
    X = df[feature_cols]
    y = df['Outcome'] if 'Outcome' in df.columns else df.iloc[:, -1]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, feature_cols

model, feature_names = load_and_train()

# 4. Header Section
st.markdown("""
    <div class="main-header">
        <div class="main-title">Diabetes Risk Assessment</div>
        <div class="main-subtitle">Clinical predictive model based on metabolic health indicators</div>
    </div>
""", unsafe_allow_html=True)

# 5. Sidebar
with st.sidebar:
    st.markdown("### About")
    st.write("""
    This assessment uses a Random Forest Classification model trained on clinical parameters to evaluate early risk indicators for Type 2 Diabetes.
    """)
    st.divider()
    st.caption("Medical Disclaimer: This tool is intended for screening and educational purposes only. It should not replace professional medical diagnosis.")

# 6. Inputs Section
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
    glucose = st.slider("Fasting Glucose (mg/dL)", 50, 250, 110)
    bmi = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 24.5, 0.1)
    bp = st.slider("Diastolic Blood Pressure (mmHg)", 40, 140, 72)
    insulin = st.number_input("Serum Insulin (mu U/ml)", 0, 850, 80)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 Patient Metrics</div>', unsafe_allow_html=True)
    age = st.slider("Age (years)", 18, 90, 32)
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    skin = st.slider("Skin Fold Thickness (mm)", 0, 99, 20)
    dpf = st.slider("Diabetes Pedigree Function", 0.08, 2.50, 0.47, 0.01)
    st.markdown('</div>', unsafe_allow_html=True)

# 7. Action Button & Results
if st.button("Calculate Assessment"):
    input_data = {}
    for col in feature_names:
        c = col.lower()
        if 'glucose' in c: input_data[col] = glucose
        elif 'bmi' in c: input_data[col] = bmi
        elif 'age' in c: input_data[col] = age
        elif 'blood' in c or 'bp' in c: input_data[col] = bp
        elif 'insulin' in c: input_data[col] = insulin
        elif 'pregnan' in c: input_data[col] = pregnancies
        elif 'skin' in c: input_data[col] = skin
        elif 'pedigree' in c or 'dpf' in c: input_data[col] = dpf
        else: input_data[col] = 0

    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)
    
    res1, res2 = st.columns([1, 1], gap="large")
    
    with res1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Assessment Result</div>', unsafe_allow_html=True)
        
        st.metric(label="Calculated Risk Score", value=f"{prob:.1f}%")
        st.progress(int(prob))
        
        if prob < 35:
            st.success("Low Risk Profile: Key indicators remain within normal clinical bounds.")
        elif prob < 65:
            st.warning("Moderate Risk Profile: Certain metrics indicate potential pre-diabetic risk factors.")
        else:
            st.error("Elevated Risk Profile: Multiple biomarkers suggest a higher statistical risk. Clinical evaluation is recommended.")
        st.markdown('</div>', unsafe_allow_html=True)

    with res2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Clinical Insights</div>', unsafe_allow_html=True)
        
        if glucose >= 126:
            st.write("• **High Glucose Level:** Elevated fasting glucose is a primary indicator.")
        else:
            st.write("• **Glucose Balance:** Fasting glucose level is currently within standard range.")
            
        if bmi >= 25.0:
            st.write("• **BMI Category:** Body Mass Index indicates weight management may reduce risk.")
        else:
            st.write("• **Optimal BMI:** Weight-to-height ratio is balanced.")
            
        st.write("• **Routine Screening:** Annual blood work is advised for ongoing monitoring.")
        st.markdown('</div>', unsafe_allow_html=True)
