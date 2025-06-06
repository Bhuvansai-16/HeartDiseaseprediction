import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Apply custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #ff4b4b;
    }
    .description {
        text-align: center;
        color: #cccccc;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">❤️ Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Estimate your heart disease risk with 13 health parameters.</div>', unsafe_allow_html=True)

# Columns for layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45, help="Your age in years")
    sex = st.selectbox("Sex", ["Male", "Female"], help="Biological sex")
    sex_val = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"
    ], help="Type of chest pain")
    cp_val = int(cp.split("(")[-1][0])

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Resting BP in mm Hg")
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240, help="Cholesterol level in mg/dl")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"], help="Is fasting blood sugar > 120?")
    fbs_val = int(fbs.split("(")[-1][0])

    restecg = st.selectbox("Resting ECG Result", [
        "Normal (0)", "ST-T Wave Abnormality (1)", "Left Ventricular Hypertrophy (2)"
    ], help="ECG result")
    restecg_val = int(restecg.split("(")[-1][0])

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 60, 250, 150, help="Max heart rate during exercise")
    exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"], help="Pain during exercise?")
    exang_val = int(exang.split("(")[-1][0])

    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise")
    slope = st.selectbox("Slope of ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"], help="ST segment slope")
    slope_val = int(slope.split("(")[-1][0])

    ca = st.slider("Number of Major Vessels Colored (ca)", 0, 3, 0, help="Number of vessels (0–3)")
    thal = st.selectbox("Thalassemia Type", ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"], help="Thalassemia type")
    thal_val = int(thal.split("(")[-1][0])

# Button
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                            thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # Confidence

    if prediction == 1:
        st.success(f"✅ You are likely at **Low risk** of heart disease. (Confidence: {prob:.2%})")
    else:
        st.error(f"⚠️ You may be at **High risk** of heart disease. (Confidence: {1 - prob:.2%})")
