import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the trained model pipeline ---
try:
    pipeline = joblib.load("heart_disease_model.joblib")
    model = pipeline['model']
    scaler = pipeline['scaler']
    label_encoders = pipeline['label_encoders']
    feature_names = pipeline['feature_names']
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'heart_disease_model.joblib' exists.")
    st.stop()

# --- Input preprocessing ---
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Label encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
        else:
            st.error(f"Missing input for column: {col}")
            st.stop()

    # Feature engineering
    df['Framingham_Risk_Factors'] = (
        (df['Age'] >= 45).astype(int) +
        (df['Sex'] == 1).astype(int) +
        (df['Cholesterol'] >= 240).astype(int) +
        (df['RestingBP'] >= 140).astype(int) +
        df['FastingBS']
    )
    df['Predicted_Max_HR'] = 220 - df['Age']
    df['HR_Achievement_Ratio'] = df['MaxHR'] / df['Predicted_Max_HR']
    df['Age_Cholesterol_Product'] = df['Age'] * df['Cholesterol'] / 1000
    df['Metabolic_Risk_Score'] = (
        df['Cholesterol'] / 200 +
        df['RestingBP'] / 120 +
        df['FastingBS']
    )

    try:
        df = df[feature_names]
    except KeyError as e:
        st.error(f"Feature mismatch. Missing: {e}")
        st.stop()

    return scaler.transform(df)

# --- UI Setup ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("🩺 Heart Disease Risk Prediction")

st.sidebar.header("Patient Information")

# --- Inputs ---
age = st.sidebar.slider("Age", 29, 77, 54)
sex = st.sidebar.radio("Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 200)
fasting_bs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.sidebar.radio("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.sidebar.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Prediction ---
if st.sidebar.button("🔍 Predict Risk"):
    user_input = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    processed_input = preprocess_input(user_input)
    prediction_proba = model.predict_proba(processed_input)[0, 1]

    st.subheader("📊 Prediction Result")
    st.metric(label="Risk of Heart Disease", value=f"{prediction_proba:.2%}")

    if prediction_proba > 0.5:
        st.error("⚠️ The model predicts a **high risk** of heart disease.")
    else:
        st.success("✅ The model predicts a **low risk** of heart disease.")
