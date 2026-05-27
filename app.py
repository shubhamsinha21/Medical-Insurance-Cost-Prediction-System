import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load model + features
# -----------------------------

model = joblib.load("insurance_xgb_model.pkl")
features = joblib.load("model_features.pkl")

# -----------------------------
# Page config (IMPORTANT for UI)
# -----------------------------

st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="💰",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------

st.title("💰 Insurance Charges Predictor")
st.markdown("Predict medical insurance cost using Machine Learning")

st.divider()

# -----------------------------
# Input Section (clean layout)
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 25)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

with col2:
    children = st.number_input("Children", 0, 5, 0)
    sex = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])

# -----------------------------
# Encoding
# -----------------------------

is_female = 1 if sex == "Female" else 0
is_smoker = 1 if smoker == "Yes" else 0
bmi_category_obese = 1 if bmi > 30 else 0
region_southwest = 0

# -----------------------------
# Prediction button
# -----------------------------

if st.button("Predict Insurance Cost 💡"):

    input_data = pd.DataFrame([[
        age,
        is_female,
        bmi,
        children,
        is_smoker,
        region_southwest,
        bmi_category_obese
    ]], columns=features)

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Charges: ₹ {prediction:,.2f}")

    st.info("Prediction generated using trained XGBoost model")
    
st.divider()
st.subheader("📊 Feature Importance (Model Insight)")

if st.button("Show Feature Importance"):

    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"]
    )

    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)