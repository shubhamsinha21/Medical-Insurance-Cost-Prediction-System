import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# LOAD MODEL + FEATURES
# -----------------------------

model = joblib.load("insurance_xgb_model.pkl")
features = joblib.load("model_features.pkl")

# -----------------------------
# PDF REPORT FUNCTION
# -----------------------------

def generate_pdf(data_dict, prediction):

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 750, "Insurance Prediction Report")

    pdf.setFont("Helvetica", 12)

    y = 700
    for key, value in data_dict.items():
        pdf.drawString(50, y, f"{key}: {value}")
        y -= 25

    pdf.drawString(50, y - 20, f"Predicted Cost: ₹ {prediction:,.2f}")

    pdf.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# HEADER
# -----------------------------

st.title("💰 Medical Insurance Cost Predictor")
st.markdown("### End-to-End ML App using XGBoost")
st.caption("Predict medical insurance cost + download report")

st.divider()

# -----------------------------
# LAYOUT
# -----------------------------

left, right = st.columns([2, 1])

# -----------------------------
# INPUT SECTION
# -----------------------------

with left:

    st.subheader("📥 Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 25)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

    with col2:
        children = st.number_input("Children", 0, 5, 0)
        sex = st.selectbox("Gender", ["Male", "Female"])
        smoker = st.selectbox("Smoker", ["Yes", "No"])
        
# Regionn assumed for simplicity
# st.info("Region is assumed as Southwest")

# -----------------------------
# RIGHT PANEL (PREDICTION)
# -----------------------------

with right:

    st.subheader("📊 Prediction")

    if st.button("Predict Cost 💡", use_container_width=True):

        is_female = 1 if sex == "Female" else 0
        is_smoker = 1 if smoker == "Yes" else 0
        bmi_category_obese = 1 if bmi > 30 else 0
        region_southwest = 0

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

        st.success("Prediction Completed 🎯")

        st.metric(
            label="Estimated Insurance Cost",
            value=f"₹ {prediction:,.0f}"
        )

        # store for report
        st.session_state["last_input"] = {
            "Age": age,
            "BMI": bmi,
            "Children": children,
            "Gender": sex,
            "Smoker": smoker
        }

        st.session_state["last_prediction"] = prediction

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.divider()

st.subheader("📊 Feature Importance")

if st.button("Show Feature Importance", use_container_width=True):

    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(df_imp["Feature"], df_imp["Importance"])
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance Score")

    st.pyplot(fig)

# -----------------------------
# PDF REPORT DOWNLOAD
# -----------------------------

st.divider()

st.subheader("📄 Download Report")

if "last_prediction" in st.session_state:

    pdf_buffer = generate_pdf(
        st.session_state["last_input"],
        st.session_state["last_prediction"]
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="insurance_report.pdf",
        mime="application/pdf"
    )

else:
    st.info("Run prediction first to generate report")