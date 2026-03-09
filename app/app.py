import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.predict import predict_churn# type:ignore
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")

st.markdown("""
This app predicts whether a customer is likely to **churn** based on their account and usage data.

Enter the customer information below and click **Predict**.
""")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", 0, 72)
    monthly_charges = st.number_input("Monthly Charges", 0.0)

with col2:
    total_charges = st.number_input("Total Charges", 0.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    if st.button("🔍 Predict Churn"):
        result = predict_churn(tenure, monthly_charges, total_charges, contract)

    if result == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
        st.sidebar.header("About")

st.sidebar.info(
"""
This project uses **Machine Learning** to predict customer churn.

Model: Random Forest  
Dataset: Telco Customer Churn  
Built with: Python, Scikit-learn, Streamlit
"""
)
