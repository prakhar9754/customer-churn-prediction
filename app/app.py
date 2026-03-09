import sys
import os

# Allow importing from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.predict import predict_churn  # type: ignore

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Customer Churn Prediction App")

st.markdown("""
This app predicts whether a customer is likely to **churn** based on their purchasing behavior.

Enter the customer metrics below and click **Predict Churn**.
""")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    recency = st.number_input(
        "Recency (Days since last purchase)",
        min_value=0
    )

    frequency = st.number_input(
        "Number of Orders",
        min_value=0
    )

    monetary = st.number_input(
        "Total Spend",
        min_value=0.0
    )

with col2:
    avg_order_value = st.number_input(
        "Average Order Value",
        min_value=0.0
    )

    avg_review_score = st.slider(
        "Average Review Score",
        min_value=1.0,
        max_value=5.0,
        value=4.0
    )

# Prediction button
if st.button("🔍 Predict Churn"):

    result = predict_churn(
        recency,
        frequency,
        monetary,
        avg_order_value,
        avg_review_score
    )

    if result == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

# Sidebar
st.sidebar.info(
"""
### About This Project

This project uses **Machine Learning** to predict customer churn.

**Model:** Random Forest  
**Features:** RFM + Behavioral Metrics  
**Built with:**  
- Python  
- Scikit-learn  
- Streamlit
"""
)