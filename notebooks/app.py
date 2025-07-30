import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('random_forest_model.pkl')

# Title of the app
st.title("Risky Taxpayer Prediction")

# Input form
st.header("Enter Taxpayer Details")

# Updated input features based on your model
late_filings = st.number_input("Late Filings", min_value=0)
compliance_violations = st.number_input("Compliance Violations", min_value=0)
tax_liability = st.number_input("Tax Liability", min_value=0.0)
tax_paid = st.number_input("Tax Paid", min_value=0.0)
expenses = st.number_input("Expenses", min_value=0.0)
industry = st.selectbox("Industry (Encoded)", options=[0, 1, 2, 3, 4])  # Replace with real industry labels if available
audit_findings = st.number_input("Audit Findings", min_value=0)

# Prediction
if st.button("Predict Risk Level"):
    # Prepare feature array
    features = np.array([[expenses, tax_liability, tax_paid, late_filings,
                          compliance_violations, industry, audit_findings]])

    # Apply scaling if needed
    # features = scaler.transform(features)

    prediction = model.predict(features)[0]

    # Map output to label
    risk_labels = {0: "High", 1: "Low", 2: "Medium"}
    st.success(f"Predicted Risk Level: **{risk_labels.get(prediction, 'Unknown')}**")
