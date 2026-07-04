"""Optional Streamlit demo for single-customer churn prediction."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.features import engineer_features
from src.data_loader import clean_data
from src.utils import MODELS_DIR, REPORTS_DIR

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Predictor")

model_path = MODELS_DIR / "best_model.pkl"
preprocessor_path = MODELS_DIR / "preprocessor.pkl"
metrics_path = REPORTS_DIR / "metrics_summary.csv"

if not model_path.exists() or not preprocessor_path.exists():
    st.warning("Run `python main.py --with-smote` before launching the demo.")
    st.stop()

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

if metrics_path.exists():
    st.sidebar.subheader("Model comparison")
    st.sidebar.dataframe(pd.read_csv(metrics_path), hide_index=True)

uploaded = st.file_uploader("Upload customers CSV", type=["csv"])
if uploaded:
    data = clean_data(pd.read_csv(uploaded))
else:
    data = pd.DataFrame(
        [
            {
                "customerID": "FORM001",
                "gender": st.selectbox("Gender", ["Female", "Male"]),
                "SeniorCitizen": st.selectbox("Senior citizen", ["No", "Yes"]),
                "Partner": st.selectbox("Partner", ["Yes", "No"]),
                "Dependents": st.selectbox("Dependents", ["No", "Yes"]),
                "tenure": st.slider("Tenure", 0, 72, 12),
                "PhoneService": "Yes",
                "MultipleLines": st.selectbox("Multiple lines", ["No", "Yes", "No phone service"]),
                "InternetService": st.selectbox("Internet service", ["DSL", "Fiber optic", "No"]),
                "OnlineSecurity": st.selectbox("Online security", ["No", "Yes", "No internet service"]),
                "OnlineBackup": st.selectbox("Online backup", ["No", "Yes", "No internet service"]),
                "DeviceProtection": st.selectbox("Device protection", ["No", "Yes", "No internet service"]),
                "TechSupport": st.selectbox("Tech support", ["No", "Yes", "No internet service"]),
                "StreamingTV": st.selectbox("Streaming TV", ["No", "Yes", "No internet service"]),
                "StreamingMovies": st.selectbox("Streaming movies", ["No", "Yes", "No internet service"]),
                "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
                "PaperlessBilling": st.selectbox("Paperless billing", ["Yes", "No"]),
                "PaymentMethod": st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                ),
                "MonthlyCharges": st.number_input("Monthly charges", 18.0, 130.0, 70.0),
                "TotalCharges": st.number_input("Total charges", 0.0, 9000.0, 840.0),
            }
        ]
    )

if st.button("Predict churn"):
    features = engineer_features(data.drop(columns=["Churn"], errors="ignore"))
    x = preprocessor.transform(features.drop(columns=["customerID"], errors="ignore"))
    probability = model.predict_proba(x)[:, 1]
    result = data.copy()
    result["churn_probability"] = probability
    st.dataframe(result[["customerID", "churn_probability"]], hide_index=True)
