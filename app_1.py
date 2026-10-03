from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(
    page_title="Customer Retention Intelligence",
    layout="wide"
)

# Load model artifacts from the repository's model directory.
MODEL_DIR = Path(__file__).resolve().parent / "model"
model = joblib.load(MODEL_DIR / "xgb_best_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

# Page title


st.title("Customer Retention Intelligence")
st.write("Estimate customer churn risk and identify the key factors driving the prediction.")
st.divider()

# two sided layout

left_col, right_col = st.columns([2, 1])

with left_col:

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

    st.subheader("Services")

    services = st.multiselect(
        "Select subscribed services",
        [
            "Multiple Lines",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies"
        ]
    )

    st.subheader("Billing")

    col1, col2 = st.columns(2)

    with col1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0
        )

    with col2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=1000.0
        )

    predict_button = st.button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

# Prediction
with right_col:

    st.subheader("Churn Risk")

    if not predict_button:

        st.info("Enter customer information and click Predict Churn Risk.")

    else:

        # ------------------------------------------
        # Build input
        # ------------------------------------------

        input_data = {
            "gender": 1 if gender == "Male" else 0,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": 1 if partner == "Yes" else 0,
            "Dependents": 1 if dependents == "Yes" else 0,
            "tenure": tenure,
            "PhoneService": 1,
            "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        input_df = pd.DataFrame([input_data])


        multiple_lines = (
            "Yes"
            if "Multiple Lines" in services
            else "No"
        )

        if internet_service == "No":

            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

        else:

            online_security = (
                "Yes" if "Online Security" in services else "No"
            )

            online_backup = (
                "Yes" if "Online Backup" in services else "No"
            )

            device_protection = (
                "Yes" if "Device Protection" in services else "No"
            )

            tech_support = (
                "Yes" if "Tech Support" in services else "No"
            )

            streaming_tv = (
                "Yes" if "Streaming TV" in services else "No"
            )

            streaming_movies = (
                "Yes" if "Streaming Movies" in services else "No"
            )

        input_df["MultipleLines"] = multiple_lines
        input_df["InternetService"] = internet_service
        input_df["OnlineSecurity"] = online_security
        input_df["OnlineBackup"] = online_backup
        input_df["DeviceProtection"] = device_protection
        input_df["TechSupport"] = tech_support
        input_df["StreamingTV"] = streaming_tv
        input_df["StreamingMovies"] = streaming_movies

        if tenure > 0:
            avg_monthly_charge = total_charges / tenure
        else:
            avg_monthly_charge = monthly_charges

        charge_ratio = monthly_charges / (total_charges + 1)

        service_score = sum([
            "Online Security" in services,
            "Online Backup" in services,
            "Device Protection" in services,
            "Tech Support" in services,
            "Streaming TV" in services,
            "Streaming Movies" in services
        ])

        if tenure <= 12:
            tenure_group = "0-1yr"
        elif tenure <= 24:
            tenure_group = "1-2yr"
        elif tenure <= 48:
            tenure_group = "2-4yr"
        elif tenure <= 60:
            tenure_group = "4-5yr"
        else:
            tenure_group = "5-6yr"

        input_df["avg_monthly_charge"] = avg_monthly_charge
        input_df["charge_ratio"] = charge_ratio
        input_df["service_score"] = service_score

        categorical_columns = [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        service_dummies = pd.get_dummies(
            input_df[categorical_columns],
            columns=categorical_columns,
            drop_first=False
        )

        input_df = pd.concat(
            [
                input_df.drop(columns=categorical_columns),
                service_dummies
            ],
            axis=1
        )

        # Contract
        input_df["Contract_Month-to-month"] = int(
            contract == "Month-to-month"
        )

        input_df["Contract_One year"] = int(
            contract == "One year"
        )

        input_df["Contract_Two year"] = int(
            contract == "Two year"
        )

        # Payment method
        input_df["PaymentMethod_Bank transfer (automatic)"] = int(
            payment_method == "Bank transfer (automatic)"
        )

        input_df["PaymentMethod_Credit card (automatic)"] = int(
            payment_method == "Credit card (automatic)"
        )

        input_df["PaymentMethod_Electronic check"] = int(
            payment_method == "Electronic check"
        )

        input_df["PaymentMethod_Mailed check"] = int(
            payment_method == "Mailed check"
        )

        # Tenure group
        input_df["tenure_group_0-1yr"] = int(
            tenure_group == "0-1yr"
        )

        input_df["tenure_group_1-2yr"] = int(
            tenure_group == "1-2yr"
        )

        input_df["tenure_group_2-4yr"] = int(
            tenure_group == "2-4yr"
        )

        input_df["tenure_group_4-5yr"] = int(
            tenure_group == "4-5yr"
        )

        input_df["tenure_group_5-6yr"] = int(
            tenure_group == "5-6yr"
        )

        # Match model's 48 features
        input_df = input_df.reindex(
            columns=feature_names,
            fill_value=0
        )


        input_scaled = scaler.transform(input_df)

        churn_probability = model.predict_proba(
            input_scaled
        )[0, 1]

        # Display

        st.metric(
            "Churn Probability",
            f"{churn_probability:.0%}"
        )

        if churn_probability >= 0.60:
            st.error("High Risk")

        elif churn_probability >= 0.30:
            st.warning("Medium Risk")

        else:
            st.success("Low Risk")

        # Shap features that contribute to churn risk

        st.subheader("Key Risk Factors")

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            input_scaled
        )

        customer_shap = shap_values[0]

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": customer_shap
        })

        risk_factors = (
            shap_df[shap_df["shap_value"] > 0]
            .sort_values(
                "shap_value",
                ascending=False
            )
            .head(4)
        )

        feature_labels = {
            "Contract_Month-to-month": "Month-to-month contract",
            "Contract_One year": "One-year contract",
            "Contract_Two year": "Two-year contract",
            "MonthlyCharges": "Monthly charges",
            "TotalCharges": "Total charges",
            "tenure": "Customer tenure",
            "charge_ratio": "Monthly charge relative to total charges",
            "avg_monthly_charge": "Average monthly charge",
            "service_score": "Number of subscribed services",
            "InternetService_Fiber optic": "Fiber optic internet",
            "InternetService_DSL": "DSL internet",
            "InternetService_No": "No internet service",
            "OnlineSecurity_No": "No online security",
            "OnlineSecurity_Yes": "Online security",
            "TechSupport_No": "No tech support",
            "TechSupport_Yes": "Tech support",
            "SeniorCitizen": "Senior citizen",
            "Partner": "Partner",
            "Dependents": "Dependents",
            "PaperlessBilling": "Paperless billing"
        }

        for _, row in risk_factors.iterrows():

            feature = row["feature"]

            label = feature_labels.get(
                feature,
                feature.replace("_", " ")
            )

            st.write(f"• {label}")