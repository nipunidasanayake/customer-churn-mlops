"""Customer input form, live prediction results, and private prediction history."""

import pandas as pd
import requests
import streamlit as st

from dashboard_common import (
    DashboardRuntimeError,
    apply_styles,
    configure_sidebar,
    fetch_json,
    post_prediction,
)

apply_styles()
api_url = configure_sidebar("Customer predictions")

st.markdown('<div class="eyebrow">CUSTOMER INTELLIGENCE / PREDICTION</div>', unsafe_allow_html=True)
st.title("Predict customer churn")
st.markdown(
    '<div class="subtle">Enter service and billing details to score a customer with the active model.</div>',
    unsafe_allow_html=True,
)

try:
    model = fetch_json(api_url, "/model-info")
    history = fetch_json(api_url, "/prediction-history?limit=100")
except (requests.RequestException, DashboardRuntimeError) as exc:
    st.error(f"Could not load the prediction service at {api_url}: {exc}")
    if api_url != "embedded://local":
        st.info("Start FastAPI with `uvicorn app.main:app --reload` and refresh this page.")
    st.stop()

summary = st.columns(4)
summary[0].metric("Total predictions", f'{history["total_predictions"]:,}')
summary[1].metric("Predicted churn", f'{history["predicted_churn_count"]:,}')
summary[2].metric("Predicted not churn", f'{history["predicted_non_churn_count"]:,}')
summary[3].metric("Active model", model.get("model_name", "Unknown"))

st.write("")
left, right = st.columns([1.6, 1], gap="large")
with left:
    st.markdown("## Customer profile")
    st.markdown(
        '<div class="subtle">Only model features are submitted; no names, email addresses, or customer IDs are requested.</div>',
        unsafe_allow_html=True,
    )
    with st.form("customer-churn-prediction"):
        profile_col, service_col, billing_col = st.columns(3)
        with profile_col:
            st.markdown("#### Profile")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has a partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        with service_col:
            st.markdown("#### Services")
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet service", ["Fiber optic", "DSL", "No"]
            )
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )
        with billing_col:
            st.markdown("#### Billing")
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly charges ($)", min_value=0.0, value=75.5, step=5.0
            )
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, value=900.0, step=25.0
            )
        submitted = st.form_submit_button(
            "🔮  Predict customer churn",
            type="primary",
            use_container_width=True,
        )

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen == "Yes"),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    try:
        with st.spinner("Scoring customer with the active model..."):
            st.session_state["last_customer_prediction"] = post_prediction(
                api_url,
                payload,
            )
        fetch_json.clear()
        st.rerun()
    except (requests.RequestException, DashboardRuntimeError) as exc:
        st.error(f"Prediction request failed: {exc}")

result = st.session_state.get("last_customer_prediction")
with right:
    st.markdown("## Prediction result")
    st.markdown(
        '<div class="subtle">The class uses the model\'s 0.50 decision threshold. '
        'Risk bands are descriptive, not business-calibrated.</div>',
        unsafe_allow_html=True,
    )
    if not result:
        with st.container(border=True):
            st.markdown("Complete the customer profile and select **Predict customer churn**.")
            st.caption("The model will return a probability, predicted class, and model version.")
    else:
        probability = float(result["churn_probability"])
        risk_class = "high" if probability >= 0.6 else "medium" if probability >= 0.3 else "low"
        risk_label = {"high": "High risk", "medium": "Medium risk", "low": "Low risk"}[risk_class]
        st.markdown(
            f"""
            <div class="risk-card {risk_class}">
              <div class="risk-title">Prediction · {risk_label}</div>
              <div class="risk-result">{result["prediction_label"]}</div>
              <div class="risk-detail">Model probability · {probability:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(probability, text=f"Churn probability · {probability:.1%}")
        detail_one, detail_two = st.columns(2)
        detail_one.metric("Predicted class", "Yes" if result["prediction"] else "No")
        detail_two.metric("Decision threshold", f'{result["decision_threshold"]:.0%}')
        st.caption(f'Model version {result["model_version"]} · ID {result["prediction_id"]}')
        st.caption(
            "Risk bands: low < 30%, medium 30–59%, high ≥ 60%. "
            "Production decisions should use a threshold validated against business costs."
        )

st.write("")
st.markdown("## Prediction history")
st.markdown(
    '<div class="subtle">Recent predictions are shown without customer profile attributes or identifiers.</div>',
    unsafe_allow_html=True,
)

predictions = history.get("predictions", [])
if predictions:
    history_frame = pd.DataFrame(predictions)
    history_frame["timestamp"] = pd.to_datetime(
        history_frame["timestamp"], utc=True
    ).dt.strftime("%b %d, %Y %H:%M:%S UTC")
    history_frame["prediction"] = history_frame["prediction"].map({1: "Churn", 0: "No churn"})
    history_frame["churn_probability"] = history_frame["churn_probability"].map(
        lambda probability: f"{probability:.1%}"
    )
    history_frame["prediction_id"] = history_frame["prediction_id"].map(
        lambda identifier: identifier[:8]
    )
    history_frame = history_frame.rename(
        columns={
            "timestamp": "Time (UTC)",
            "prediction_id": "Reference",
            "prediction": "Prediction",
            "churn_probability": "Churn probability",
            "model_version": "Model version",
        }
    )
    st.dataframe(history_frame, use_container_width=True, hide_index=True)
else:
    with st.container(border=True):
        st.info("No customer predictions recorded yet. Your first result will appear here.")

st.caption(
    f'History includes up to {len(predictions)} recent records. '
    "Prediction telemetry is retained according to the API retention settings."
)
