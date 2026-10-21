"""Home page for the customer churn prediction and MLOps dashboard."""

import requests
import streamlit as st

from dashboard_common import apply_styles, configure_sidebar, fetch_json

st.set_page_config(
    page_title="Customer Churn AI Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def show_home() -> None:
    apply_styles()
    api_url = configure_sidebar("Overview")

    st.markdown('<div class="eyebrow">CUSTOMER INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.title("Customer Churn AI Platform")
    st.markdown(
        '<div class="subtle">Generate customer-level churn predictions and keep an eye on the '
        'model and prediction service from one workspace.</div>',
        unsafe_allow_html=True,
    )

    try:
        health = fetch_json(api_url, "/health")
        model = fetch_json(api_url, "/model-info")
        monitoring = fetch_json(api_url, "/monitoring")
    except requests.RequestException as exc:
        st.error(f"Could not connect to the prediction API at {api_url}: {exc}")
        st.info("Start the API with `uvicorn app.main:app --reload` or update its URL in the sidebar.")
        st.stop()

    api_metrics = monitoring.get("api_metrics", {})
    metrics = model.get("test_metrics", {})
    columns = st.columns(4)
    columns[0].metric("Predictions recorded", f'{monitoring["observations_total"]:,}')
    columns[1].metric(
        "Churn risk · recent window",
        f'{monitoring["predicted_high_risk_rate"]:.1%}'
        if monitoring["predicted_high_risk_rate"] is not None else "—",
        f'{monitoring["window_size"]:,} recent predictions',
    )
    columns[2].metric("Serving model", model.get("model_name", "Unknown"))
    columns[3].metric(
        "API status",
        health.get("status", "unknown").title(),
        f'{api_metrics.get("request_count", 0)} requests in 24h',
    )

    st.write("")
    prediction_col, monitor_col = st.columns(2)
    with prediction_col:
        st.markdown(
            """
            <div class="feature-card">
              <div class="feature-icon">01 / PREDICT</div>
              <div class="feature-title">Score a customer</div>
              <div class="feature-copy">Enter a customer's service and billing details to get a live churn
              probability, risk band, and predicted class from the active model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(
            prediction_page,
            label="Open customer predictions",
            icon="🔮",
            use_container_width=True,
        )

    with monitor_col:
        st.markdown(
            """
            <div class="feature-card">
              <div class="feature-icon">02 / MONITOR</div>
              <div class="feature-title">Monitor the AI service</div>
              <div class="feature-copy">Review prediction activity, API latency and errors, feature drift,
              and the model's offline evaluation metrics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(
            monitor_page,
            label="Open MLOps monitoring",
            icon="📊",
            use_container_width=True,
        )

    st.write("")
    st.markdown("## Current model")
    st.markdown(
        '<div class="subtle">Offline holdout metrics are shown here. Production metrics require verified customer outcomes.</div>',
        unsafe_allow_html=True,
    )
    quality = st.columns(5)
    for column, label, key in zip(
        quality,
        ("Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"),
        ("accuracy", "precision", "recall", "f1", "roc_auc"),
    ):
        value = metrics.get(key)
        column.metric(label, f"{value:.1%}" if value is not None else "—")

    st.caption(
        f'Active version {model.get("model_version", "unknown")} · '
        f'{monitoring["window_size"]} predictions in the recent drift window · '
        "Experiment history is available in the separate MLflow UI."
    )


prediction_page = st.Page(
    "pages/1_Customer_Predictions.py",
    title="Customer Predictions",
    icon="🔮",
)
monitor_page = st.Page(
    "pages/2_MLOps_Monitor.py",
    title="MLOps Monitoring",
    icon="📊",
)
home_page = st.Page(
    show_home,
    title="Overview",
    icon="🏠",
    default=True,
)

navigation = st.navigation(
    {
        "Churn platform": [home_page, prediction_page, monitor_page],
    },
    position="sidebar",
)
navigation.run()
