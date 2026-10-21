"""Operational monitoring for prediction activity, data drift, and model quality."""

import altair as alt
import pandas as pd
import requests
import streamlit as st

from dashboard_common import apply_styles, configure_sidebar, fetch_json

apply_styles()
api_url = configure_sidebar("MLOps monitoring")

st.markdown('<div class="eyebrow">MODEL OPERATIONS / MONITORING</div>', unsafe_allow_html=True)
st.title("MLOps monitoring")
st.markdown(
    '<div class="subtle">Prediction traffic, API service quality, input drift, and model evaluation.</div>',
    unsafe_allow_html=True,
)
refresh_column, status_column = st.columns([1, 5])
with refresh_column:
    if st.button("↻  Refresh metrics", type="primary"):
        fetch_json.clear()
        st.rerun()

try:
    health = fetch_json(api_url, "/health")
    model = fetch_json(api_url, "/model-info")
    snapshot = fetch_json(api_url, "/monitoring")
    history = fetch_json(api_url, "/prediction-history?limit=100")
    drift_history = fetch_json(api_url, "/drift-history?limit=100")
except requests.RequestException as exc:
    st.error(f"Could not load monitoring data from {api_url}: {exc}")
    st.info("Start FastAPI with `uvicorn app.main:app --reload` and refresh this page.")
    st.stop()

api_metrics = snapshot["api_metrics"]
with status_column:
    status_column.success(
        f'API {health.get("status", "unknown").upper()} · '
        f'Model {model.get("model_version", "unknown")}'
    )

st.write("")
traffic, probabilities, api_quality, drift_count = st.columns(4)
traffic.metric("Predictions · 30 days", f'{snapshot["observations_total"]:,}')
probabilities.metric(
    "Mean churn probability",
    f'{snapshot["mean_churn_probability"]:.1%}'
    if snapshot["mean_churn_probability"] is not None else "—",
)
api_quality.metric(
    "API errors · 24h",
    f'{api_metrics["error_count"]:,}',
)
api_quality.caption(
    f'{api_metrics["error_rate"]:.1%} error rate'
    if api_metrics["error_rate"] is not None else "No requests yet"
)
has_drift_sample = snapshot["window_size"] >= snapshot["minimum_window_size"]
drift_count.metric(
    "Drifted features" if has_drift_sample else "Drift sample",
    f'{len(snapshot["drifted_features"])} / {len(snapshot["feature_drift"])}'
    if has_drift_sample
    else f'{snapshot["window_size"]} / {snapshot["minimum_window_size"]}',
)
drift_count.caption(
    "Features flagged"
    if has_drift_sample
    else "Predictions / minimum sample"
)

st.write("")
st.markdown("## Prediction and API activity")
st.markdown(
    '<div class="subtle">API service metrics use actual observed request status codes and elapsed times. '
    'Monitoring and health polls are excluded.</div>',
    unsafe_allow_html=True,
)
latency_col, error_col, activity_col = st.columns(3)
latency_col.metric(
    "Average response time",
    f'{api_metrics["average_response_time_ms"]:.0f} ms'
    if api_metrics["average_response_time_ms"] is not None else "—",
)
latency_col.metric(
    "95th percentile response time",
    f'{api_metrics["p95_response_time_ms"]:.0f} ms'
    if api_metrics["p95_response_time_ms"] is not None else "—",
)
error_col.metric("Requests · 24h", f'{api_metrics["request_count"]:,}')
activity_col.metric(
    "Predicted churn · recent history",
    f'{history["predicted_churn_count"]:,}',
    f'{history["predicted_non_churn_count"]:,} predicted not to churn',
)

hourly = pd.DataFrame(api_metrics["requests_by_hour"])
if hourly.empty:
    st.info("API activity will appear after users submit predictions or make API requests.")
else:
    hourly["hour"] = pd.to_datetime(hourly["hour"], utc=True).dt.strftime("%d %b %H:00")
    hour_order = hourly["hour"].tolist()
    chart_data = hourly.melt(
        id_vars="hour",
        value_vars=["requests", "errors"],
        var_name="request_type",
        value_name="count",
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "hour:O",
                title="Hour (UTC)",
                sort=hour_order,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("count:Q", title="API requests", stack=None),
            color=alt.Color(
                "request_type:N",
                title="Request",
                scale=alt.Scale(domain=["requests", "errors"], range=["#4169e1", "#e56b61"]),
            ),
            xOffset="request_type:N",
            tooltip=[
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("request_type:N", title="Type"),
                alt.Tooltip("count:Q", title="Requests"),
            ],
        )
        .properties(height=230)
        .configure_view(stroke=None)
        .configure_axis(domain=False, tickSize=0, labelColor="#77849a")
    )
    st.altair_chart(chart, use_container_width=True)

st.write("")
st.markdown("## Data drift")
st.markdown(
    f'<div class="subtle">Population Stability Index compares the last {snapshot["window_size"]} '
    f'prediction inputs with training data. Alert threshold: {snapshot["drift_threshold"]:.2f}. '
    f'At least {snapshot["minimum_window_size"]} predictions are required.</div>',
    unsafe_allow_html=True,
)

drift = pd.DataFrame(snapshot["feature_drift"])
if snapshot["window_size"] < snapshot["minimum_window_size"]:
    st.warning(
        f'Insufficient data for drift conclusions: {snapshot["window_size"]} of '
        f'{snapshot["minimum_window_size"]} required predictions.'
    )
    st.progress(
        min(snapshot["window_size"] / snapshot["minimum_window_size"], 1.0),
        text="Monitoring sample progress",
    )
else:
    drift_chart_col, drift_summary_col = st.columns([3, 1])
    numeric_drift = drift.dropna(subset=["psi"]).sort_values("psi", ascending=True)
    if not numeric_drift.empty:
        drift_chart = (
            alt.Chart(numeric_drift)
            .mark_bar(cornerRadiusEnd=4, height=16)
            .encode(
                x=alt.X(
                    "psi:Q",
                    title="Population Stability Index",
                    axis=alt.Axis(gridColor="#edf0f5"),
                ),
                y=alt.Y("feature:N", title=None, sort="-x"),
                color=alt.condition(
                    alt.datum.psi >= snapshot["drift_threshold"],
                    alt.value("#e56b61"),
                    alt.value("#4169e1"),
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("psi:Q", title="PSI", format=".3f"),
                    alt.Tooltip("status:N", title="Status"),
                ],
            )
            .properties(height=max(240, len(numeric_drift) * 22))
            .configure_view(stroke=None)
            .configure_axis(domain=False, tickSize=0, labelColor="#536176")
        )
        threshold_line = (
            alt.Chart(pd.DataFrame({"threshold": [snapshot["drift_threshold"]]}))
            .mark_rule(color="#e56b61", strokeDash=[5, 4])
            .encode(x="threshold:Q")
        )
        with drift_chart_col:
            st.altair_chart(drift_chart + threshold_line, use_container_width=True)
    with drift_summary_col:
        if snapshot["drifted_features"]:
            st.error(f'{len(snapshot["drifted_features"])} feature(s) above threshold')
            for feature in snapshot["drifted_features"]:
                st.markdown(f"- `{feature}`")
        else:
            st.success("No monitored features exceed the PSI threshold.")
    display_drift = drift.copy()
    display_drift["psi"] = display_drift["psi"].map(
        lambda value: f"{value:.4f}" if pd.notna(value) else "—"
    )
    st.dataframe(
        display_drift.rename(
            columns={"feature": "Feature", "psi": "PSI", "status": "Status"}
        ),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### Historical drift reports")
drift_reports = drift_history.get("reports", [])
if drift_reports:
    reports_frame = pd.DataFrame(drift_reports)
    reports_frame["timestamp"] = pd.to_datetime(reports_frame["timestamp"], utc=True)
    drift_history_chart = (
        alt.Chart(reports_frame)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("timestamp:T", title="Report time", axis=alt.Axis(format="%d %b %H:%M")),
            y=alt.Y(
                "drift_percentage:Q",
                title="Features flagged",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Report time"),
                alt.Tooltip("drift_percentage:Q", title="Features flagged", format=".1%"),
                alt.Tooltip("drifted_features:Q", title="Drifted features"),
                alt.Tooltip("window_size:Q", title="Prediction sample"),
                alt.Tooltip("model_version:N", title="Model version"),
            ],
        )
        .properties(height=200)
        .configure_view(stroke=None)
        .configure_axis(domain=False, tickSize=0, labelColor="#77849a")
    )
    st.altair_chart(drift_history_chart, use_container_width=True)
    with st.expander(f'View {len(drift_reports)} recent drift reports'):
        report_table = reports_frame[
            ["timestamp", "model_version", "window_size", "drifted_features", "drift_percentage"]
        ].copy()
        report_table["timestamp"] = report_table["timestamp"].dt.strftime("%b %d, %Y %H:%M UTC")
        report_table["drift_percentage"] = report_table["drift_percentage"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(
            report_table.rename(
                columns={
                    "timestamp": "Report time",
                    "model_version": "Model version",
                    "window_size": "Prediction sample",
                    "drifted_features": "Drifted features",
                    "drift_percentage": "Features flagged",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.caption(
        "Historical reports are saved when the monitoring page evaluates a new prediction window "
        "with at least 30 observations."
    )

st.write("")
st.markdown("## Model performance")
st.markdown(
    '<div class="subtle">Offline holdout evaluation for the currently served model. '
    'Live production performance requires actual, verified customer outcomes.</div>',
    unsafe_allow_html=True,
)
test_metrics = model.get("test_metrics", {})
metric_columns = st.columns(5)
for column, label, key in zip(
    metric_columns,
    ("Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"),
    ("accuracy", "precision", "recall", "f1", "roc_auc"),
):
    value = test_metrics.get(key)
    column.metric(label, f"{value:.1%}" if value is not None else "—")
st.caption(
    f'Model {model.get("model_name", "unknown")} · Version {model.get("model_version", "unknown")} · '
    "Prediction outcomes alone are not ground-truth labels."
)
trained_at = model.get("trained_at")
if trained_at:
    st.caption(
        f"Last successful training: {pd.to_datetime(trained_at, utc=True).strftime('%b %d, %Y %H:%M UTC')}"
    )
else:
    st.caption("Last successful training time is not available in this model's metadata.")
