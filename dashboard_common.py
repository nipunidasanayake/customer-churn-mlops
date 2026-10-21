"""Shared layout and API helpers for the Streamlit dashboard pages."""

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #142033;
            --muted: #68758a;
            --blue: #4169e1;
            --line: #e7ebf2;
            --canvas: #f5f7fb;
        }
        .stApp { background: var(--canvas); }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--ink); font-weight: 700 !important; }
        h1 { letter-spacing: -.035em; }
        h2 { font-size: 1.15rem !important; }
        [data-testid="stSidebar"] { background: #111c2e; border-right: 1px solid #202c40; }
        [data-testid="stSidebar"] * { color: #eef3ff; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #aab7cc; }
        [data-testid="stSidebar"] input { background: #1c2940; border-color: #33435d; }
        [data-testid="stSidebar"] div.stButton > button {
            background: #1d2b42; border: 1px solid #3b4c68; color: #eef3ff;
        }
        [data-testid="stSidebar"] div.stButton > button:hover {
            background: #293b59; border-color: #6685c8; color: #fff;
        }
        [data-testid="stMetric"] {
            background: #fff; border: 1px solid var(--line); border-radius: 13px;
            padding: 16px 18px; box-shadow: 0 3px 12px rgba(24, 39, 75, .035);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; font-weight: 600; }
        [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.55rem; font-weight: 700; }
        div.stButton > button, div.stFormSubmitButton > button {
            border-radius: 9px; font-weight: 700; min-height: 2.6rem;
        }
        div[data-testid="stDataFrame"] {
            background: #fff; border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
        }
        .eyebrow {
            color: var(--blue); font-size: .72rem; font-weight: 700;
            letter-spacing: .13em; text-transform: uppercase; margin-bottom: .45rem;
        }
        .subtle { color: var(--muted); font-size: .86rem; }
        .brand {
            align-items: center; color: #f4f7ff; display: flex;
            font-size: 1rem; font-weight: 800; padding: .2rem 0 .65rem;
        }
        .brand-icon {
            align-items: center; background: #1d2b42; border: 1px solid #364761;
            border-radius: 10px; color: #8dabff; display: inline-flex;
            height: 36px; justify-content: center; margin-right: 10px;
            width: 36px; font-weight: 800;
        }
        .sidebar-copy { color: #aab7cc; font-size: .78rem; line-height: 1.5; margin-bottom: 1.1rem; }
        .risk-card {
            background: #fff; border: 1px solid var(--line); border-radius: 16px;
            padding: 23px; margin: .5rem 0 1rem;
        }
        .risk-card.high { border-left: 5px solid #e56b61; }
        .risk-card.medium { border-left: 5px solid #efa83b; }
        .risk-card.low { border-left: 5px solid #35a873; }
        .risk-title { color: var(--muted); font-size: .76rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
        .risk-result { color: var(--ink); font-size: 1.55rem; font-weight: 800; margin: .3rem 0; }
        .risk-detail { color: var(--muted); font-size: .87rem; }
        .feature-card {
            background: #fff; border: 1px solid var(--line); border-radius: 14px;
            height: 100%; padding: 20px;
        }
        .feature-icon { color: var(--blue); font-size: 1.25rem; margin-bottom: .7rem; }
        .feature-title { color: var(--ink); font-size: 1rem; font-weight: 700; margin-bottom: .4rem; }
        .feature-copy { color: var(--muted); font-size: .84rem; line-height: 1.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_sidebar(active_page: str) -> str:
    st.sidebar.markdown(
        """
        <div class="brand"><span class="brand-icon">C</span>Churn Intelligence</div>
        <div class="sidebar-copy">Customer churn prediction<br>and model operations</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("#### WORKSPACE")
    api_url = st.sidebar.text_input(
        "Prediction API",
        value=st.session_state.get("dashboard_api_url", DEFAULT_API_URL),
        help="Base URL for the FastAPI prediction service.",
    ).rstrip("/")
    st.session_state["dashboard_api_url"] = api_url
    st.sidebar.caption(f"Current page · {active_page}")
    st.sidebar.divider()
    st.sidebar.caption("Historical prediction records exclude customer identifiers.")
    return api_url


@st.cache_data(ttl=10, show_spinner=False)
def fetch_json(api_url: str, path: str) -> dict:
    response = requests.get(f"{api_url.rstrip('/')}{path}", timeout=10)
    response.raise_for_status()
    return response.json()
