"""Entrypoint for the no-card, single-app Streamlit Community Cloud demo."""

import os
import runpy
import sys
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["CHURN_DASHBOARD_MODE"] = "embedded"

required_secrets = (
    "MODEL_BUNDLE_URL",
    "MODEL_BUNDLE_SHA256",
    "MODEL_BUNDLE_TOKEN",
)
try:
    missing_secrets = [
        name for name in required_secrets if not st.secrets.get(name)
    ]
except StreamlitSecretNotFoundError:
    missing_secrets = list(required_secrets)

if missing_secrets:
    st.error(
        "Configure these values in the Streamlit app's Secrets settings: "
        f"{', '.join(missing_secrets)}. Do not commit them to the repository."
    )
    st.stop()

for secret_name in required_secrets:
    os.environ[secret_name] = str(st.secrets[secret_name])

runpy.run_path(str(PROJECT_ROOT / "dashboard.py"), run_name="__main__")
