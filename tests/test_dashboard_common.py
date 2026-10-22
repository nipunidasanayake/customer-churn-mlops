from types import SimpleNamespace

import pytest

import dashboard_common
from src.dashboard_runtime import DashboardRuntimeError
from dashboard_common import (
    DEMO_IP_LIMITS,
    DemoPredictionRateLimiter,
    EMBEDDED_API_URL,
    MAX_DEMO_PREDICTIONS_PER_SESSION,
    has_demo_prediction_quota,
)


@pytest.mark.parametrize(
    ("predictions_used", "expected"),
    [
        (0, True),
        (MAX_DEMO_PREDICTIONS_PER_SESSION - 1, True),
        (MAX_DEMO_PREDICTIONS_PER_SESSION, False),
        (MAX_DEMO_PREDICTIONS_PER_SESSION + 1, False),
        (-1, False),
    ],
)
def test_demo_prediction_quota(predictions_used, expected):
    assert has_demo_prediction_quota(predictions_used) is expected


def test_ip_rate_limit_is_shared_across_sessions_and_expires():
    limiter = DemoPredictionRateLimiter()

    for _ in range(DEMO_IP_LIMITS[0][1]):
        limiter.check("192.0.2.1", now=100)

    with pytest.raises(DashboardRuntimeError, match="10 prediction attempts per IP per 10 minutes"):
        limiter.check("192.0.2.1", now=100)

    limiter.check("192.0.2.2", now=100)
    limiter.check("192.0.2.1", now=100 + DEMO_IP_LIMITS[0][0])


def test_ip_rate_limit_enforces_daily_limit():
    limiter = DemoPredictionRateLimiter()
    daily_limit = DEMO_IP_LIMITS[1][1]
    interval = DEMO_IP_LIMITS[0][0]

    for attempt in range(daily_limit):
        limiter.check("192.0.2.1", now=attempt * interval)

    with pytest.raises(DashboardRuntimeError, match="50 prediction attempts per IP per 24 hours"):
        limiter.check("192.0.2.1", now=daily_limit * interval)

    limiter.check("192.0.2.1", now=86_400)


@pytest.mark.parametrize("ip_address", [None, "", "not-an-ip"])
def test_ip_rate_limit_rejects_missing_or_invalid_ip(ip_address):
    limiter = DemoPredictionRateLimiter()

    with pytest.raises(DashboardRuntimeError, match="client IP"):
        limiter.check(ip_address, now=100)


def test_ip_rate_limit_normalizes_ipv6_addresses():
    limiter = DemoPredictionRateLimiter()

    for _ in range(DEMO_IP_LIMITS[0][1]):
        limiter.check("2001:0db8:0000:0000:0000:0000:0000:0001", now=100)

    with pytest.raises(DashboardRuntimeError, match="10 prediction attempts per IP"):
        limiter.check("2001:db8::1", now=100)


def test_embedded_predictions_are_limited_by_client_ip(monkeypatch):
    limiter = DemoPredictionRateLimiter()
    scored = []
    monkeypatch.setattr(dashboard_common, "_DEMO_PREDICTION_RATE_LIMITER", limiter)
    monkeypatch.setattr(dashboard_common, "_dashboard_runtime", lambda: object())
    monkeypatch.setattr(
        dashboard_common,
        "predict_customer",
        lambda runtime, payload: scored.append(payload) or {},
    )
    monkeypatch.setattr(
        dashboard_common,
        "st",
        SimpleNamespace(context=SimpleNamespace(ip_address="192.0.2.1")),
    )

    for _ in range(DEMO_IP_LIMITS[0][1]):
        dashboard_common.post_prediction(EMBEDDED_API_URL, {})

    with pytest.raises(DashboardRuntimeError, match="10 prediction attempts per IP"):
        dashboard_common.post_prediction(EMBEDDED_API_URL, {})
    assert len(scored) == DEMO_IP_LIMITS[0][1]
