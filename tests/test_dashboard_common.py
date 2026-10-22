import pytest

from dashboard_common import (
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
