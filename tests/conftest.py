import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Sensor data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random state for deterministic tests."""
    return np.random.RandomState(12345)


@pytest.fixture
def sample_sensor_df(rng):
    """Small DataFrame mimicking NEOM grid sensor readings.

    Contains solar, wind, grid-level columns, and anomaly labels.
    100 rows at 1-minute intervals.
    """
    n = 100
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "solar_0_kw": rng.uniform(0, 80, n),
        "solar_1_kw": rng.uniform(0, 80, n),
        "wind_0_kw": rng.uniform(0, 500, n),
        "wind_1_kw": rng.uniform(0, 500, n),
        "battery_0_soc": rng.uniform(20, 95, n),
        "battery_0_charge_rate": rng.normal(0, 1, n),
        "grid_frequency_hz": rng.normal(60, 0.01, n),
        "grid_voltage_v": rng.normal(230, 0.5, n),
        "total_generation_mw": rng.uniform(1, 5, n),
        "total_load_mw": rng.uniform(1.5, 3, n),
        "is_anomaly": np.concatenate([np.zeros(90), np.ones(10)]).astype(int),
        "anomaly_type": ["normal"] * 90 + ["sensor_drift"] * 10,
    })


@pytest.fixture
def numeric_matrix(rng):
    """Plain numpy array (200 x 10) for model-level tests."""
    return rng.randn(200, 10)


@pytest.fixture
def small_numeric_matrix(rng):
    """Tiny numpy array (30 x 5) for fast unit tests."""
    return rng.randn(30, 5)


@pytest.fixture
def anomaly_scores_mixed():
    """Array of anomaly scores spanning normal / warning / critical ranges."""
    return np.array([
        0.0, 0.1, 0.2, 0.3, 0.4,   # normal
        0.5, 0.55, 0.6, 0.7, 0.79,  # warning
        0.8, 0.85, 0.9, 0.95, 1.0,  # critical
    ])


@pytest.fixture
def stream_records():
    """List of dict records suitable for StreamProcessor.publish()."""
    rng_local = np.random.RandomState(99)
    records = []
    for _ in range(25):
        records.append({
            "solar_0_kw": float(rng_local.uniform(0, 80)),
            "wind_0_kw": float(rng_local.uniform(0, 500)),
            "grid_frequency_hz": float(rng_local.normal(60, 0.01)),
        })
    return records
