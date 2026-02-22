import functools
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.autoencoder_model import GridAutoencoder
from src.isolation_forest import IsolationForestDetector
from src.alert_engine import AlertEngine

MODEL_DIR = Path("models")

alert_engine = AlertEngine(warning_threshold=0.5, critical_threshold=0.8)


@functools.lru_cache(maxsize=1)
def get_autoencoder():
    """Load and cache the autoencoder model.

    Returns
    -------
    GridAutoencoder or None
        Loaded autoencoder, or None if checkpoint not found.
    """
    ae_path = MODEL_DIR / "autoencoder.pth"
    if ae_path.exists():
        ae = GridAutoencoder()
        ae.load(str(MODEL_DIR))
        print("Autoencoder loaded")
        return ae
    return None


@functools.lru_cache(maxsize=1)
def get_isolation_forest():
    """Load and cache the Isolation Forest model.

    Returns
    -------
    IsolationForestDetector or None
        Loaded detector, or None if checkpoint not found.
    """
    if_path = MODEL_DIR / "isolation_forest.pkl"
    if if_path.exists():
        ifd = IsolationForestDetector()
        ifd.load(str(MODEL_DIR))
        print("Isolation Forest loaded")
        return ifd
    return None


@asynccontextmanager
async def lifespan(app):
    """Application lifespan handler for model loading.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    """
    # Warm up the caches on startup
    get_autoencoder()
    get_isolation_forest()
    yield


app = FastAPI(
    title="NEOM Energy Grid Anomaly Detector API",
    description="Real-time anomaly detection for NEOM's renewable energy microgrid.",
    version="1.0.0",
    lifespan=lifespan,
)


class SensorReading(BaseModel):
    """Schema for a single sensor reading submission."""
    solar_0_kw: float = 45.0
    solar_1_kw: float = 50.0
    wind_0_kw: float = 200.0
    wind_1_kw: float = 180.0
    battery_0_soc: float = 65.0
    battery_0_charge_rate: float = 0.5
    hydrogen_0_prod_rate: float = 8.0
    hydrogen_0_efficiency: float = 0.72
    hydrogen_0_temp: float = 68.0
    grid_frequency_hz: float = 60.0
    grid_voltage_v: float = 230.0
    total_load_mw: float = 2.5
    total_generation_mw: float = 3.0


class BatchSensorReading(BaseModel):
    """Schema for batch sensor reading submissions."""
    readings: list


class DetectionResult(BaseModel):
    """Schema for anomaly detection response."""
    anomaly_score: float
    is_anomaly: bool
    alert_level: str
    model_used: str


@app.get("/health")
def health():
    """Health check endpoint.

    Returns
    -------
    dict
        Service status and model availability.
    """
    autoencoder = get_autoencoder()
    isolation_forest = get_isolation_forest()
    return {
        "status": "ok",
        "autoencoder_loaded": autoencoder is not None,
        "isolation_forest_loaded": isolation_forest is not None,
    }


@app.get("/model-info")
def model_info():
    """Return metadata about loaded models.

    Returns
    -------
    dict
        Model details for each loaded detector.
    """
    autoencoder = get_autoencoder()
    isolation_forest = get_isolation_forest()
    info = {}

    if autoencoder is not None:
        info["autoencoder"] = autoencoder.get_model_info()
    else:
        info["autoencoder"] = {"status": "not loaded"}

    if isolation_forest is not None:
        info["isolation_forest"] = isolation_forest.get_model_info()
    else:
        info["isolation_forest"] = {"status": "not loaded"}

    return info


@app.post("/detect")
def detect_anomaly(reading: SensorReading):
    """Submit a sensor reading and get anomaly detection results.

    Parameters
    ----------
    reading : SensorReading
        Sensor reading with grid measurements.

    Returns
    -------
    dict
        Anomaly score, binary prediction, and alert level.
    """
    autoencoder = get_autoencoder()
    isolation_forest = get_isolation_forest()
    features = pd.DataFrame([reading.model_dump()])

    if autoencoder is not None:
        scores = autoencoder.compute_anomaly_scores(features)
        score = float(scores[0])
        is_anomaly = bool(score > autoencoder.threshold) if autoencoder.threshold else False
        model_used = "autoencoder"
    elif isolation_forest is not None:
        scores = isolation_forest.compute_anomaly_scores(features)
        score = float(scores[0])
        preds = isolation_forest.predict(features)
        is_anomaly = bool(preds[0])
        model_used = "isolation_forest"
    else:
        raise HTTPException(status_code=503, detail="No models loaded")

    alert_level = alert_engine.classify(score)
    alert_engine.process(score, sensor_id="api_request")

    return {
        "anomaly_score": round(score, 4),
        "is_anomaly": is_anomaly,
        "alert_level": alert_level,
        "model_used": model_used,
    }


@app.post("/detect/batch")
def detect_batch(batch: BatchSensorReading):
    """Submit a batch of sensor readings for anomaly detection.

    Parameters
    ----------
    batch : BatchSensorReading
        Batch of reading dicts.

    Returns
    -------
    dict
        Results for each reading and summary statistics.
    """
    autoencoder = get_autoencoder()
    isolation_forest = get_isolation_forest()

    if not batch.readings:
        raise HTTPException(status_code=400, detail="No readings provided")

    features = pd.DataFrame(batch.readings)
    results = []

    if autoencoder is not None:
        scores = autoencoder.compute_anomaly_scores(features)
        model_used = "autoencoder"
        for i, score in enumerate(scores):
            s = float(score)
            is_anomaly = bool(s > autoencoder.threshold) if autoencoder.threshold else False
            alert_level = alert_engine.classify(s)
            results.append({
                "index": i,
                "anomaly_score": round(s, 4),
                "is_anomaly": is_anomaly,
                "alert_level": alert_level,
            })
    elif isolation_forest is not None:
        scores = isolation_forest.compute_anomaly_scores(features)
        preds = isolation_forest.predict(features)
        model_used = "isolation_forest"
        for i, (score, pred) in enumerate(zip(scores, preds)):
            s = float(score)
            alert_level = alert_engine.classify(s)
            results.append({
                "index": i,
                "anomaly_score": round(s, 4),
                "is_anomaly": bool(pred),
                "alert_level": alert_level,
            })
    else:
        raise HTTPException(status_code=503, detail="No models loaded")

    n_anomalies = sum(1 for r in results if r["is_anomaly"])
    return {
        "model_used": model_used,
        "total_readings": len(results),
        "anomalies_detected": n_anomalies,
        "anomaly_rate": round(n_anomalies / len(results), 4),
        "results": results,
    }


@app.get("/alerts/recent")
def recent_alerts(n: int = 20):
    """Retrieve the most recent anomaly alerts.

    Parameters
    ----------
    n : int
        Number of recent alerts to return.

    Returns
    -------
    dict
        Recent alerts and summary statistics.
    """
    alerts = alert_engine.get_recent_alerts(n=n)
    summary = alert_engine.get_summary()
    return {
        "alerts": alerts,
        "summary": summary,
    }
