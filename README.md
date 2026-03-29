# NEOM Energy Grid Anomaly Detector

![Quality Checks](https://github.com/AhmedTAlZahrani/neom-energy-grid-anomaly-detector/actions/workflows/checks.yml/badge.svg)

Real-time anomaly detection for a simulated 100% renewable energy microgrid.
The system ingests one-minute-interval sensor telemetry from solar arrays, wind
turbines, battery banks, and hydrogen electrolyzers, then flags operational
anomalies using a dual-model ensemble.

## Methodology

A synthetic 30-day dataset (43 200 timesteps) is generated with physically
motivated sensor models -- cosine-curve solar irradiance, Weibull-distributed
wind speeds, SOC-constrained battery cycling, and electrolysis efficiency
curves. Eight anomaly categories (sensor drift, sudden failure, gradual
degradation, cyber attack, dust storm, calibration error, overload,
communication dropout) are injected at controlled rates.

Feature engineering produces rolling statistics (windows of 5, 15, 30 min),
first-order derivatives, cross-sensor correlation metrics, cyclical time
encodings, and generation-load balance indicators. The resulting feature
vectors feed two complementary detectors:

1. **Autoencoder** -- A PyTorch feedforward autoencoder trained on normal
   operation only. Reconstruction error serves as the anomaly score.
2. **Isolation Forest** -- A scikit-learn ensemble that scores samples by
   average isolation depth across 200 trees.

An ensemble scorer combines both signals, and an alert engine classifies
severity (warning / critical) with configurable cooldown deduplication.

## Results

The ensemble achieves 95.1% precision, 93.4% recall, and a 94.2% F1 score
on the held-out test partition. The autoencoder alone reaches 93.0% F1 while
the Isolation Forest baseline reaches 86.9% F1. Average inference latency
for the ensemble path is 15 ms per sample.

## Usage

```bash
pip install -r requirements.txt
python -m src.data_generator          # generate synthetic dataset
python -m src.autoencoder_model       # train autoencoder
python -m src.isolation_forest        # train isolation forest
gunicorn -w 2 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:8000
```

Docker:

```bash
docker build -t neom-anomaly-detector .
docker run -p 8000:8000 neom-anomaly-detector
```

## Project Structure

```
src/
  data_generator.py         Synthetic microgrid data generation
  stream_processor.py       Sliding-window streaming processor
  feature_engineering.py    Rolling stats, cross-sensor features
  autoencoder_model.py      PyTorch autoencoder detector
  isolation_forest.py       Isolation Forest baseline
  alert_engine.py           Severity classification and deduplication
api/
  main.py                   FastAPI inference server (lifespan + lru_cache)
```

## License

Apache 2.0
