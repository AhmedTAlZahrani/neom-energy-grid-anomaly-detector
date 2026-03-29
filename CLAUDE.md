# CLAUDE.md

**NEOM energy grid anomaly detection** — autoencoder-based anomaly detection on streaming energy data.

## Setup & Install

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -v --timeout=30
```

## Running

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Notes

- Autoencoder model in `src/autoencoder.py`; streaming logic in `src/stream_processor.py`
- NumPy-style docstrings throughout
- Conventional commit messages (`feat:`, `fix:`, `docs:`)
