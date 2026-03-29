.PHONY: install test stream

install:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt

test:
	.venv/Scripts/pytest tests/ -v --timeout=30

stream:
	.venv/Scripts/python -m src.stream_processor
