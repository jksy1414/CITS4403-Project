# CITS4403 — CA Model of Fake vs Real Information Spread

## Quickstart
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

## Run a simulation
python -m src.ca.run

Outputs:
- JSON: data/runs/CA_run_YYYYMMDD-HHMMSS.json
- Figures (from notebooks): data/figures/*.png

## Repo layout
(src/ca engine, utils for IO/metrics/plotting, notebooks per experiment)

## Reproducibility
- All params embedded in each JSON (including RNG seed).
- Notebooks load JSON and regenerate figures.

## Tests
pytest -q
