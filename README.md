# CITS4403 — CA Model of Fake vs Real Information Spread

## 1. Preparation
Before running the project, ensure you have Python 3.07 installed.
Then, create a virtual environment and install the required dependencies.

Create and activate a virtual environment:
```bash
    python -m venv .venv
    # For macOS/Linux
    source .venv/bin/activate
    # For Windows
    .venv\\Scripts\\activate
```

Install dependencies:
```bash
    pip install -r requirements.txt
```

## 2. Running the Simulation
All simulations are run using the following pattern:
```bash
    python -m src.ca.run [Option]
```
Key Option are 
```bash
--scheme {sync,async}: update engine (default sync)
--runs INT: number of independent runs (default 1)
--label STR: optional tag added to output folder name
--seed INT: RNG seed (optional)
--micro LIST: micro behaviours (comma-separated): async,refractory,misclass
--macro LIST: macro factors (comma-separated): hetero,spatial
--eta FLOAT: misclassification rate (default 0.02)
--hetero-sd FLOAT: heterogeneity SD (default 0.20)
--spatial-strength FLOAT: spatial strength (default 0.35)
```

Examples: 
Phase 0 baseline (sync engine, no micro/macro):
```bash 
python -m src.ca.run --scheme sync --runs 20
```

Macro only:
```bash
python -m src.ca.run --macro hetero --hetero-sd 0.20 --runs 20
python -m src.ca.run --macro spatial --spatial-strength 0.35 --runs 20
python -m src.ca.run --macro hetero,spatial --hetero-sd 0.20 --spatial-strength 0.35 --runs 20
```

## 3. Outputs
All outputs are saved automatically under:
```bash
    data/runs/ca/
```
The result consist of: 
- Single runs: one JSON per run (time series + params)
- Batch runs: summary.csv and summary.json in the batch folder

## 4. Visualising Results
If you’re using the notebook utilities that write Excel/figures, they’ll save under data/figures/phase*/

```bash
from src.ca.io import load_json         # adjust import to your actual path
from src.ca.plotting import plot_macro, plot_shares  # if you keep these helpers
import matplotlib.pyplot as plt

run = load_json("data/runs/ca/baseline_20251013-180746/run_0001.json")
plot_macro(run, title="Active Posters Over Time")
plot_shares(run, title="New Shares per Step")
plt.show()
```
## 5. Reproducibility
- Every run embeds its parameters and random seed in the JSON file.
- Running with the same seed reproduces identical results.
- Batch runs summarise metrics such as reach and peak times.

### 6. Tests
To run basic tests:
```bash
    pytest -q
```