# CITS4403 — CA Model of Fake vs Real Information Spread

## A simple guide to set up and run the simulation model for analysing how fake and real information spread using a Cellular Automata (CA) approach.

### 1. Preparation
Before running the project, ensure you have Python 3.11+ installed.
Then, create a virtual environment and install the required dependencies.

Create and activate a virtual environment:
    python -m venv .venv
    # For macOS/Linux
    source .venv/bin/activate
    # For Windows
    .venv\\Scripts\\activate

Install dependencies:
    pip install -r requirements.txt

### 2. Running the Simulation
All simulations are run using the following pattern:
    python -m src.ca.run

Where:
- src.ca.run points to the Cellular Automata (CA) simulation engine.
- [OPTIONS] are arguments that modify how the model behaves.

Available Command-line Arguments

--scheme (sync or async, default: sync)

Determines whether updates are synchronous (all agents update together) or asynchronous (random order).

--runs (integer, default: 1)

Number of independent runs to perform. Useful for batch mode.

--label (string, default: None)

Optional label added to output folder names.

--seed (integer, default: None)

RNG seed for reproducibility.

--micro (comma-separated, default: "")

Toggles for micro-level behaviours — available options:
async, refractory, misclass.

--macro (comma-separated, default: "")

Toggles for macro-level factors — available options:
hetero, spatial.

--eta (float [0–1], default: 0.02)

Misclassification probability (chance that an agent misreads fake ↔ real).

--hetero-sd (float [0–1], default: 0.20)

Standard deviation controlling variation in individual behaviour (heterogeneity).

--spatial-strength (float [0–1], default: 0.35)

Controls the strength of spatial influence — 0 means no effect, 1 means strong spatial variation.

Example：
``` 
- python -m src.ca.run --scheme sync --runs 1

- python -m src.ca.run --scheme async --micro async refractory,misclass --eta 0.02 --runs 1
```

### 3. Outputs
All outputs are saved automatically under:
    data/runs/ca/

Each run produces a JSON file containing parameters, time-series, and reach values.
Batch runs also generate summary.csv and summary.json files.

### 4. Visualising Results
You can load and plot any simulation result with the provided utilities.

Example:
from utils.io import load_json
from utils.plotting import plot_macro, plot_shares
import matplotlib.pyplot as plt

run = load_json("data/runs/ca/CA_baseline_20251012-165225.json")
plot_macro(run, title="Active Posters Over Time")
plot_shares(run, title="New Shares per Step")
plt.show()

This will display graphs comparing fake vs real spread dynamics.

### 5. Reproducibility
- Every run embeds its parameters and random seed in the JSON file.
- Running with the same seed reproduces identical results.
- Batch runs summarise metrics such as reach and peak times.

### 6. Tests
To run basic tests (if available):
    pytest -q
