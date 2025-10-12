# Cellular Automata (CA) Model — CITS4403 Project

This folder contains the **Cellular Automata (CA) implementation** for simulating the spread of fake vs real news.  
Each file has a specific role in the model:

---

## Files

### `states.py`
- Defines the **state codes** for each person (e.g., Susceptible, Exposed to Fake, Exposed to Real, Posting Fake, Posting Real).  
- Contains the **Params** class where simulation parameters are set:
  - Grid size `N`  
  - Number of time steps `T`  
  - Sharing probabilities (`beta_share_f`, `beta_share_r`)  
  - Probability of seeing a neighbour’s post (`beta_see`)  
  - Number of initial seeds (`seeds_f0`, `seeds_r0`)  
  - Random seed for reproducibility  

---

### `grid.py`
- Defines **neighbourhood logic**.  
- Currently supports **8-neighbour (Moore) toroidal wrap-around**.  
- Provides functions to check if a person has fake or real posting neighbours.  
- Core idea: “who can each person see?”

---

### `simulate.py`
- Implements the **simulation engine**.  
- Contains:
  - `seed_initial` → creates the grid and places initial fake/real posters.  
  - `step` → applies one tick of CA rules (see → exposed → share → post).  
  - `simulate` → runs for `T` ticks, returns counts of fake and real posters over time.  

---

### `run.py`
- **Entry point** for running the CA simulation directly from terminal.  
- Uses default parameters (can be tweaked).  
- Runs the simulation, prints results (peak fake/real), and saves them into `data/runs/` as JSON.  

### `__init__.py`
- Marks this folder as a Python package.
- Allows you to import files like:
```
from johnkoh.model.simulate import simulate
```

### `Summary`
Together, these files form the CA model:
 - states.py → definitions & parameters
 - grid.py → neighbourhood logic
 - simulate.py → simulation loop
 - run.py → runnable entry point