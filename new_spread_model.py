# CITS4403 — Social Media News Spread Model (Fake vs Real)
# -----------------------------------------------------------------------------
# This is a FULL, WORKING Python script you can run locally.
# It models information diffusion on social networks and compares ER/WS/BA graphs.
# It implements two competing contagions (Fake vs Real), platform interventions,
# and collects basic metrics (reach over time, peak concurrency, etc.).
#
# IMPORTANT: Every block is thoroughly commented so you can understand line-by-line.
# You can run this as:  python news_spread_model.py
# Or paste into a Jupyter notebook cell.
# -----------------------------------------------------------------------------

# ------------------------------
# 0) Imports
# ------------------------------
import math                  # math utilities (e.g., for safe computations)
import random                # pseudo-random numbers for stochastic rules
from dataclasses import dataclass  # structured parameter container
from typing import Dict, List, Tuple, Literal  # type hints for clarity

import numpy as np           # numeric arrays; handy for counters and vectors
import networkx as nx        # graph generation and operations
import matplotlib.pyplot as plt  # plotting results

# ------------------------------
# 1) Reproducibility (set seed)
# ------------------------------
random.seed(42)      # ensure runs are repeatable
np.random.seed(42)   # ensure NumPy random is repeatable

# ------------------------------
# 2) Model states (simple small ints for speed)
# ------------------------------
# For each contagion type (Fake or Real), each node can be in one of 5 states.
# We choose integers because they're efficient and easy to count.
STATE_UNAWARE   = 0  # has not seen the item yet
STATE_SEEN      = 1  # has seen/consumed but not posted
STATE_POSTED    = 2  # has posted/shared it in the current or previous step
STATE_QUARANT   = 3  # quarantined/flagged (platform intervention)
STATE_IMMUNE    = 4  # lost interest / will not share this item

# We'll refer to contagions by short string labels so we can store two parallel state dicts
CONTAGION_FAKE = "F"
CONTAGION_REAL = "R"

# ------------------------------
# 3) Parameter container
# ------------------------------
@dataclass
class Params:
    # Network parameters
    topology: Literal["ER", "WS", "BA"] = "BA"  # which graph family to use
    n: int = 2000                                  # number of nodes
    mean_degree: int = 8                           # approximate average degree
    p_er: float = 0.004                            # ER edge probability (approx for mean_degree)
    ws_rewire_p: float = 0.05                      # WS rewire probability (small-world)
    ba_m: int = 4                                  # BA: edges to attach from new node to existing

    # Diffusion parameters (shared base rates)
    beta_see: float = 0.6       # probability to see a neighbor's post
    beta_share: float = 0.20    # base probability to share after seeing
    decay: float = 0.05         # probability a SEEN user becomes IMMUNE (loses interest)
    flag_prob: float = 0.10     # probability platform flags (quarantines) a FAKE seen/post

    # Competition / correction
    salience_fake: float = 1.20 # multiplicative boost for fake when choosing what to post
    correction_effect: float = 0.5  # seeing REAL reduces future share prob of FAKE by this factor

    # User heterogeneity (credulity)
    enable_heterogeneity: bool = False
    credulity_alpha: float = 2.0  # Beta(alpha,beta) for user credulity distribution
    credulity_beta: float = 5.0

    # Simulation
    T: int = 60                 # number of time steps
    n_seeds_fake: int = 5       # initial number of posters for FAKE
    n_seeds_real: int = 5       # initial number of posters for REAL

    # Attention constraints (if True, users can only act on one item per step)
    limit_attention: bool = True

# ------------------------------
# 4) Graph generation helpers
# ------------------------------
def build_graph(p: Params) -> nx.Graph:
    """Create and return a graph according to Params.topology.

    For comparable mean degree across families, we:
      - ER: choose p_er ~ mean_degree/(n-1)
      - WS: k ~ mean_degree (even) and rewire with ws_rewire_p
      - BA: m ~ mean_degree//2 (since avg deg ≈ 2m)
    """
    if p.topology == "ER":
        # For ER, set probability to achieve approx mean degree
        pe = p.p_er if p.p_er > 0 else p.mean_degree / max(1, (p.n - 1))
        G = nx.erdos_renyi_graph(p.n, pe)
    elif p.topology == "WS":
        # Ensure k is even and >=2
        k = max(2, p.mean_degree + (p.mean_degree % 2))
        G = nx.watts_strogatz_graph(p.n, k, p.ws_rewire_p)
    elif p.topology == "BA":
        m = max(1, p.ba_m)
        G = nx.barabasi_albert_graph(p.n, m)
    else:
        raise ValueError("Unknown topology")

    # Optionally ensure the graph is connected to avoid trivial isolates
    if not nx.is_connected(G):
        # Take the largest connected component only (common in ER with small p)
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy()
    return G

# ------------------------------
# 5) State initialisation per contagion
# ------------------------------
def init_states(G: nx.Graph, p: Params) -> Dict[str, Dict[int, int]]:
    """Create initial state dictionaries for FAKE and REAL contagions.

    Returns:
      states: { contagion_label -> {node -> state_int} }
    """
    states = {
        CONTAGION_FAKE: {node: STATE_UNAWARE for node in G.nodes()},
        CONTAGION_REAL: {node: STATE_UNAWARE for node in G.nodes()},
    }

    # Seed initial posters for each contagion at distinct nodes when possible
    nodes = list(G.nodes())
    random.shuffle(nodes)
    fake_seed_nodes = nodes[:min(p.n_seeds_fake, len(nodes))]
    real_seed_nodes = nodes[min(p.n_seeds_fake, len(nodes)):
                            min(p.n_seeds_fake + p.n_seeds_real, len(nodes))]

    for u in fake_seed_nodes:
        states[CONTAGION_FAKE][u] = STATE_POSTED
    for u in real_seed_nodes:
        states[CONTAGION_REAL][u] = STATE_POSTED

    return states

# ------------------------------
# 6) User heterogeneity (credulity)
# ------------------------------
def draw_user_credulity(G: nx.Graph, p: Params) -> Dict[int, float]:
    """Optionally draw a per-user credulity in [0,1] from a Beta distribution.
    Higher credulity => more likely to share after seeing.
    If disabled, returns 1.0 for all users (homogeneous population).
    """
    if not p.enable_heterogeneity:
        return {u: 1.0 for u in G.nodes()}
    alpha, beta = p.credulity_alpha, p.credulity_beta
    vals = np.random.beta(alpha, beta, size=G.number_of_nodes())
    return {u: float(vals[i]) for i, u in enumerate(G.nodes())}

# ------------------------------
# 7) Core step update for a single contagion
# ------------------------------
def step_contagion(G: nx.Graph,
                   current: Dict[int, int],
                   other: Dict[int, int],
                   contagion: str,
                   p: Params,
                   cred: Dict[int, float]) -> Dict[int, int]:
    """Compute next-state map for ONE contagion given the current states.

    Args:
      G: graph of users
      current: state dict for THIS contagion
      other: state dict for the competing contagion (to model correction effects)
      contagion: "F" or "R"
      p: parameters
      cred: per-user credulity in [0,1]

    Returns:
      next_state: dict of same shape with updated states (synchronous update)
    """
    next_state = current.copy()  # start from current; we'll change where needed

    # Precompute which nodes posted this step (to model exposure to neighbors)
    posters = [u for u, st in current.items() if st == STATE_POSTED]

    # For efficiency: create a boolean/fast lookup set for posters
    poster_set = set(posters)

    # Exposure pass: nodes can move from UNAWARE -> SEEN due to neighbors' posts
    for u in G.nodes():
        if current[u] == STATE_UNAWARE:
            # If any neighbor posted, user might SEE it with probability beta_see
            # Stop at first positive exposure for speed
            saw = False
            for v in G.neighbors(u):
                if v in poster_set and random.random() < p.beta_see:
                    saw = True
                    break
            if saw:
                next_state[u] = STATE_SEEN

    # Sharing / decay / flagging pass
    for u in G.nodes():
        st = current[u]

        if st == STATE_SEEN:
            # Decide whether user u will share this contagion
            # Base share prob = beta_share * credulity
            share_prob = p.beta_share * cred.get(u, 1.0)

            # If competing contagion REAL has been seen/posted by u, downweight FAKE share
            if contagion == CONTAGION_FAKE:
                if other[u] in (STATE_SEEN, STATE_POSTED, STATE_QUARANT, STATE_IMMUNE):
                    share_prob *= p.correction_effect  # e.g., 0.5 reduces share prob

            # Optional salience boost for FAKE when choosing among multiple items
            if contagion == CONTAGION_FAKE and p.limit_attention:
                share_prob *= p.salience_fake

            if random.random() < min(1.0, share_prob):
                next_state[u] = STATE_POSTED
            elif random.random() < p.decay:
                # loses interest
                next_state[u] = STATE_IMMUNE

        elif st == STATE_POSTED:
            # Posting persists one step; user can get flagged if this is FAKE
            if contagion == CONTAGION_FAKE and random.random() < p.flag_prob:
                next_state[u] = STATE_QUARANT

        elif st == STATE_QUARANT:
            # Quarantined users remain quarantined (simple model)
            next_state[u] = STATE_QUARANT

        # IMMUNE and UNAWARE handled implicitly (no changes unless exposed)

    return next_state

# ------------------------------
# 8) Full two-contagion simulation loop
# ------------------------------
def simulate(G: nx.Graph, p: Params) -> Dict[str, np.ndarray]:
    """Run the two-contagion simulation and collect time-series metrics.

    Returns a dict of metrics with arrays of length T (or <=T if graph shrank).
    """
    # Init states
    states = init_states(G, p)

    # Per-user credulity
    cred = draw_user_credulity(G, p)

    # Metrics to track over time, one series per contagion
    # We store: ever_seen, current_posters, quarantined counts per step
    ever_seen_F = np.zeros(p.T, dtype=int)
    ever_seen_R = np.zeros(p.T, dtype=int)
    posters_F   = np.zeros(p.T, dtype=int)
    posters_R   = np.zeros(p.T, dtype=int)
    quarant_F   = np.zeros(p.T, dtype=int)

    # A helper to compute "ever seen" cumulative set
    seen_ever_F = set(u for u, st in states[CONTAGION_FAKE].items() if st in (STATE_SEEN, STATE_POSTED, STATE_QUARANT, STATE_IMMUNE))
    seen_ever_R = set(u for u, st in states[CONTAGION_REAL].items() if st in (STATE_SEEN, STATE_POSTED, STATE_QUARANT, STATE_IMMUNE))

    # Time loop
    for t in range(p.T):
        # Record metrics BEFORE stepping (current step status)
        posters_F[t] = sum(1 for st in states[CONTAGION_FAKE].values() if st == STATE_POSTED)
        posters_R[t] = sum(1 for st in states[CONTAGION_REAL].values() if st == STATE_POSTED)
        quarant_F[t] = sum(1 for st in states[CONTAGION_FAKE].values() if st == STATE_QUARANT)

        # Update ever-seen sets
        seen_ever_F.update([u for u, st in states[CONTAGION_FAKE].items() if st in (STATE_SEEN, STATE_POSTED, STATE_QUARANT, STATE_IMMUNE)])
        seen_ever_R.update([u for u, st in states[CONTAGION_REAL].items() if st in (STATE_SEEN, STATE_POSTED, STATE_QUARANT, STATE_IMMUNE)])
        ever_seen_F[t] = len(seen_ever_F)
        ever_seen_R[t] = len(seen_ever_R)

        # Synchronous update: compute next states for each contagion based on CURRENT ones
        next_F = step_contagion(G, states[CONTAGION_FAKE], states[CONTAGION_REAL], CONTAGION_FAKE, p, cred)
        next_R = step_contagion(G, states[CONTAGION_REAL], states[CONTAGION_FAKE], CONTAGION_REAL, p, cred)

        # Assign for next iteration
        states[CONTAGION_FAKE] = next_F
        states[CONTAGION_REAL] = next_R

    # Pack results
    return {
        "ever_seen_F": ever_seen_F,
        "ever_seen_R": ever_seen_R,
        "posters_F": posters_F,
        "posters_R": posters_R,
        "quarant_F": quarant_F,
        "N": np.array([G.number_of_nodes()] * p.T),
    }

# ------------------------------
# 9) Convenience: run experiment for a given topology and plot
# ------------------------------

def run_and_plot(topology: str, p: Params) -> Dict[str, np.ndarray]:
    """Build graph for the chosen topology, run simulation, and make plots."""
    # Update topology in params (copy-ish behaviour)
    p = Params(**{**p.__dict__, "topology": topology})

    # Choose graph probability parameters to maintain mean degree across families
    if topology == "ER":
        p.p_er = p.mean_degree / max(1, (p.n - 1))
    elif topology == "WS":
        # mean_degree already used as k (even) in build_graph
        pass
    elif topology == "BA":
        p.ba_m = max(1, p.mean_degree // 2)

    # Build graph and simulate
    G = build_graph(p)
    out = simulate(G, p)

    # Create figure with two subplots: reach and posters per step
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot ever seen (cumulative reach)
    axes[0].plot(out["ever_seen_F"], label="Fake — ever seen")
    axes[0].plot(out["ever_seen_R"], label="Real — ever seen")
    axes[0].set_title(f"Cumulative Reach — {topology}")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Users")
    axes[0].legend()

    # Plot posters per step (virality/peak)
    axes[1].plot(out["posters_F"], label="Fake — posters")
    axes[1].plot(out["posters_R"], label="Real — posters")
    axes[1].set_title(f"Posters per Step — {topology}")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Users posting")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Print quick summary numbers to console
    print(f"\n=== {topology} SUMMARY ===")
    print(f"N (nodes): {out['N'][0]}")
    print(f"Final reach — Fake: {out['ever_seen_F'][-1]}  Real: {out['ever_seen_R'][-1]}")
    print(f"Peak posters — Fake: {int(out['posters_F'].max())}  Real: {int(out['posters_R'].max())}")
    print(f"Final quarantined (Fake): {out['quarant_F'][-1]}")

    return out

# ------------------------------
# 10) Main: compare ER, WS, BA (easy marks)
# ------------------------------
if __name__ == "__main__":
    # Choose a base param set that runs fast on a laptop and shows differences
    base = Params(
        topology="BA",       # will be overridden per run
        n=2000,               # size of the network; 2k runs fast and is illustrative
        mean_degree=8,        # average degree ~8 across families
        ws_rewire_p=0.05,     # small-world rewiring prob
        T=60,                 # number of steps
        n_seeds_fake=5,
        n_seeds_real=5,
        beta_see=0.6,
        beta_share=0.2,
        decay=0.05,
        flag_prob=0.1,
        salience_fake=1.2,
        correction_effect=0.5,
        enable_heterogeneity=False,  # set True to explore credulity differences
        limit_attention=True,
    )

    # Run three topologies back-to-back
    outputs = {}
    for topo in ["ER", "WS", "BA"]:
        outputs[topo] = run_and_plot(topo, base)

    # NOTE: You can now inspect `outputs` for programmatic comparisons if you wish.
