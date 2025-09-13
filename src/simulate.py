import random
import numpy as np
from .model import *
from .graph_builders import build_graph
from .contagion_rules import draw_credulity, step_contagion

def init_states(G, p):
    """Seed initial posters for Fake and Real."""
    states = {
        CONTAGION_FAKE: {u: STATE_UNAWARE for u in G.nodes()},
        CONTAGION_REAL: {u: STATE_UNAWARE for u in G.nodes()},
    }
    nodes = list(G.nodes()); random.shuffle(nodes)
    fake = nodes[:min(p.n_seeds_fake, len(nodes))]
    real = nodes[min(p.n_seeds_fake, len(nodes)):
                 min(p.n_seeds_fake + p.n_seeds_real, len(nodes))]
    for u in fake: states[CONTAGION_FAKE][u] = STATE_POSTED
    for u in real: states[CONTAGION_REAL][u] = STATE_POSTED
    return states

def simulate(p: Params):
    """Run the two-contagion simulation and return time-series metrics."""
    G = build_graph(p)
    states = init_states(G, p)
    cred = draw_credulity(G, p)

    T = p.T
    out = {
        "ever_seen_F": np.zeros(T, int),
        "ever_seen_R": np.zeros(T, int),
        "posters_F":   np.zeros(T, int),
        "posters_R":   np.zeros(T, int),
        "quarant_F":   np.zeros(T, int),
        "N":           np.array([G.number_of_nodes()] * T, int),
    }

    seenF = {u for u,st in states[CONTAGION_FAKE].items() if st!=STATE_UNAWARE}
    seenR = {u for u,st in states[CONTAGION_REAL].items() if st!=STATE_UNAWARE}

    for t in range(T):
        # record metrics
        out["posters_F"][t] = sum(st==STATE_POSTED for st in states[CONTAGION_FAKE].values())
        out["posters_R"][t] = sum(st==STATE_POSTED for st in states[CONTAGION_REAL].values())
        out["quarant_F"][t] = sum(st==STATE_QUARANT for st in states[CONTAGION_FAKE].values())

        # cumulative reach
        seenF.update(u for u,st in states[CONTAGION_FAKE].items() if st!=STATE_UNAWARE)
        seenR.update(u for u,st in states[CONTAGION_REAL].items() if st!=STATE_UNAWARE)
        out["ever_seen_F"][t] = len(seenF)
        out["ever_seen_R"][t] = len(seenR)

        # synchronous next states
        nextF = step_contagion(G, states[CONTAGION_FAKE], states[CONTAGION_REAL], CONTAGION_FAKE, p, cred)
        nextR = step_contagion(G, states[CONTAGION_REAL], states[CONTAGION_FAKE], CONTAGION_REAL, p, cred)
        states[CONTAGION_FAKE], states[CONTAGION_REAL] = nextF, nextR

    return out
