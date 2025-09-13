import random
import numpy as np
from .model import *

def draw_credulity(G, p):
    """Return per-user credulity in [0,1]. If heterogeneity is off, everyone=1.0."""
    if not p.enable_heterogeneity:
        return {u: 1.0 for u in G.nodes()}
    vals = np.random.beta(p.cred_alpha, p.cred_beta, size=G.number_of_nodes())
    return {u: float(vals[i]) for i, u in enumerate(G.nodes())}

def step_contagion(G, current, other, contagion, p, cred):
    """
    current: state dict for this contagion (F or R)
    other: state dict for the competing contagion
    Returns a NEW dict (synchronous update).
    """
    next_state = current.copy()
    poster_set = {u for u, st in current.items() if st == STATE_POSTED}

    # Exposure: UNAWARE -> SEEN if any neighbor posted (prob beta_see)
    for u in G.nodes():
        if current[u] == STATE_UNAWARE:
            for v in G.neighbors(u):
                if v in poster_set and random.random() < p.beta_see:
                    next_state[u] = STATE_SEEN
                    break

    # Decision for SEEN: share, decay, or wait
    for u in G.nodes():
        if current[u] == STATE_SEEN:
            share_prob = p.beta_share * cred.get(u, 1.0)

            # Real corrects Fake (downweight fake after real contact)
            if contagion == CONTAGION_FAKE:
                if other[u] in (STATE_SEEN, STATE_POSTED, STATE_QUARANT, STATE_IMMUNE):
                    share_prob *= p.correction_effect
                if p.limit_attention:
                    share_prob *= p.salience_fake

            if random.random() < min(1.0, share_prob):
                next_state[u] = STATE_POSTED
            elif random.random() < p.decay:
                next_state[u] = STATE_IMMUNE

        # Flagging: Fake posts can be quarantined
        elif current[u] == STATE_POSTED and contagion == CONTAGION_FAKE:
            if random.random() < p.flag_prob:
                next_state[u] = STATE_QUARANT

    return next_state
