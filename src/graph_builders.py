import networkx as nx
from .model import Params

def build_graph(p: Params) -> nx.Graph:
    if p.topology == "ER":
        pe = p.p_er if p.p_er > 0 else p.mean_degree / max(1, p.n - 1)
        G = nx.erdos_renyi_graph(p.n, pe)
    elif p.topology == "WS":
        k = max(2, p.mean_degree + (p.mean_degree % 2))  # even k
        G = nx.watts_strogatz_graph(p.n, k, p.ws_rewire_p)
    elif p.topology == "BA":
        m = max(1, p.ba_m or (p.mean_degree // 2))
        G = nx.barabasi_albert_graph(p.n, m)
    else:
        raise ValueError("Unknown topology")

    # Keep largest connected component (avoid isolates that never see anything)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G
