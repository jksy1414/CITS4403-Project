import numpy as np, statistics as stats

def t50(cum_reach, N):  # time to 50% reach
    arr = np.asarray(cum_reach)
    idx = np.argmax(arr >= 0.5 * N)
    return int(idx)

def peak(series):       # peak posters
    return int(np.max(series))

def auc(series):        # area under posters curve (total posting activity)
    return int(np.sum(series))

def mean_ci(xs):        # mean ± 95% CI
    if not xs: return 0.0, 0.0
    mu = stats.mean(xs)
    sd = stats.pstdev(xs) if len(xs) > 1 else 0.0
    ci = 1.96 * sd / (len(xs) ** 0.5 if len(xs) > 0 else 1)
    return mu, ci
