# delong.py
# Minimal DeLong implementation for correlated ROC AUCs
# Ref: DeLong et al. (1988), Sun & Xu (2014)

import numpy as np
from scipy.stats import norm

def _midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ties = counts[inv]
    return ranks - (ties - 1) / 2.0

def _fast_delong(y_true, y_score):
    """Return AUC and DeLong covariance components (V10, V01)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("Both positive and negative samples are required.")
    # midranks
    r_pos = _midrank(pos)
    r_neg = _midrank(neg)
    r_all = _midrank(np.concatenate([pos, neg]))
    # AUC
    auc = (np.sum(r_all[:m]) - m * (m + 1) / 2.0) / (m * n)
    # V10, V01
    v10 = (r_pos - (m + 1) / 2.0) / n
    v01 = (r_neg - (n + 1) / 2.0) / m
    return auc, v10, v01

def delong_ci(y_true, y_score, alpha=0.05):
    """AUC ve (1-alpha) GA (DeLong varyansı ile)."""
    auc, v10, v01 = _fast_delong(y_true, y_score)
    var = (np.var(v10, ddof=1) / len(v10)) + (np.var(v01, ddof=1) / len(v01))
    se = np.sqrt(var)
    z = norm.ppf(1 - alpha / 2.0)
    lo = auc - z * se
    hi = auc + z * se
    return float(auc), float(lo), float(hi), float(se)

def delong_roc_test(y_true, score1, score2):
    """İki AUC farkı için DeLong z ve p (iki kuyruk)."""
    auc1, v10_1, v01_1 = _fast_delong(y_true, score1)
    auc2, v10_2, v01_2 = _fast_delong(y_true, score2)
    # Kovaryans
    cov_v10 = np.cov(v10_1, v10_2, ddof=1)[0, 1] / len(v10_1)
    cov_v01 = np.cov(v01_1, v01_2, ddof=1)[0, 1] / len(v01_1)
    var1 = (np.var(v10_1, ddof=1) / len(v10_1)) + (np.var(v01_1, ddof=1) / len(v01_1))
    var2 = (np.var(v10_2, ddof=1) / len(v10_2)) + (np.var(v01_2, ddof=1) / len(v01_2))
    cov12 = cov_v10 + cov_v01
    var_diff = var1 + var2 - 2 * cov12
    se_diff = np.sqrt(var_diff)
    if se_diff == 0:
        z = np.inf
        p = 0.0 if abs(auc1 - auc2) > 0 else 1.0
    else:
        z = (auc1 - auc2) / se_diff
        p = 2 * (1 - norm.cdf(abs(z)))
    return {"auc1": float(auc1), "auc2": float(auc2), "z": float(z), "p": float(p)}
