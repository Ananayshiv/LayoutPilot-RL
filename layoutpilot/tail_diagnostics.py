"""Rank-based extreme-tail diagnostics for a fixed learned proxy.

No network training, RNG, or gold-label queries occur in this module.
The 'gold' argument must be obtained only AFTER selection is frozen.
"""
from __future__ import annotations
import numpy as np


def _check(*values):
    arrays = [np.asarray(a, dtype=np.float64).reshape(-1) for a in values]
    if not arrays or any(len(x) != len(arrays[0]) for x in arrays):
        raise ValueError('unequal vector sizes')
    if len(arrays[0]) == 0 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('empty or non-finite vector')
    return arrays


def average_ranks(a):
    """Average ranks (one-based) for ties, no scipy dependency."""
    x, = _check(a)
    order = np.argsort(x, kind='mergesort')
    result = np.empty(len(x), dtype=np.float64)
    values = x[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    ends = np.r_[starts[1:], len(x)]
    for start, end in zip(starts, ends):
        result[order[start:end]] = (start+end+1) / 2
    return result


def rank_correlations(proxy, gold):
    """Spearman's rho and Kendall tau-b, with average ranks and correct ties.

    Kendall uses O(K^2) comparisons; intended for K<=128, not whole banks.
    Degenerate correlations are null (None), not fabricated zeros.
    """
    p, g = _check(proxy, gold)
    if len(p) < 2:
        return {'spearman': None, 'kendall_tau_b': None}
    rp, rg = average_ranks(p), average_ranks(g)
    sp = float(np.corrcoef(rp, rg)[0, 1]) if rp.std()>1e-12 and rg.std()>1e-12 else None
    ii, jj = np.triu_indices(len(p), k=1)
    dx = np.sign(p[ii]-p[jj]); dy = np.sign(g[ii]-g[jj])
    denom = np.sqrt(float(np.count_nonzero(dx))*float(np.count_nonzero(dy)))
    tau = float(np.sum(dx*dy)/denom) if denom > 0 else None
    return {'spearman': sp, 'kendall_tau_b': tau}


def top_tail_metrics(predicted, gold, sigma=None, *, top_k=32, tail_fraction=.05,
                     interval_radius=None):
    """Compute ranking, error, and uncertainty coverage on *proxy-selected* tail.

    'top K' concerns the K highest proxy predictions within the full bank,
    rather than the top K by gold. Tail slice size ceil(fraction*pool size).
    Coverage refers to symmetric uncertainty interval mu +- radius*raw_sigma.
    """
    pred, y = _check(predicted, gold)
    if len(pred)<2 or top_k<2 or not (0 < tail_fraction <= 1):
        raise ValueError('invalid tail configuration')
    k = min(int(top_k), len(pred))
    tail_count = min(len(pred), max(1, int(np.ceil(len(pred)*tail_fraction))))
    order = np.argsort(-pred, kind='mergesort')
    ranks = rank_correlations(pred[order[:k]], y[order[:k]])
    inds = order[:tail_count]
    err = pred[inds]-y[inds]
    result = {'top_k_effective': k, 'tail_count': tail_count,
              'tail_spearman': ranks['spearman'],
              'tail_kendall_tau_b': ranks['kendall_tau_b'],
              'tail_bias_proxy_minus_gold': float(err.mean()),
              'tail_mae': float(np.abs(err).mean()),
              'tail_rmse': float(np.sqrt(np.square(err).mean())),
              'overall_mae': float(np.abs(pred-y).mean())}
    if sigma is not None:
        unc, = _check(sigma)
        if len(unc)!=len(pred) or np.any(unc<0):
            raise ValueError('sigma must be nonnegative and match predictions')
        result['tail_mean_sigma'] = float(unc[inds].mean())
        result['overall_mean_sigma'] = float(unc.mean())
        if interval_radius is not None:
            if interval_radius < 0:
                raise ValueError('interval radius must be >=0')
            covered = np.abs(pred-y) <= interval_radius*unc
            result['tail_interval_coverage'] = float(covered[inds].mean())
            result['overall_interval_coverage'] = float(covered.mean())
    return result


def turnover_diagnostic(search_budgets, gold_curve, *, epsilon=.01):
    """Exploratory turnover (NOT an inferential result; confirmation required).

    The peak is chosen on the given curve and the decline is peak minus final.
    A plateau counts as no observed turnover. For evidence use independent
    confirmation templates with the selected budget frozen.
    """
    n, g = _check(search_budgets, gold_curve)
    if len(n)<2 or np.any(np.diff(n)<=0) or epsilon < 0:
        raise ValueError('N must increase strictly; epsilon >=0')
    peak = int(np.argmax(g))
    drop = float(g[peak] - g[-1])
    return {'peak_n': int(n[peak]), 'final_n': int(n[-1]),
            'max_gold_minus_final_gold': drop,
            'interior_peak': bool(peak < len(n)-1),
            'observed_turnover': bool(peak < len(n)-1 and drop > epsilon),
            'turnover_threshold': float(epsilon)}
