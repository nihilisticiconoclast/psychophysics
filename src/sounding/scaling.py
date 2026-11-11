"""Bradley-Terry scaling of cue intensity from pairwise human judgements.

Raters see two cues and pick the one that more strongly signals vulnerable
circumstances. Fitting Bradley-Terry to those choices puts every cue on one
interval scale, which then replaces the provisional 1-4 levels as the x-axis of
the psychometric function.
"""

from __future__ import annotations

import numpy as np


def fit_bradley_terry(wins: np.ndarray, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Fit by the MM algorithm. ``wins[i, j]`` counts times item i beat item j.

    Returns log-strengths centred on zero. Larger means a stronger cue.
    """
    wins = np.asarray(wins, dtype=float)
    if wins.ndim != 2 or wins.shape[0] != wins.shape[1]:
        raise ValueError("wins must be a square matrix")
    n = wins.shape[0]
    games = wins + wins.T
    total_wins = wins.sum(axis=1)
    if (total_wins == 0).any() or (wins.sum(axis=0) == 0).any():
        raise ValueError("every item must win and lose at least once for a finite estimate")
    p = np.ones(n)
    for _ in range(iters):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        new = total_wins / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    return np.log(p)


def simulate_comparisons(log_strengths, n_per_pair: int, seed: int = 0) -> np.ndarray:
    """Simulate a round-robin of pairwise judgements for testing."""
    rng = np.random.default_rng(seed)
    s = np.asarray(log_strengths, dtype=float)
    n = len(s)
    wins = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p_i = 1 / (1 + np.exp(s[j] - s[i]))
            w = rng.binomial(n_per_pair, p_i)
            wins[i, j], wins[j, i] = w, n_per_pair - w
    return wins
