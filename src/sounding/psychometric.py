"""The psychometric function and a maximum-likelihood fit.

    P(detect | signal at intensity x) = gamma + (1 - gamma - lam) * logistic((x - alpha) / beta)
    P(detect | catch trial)           = gamma

alpha  threshold: the intensity at which detection is halfway up the curve
beta   slope: how sharply detection rises around the threshold
gamma  false-alarm rate: how often a cue is "detected" when there is none
lam    lapse rate: how often even an unmissable cue is missed

The MLE here is the fast, dependency-light fit used for simulation, power
analysis and tests. The hierarchical Bayesian model is in ``sounding.bayes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class Params:
    alpha: float
    beta: float
    gamma: float
    lam: float

    def shifted(self, d_alpha: float = 0.0, d_gamma: float = 0.0) -> Params:
        """A copy with the threshold and/or false-alarm rate moved.

        ``d_alpha < 0`` is a real gain in sensitivity. ``d_gamma > 0`` alone is
        only a more liberal criterion. Telling these apart is the project.
        """
        return Params(self.alpha + d_alpha, self.beta, self.gamma + d_gamma, self.lam)


def p_detect(x, signal, params: Params) -> np.ndarray:
    """Detection probability per trial. ``x`` is ignored where ``signal`` is False."""
    x = np.asarray(x, dtype=float)
    signal = np.asarray(signal, dtype=bool)
    curve = params.gamma + (1 - params.gamma - params.lam) * expit((x - params.alpha) / params.beta)
    return np.where(signal, curve, params.gamma)


def _nll(theta, x, signal, y) -> float:
    p = np.clip(p_detect(x, signal, Params(*theta)), 1e-9, 1 - 1e-9)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_mle(x, signal, y) -> Params:
    """Fit the four parameters by maximum likelihood, from several starting points."""
    x = np.asarray(x, dtype=float)
    signal = np.asarray(signal, dtype=bool)
    y = np.asarray(y, dtype=float)
    if not signal.any() or signal.all():
        raise ValueError("need both signal and catch trials to identify the false-alarm rate")
    lo, hi = x[signal].min(), x[signal].max()
    bounds = [(lo - 2, hi + 2), (0.05, 5.0), (0.0, 0.5), (0.0, 0.3)]
    best = None
    for a0 in np.linspace(lo, hi, 4):
        for b0 in (0.3, 1.0):
            res = minimize(
                _nll, x0=[a0, b0, 0.05, 0.02], args=(x, signal, y), bounds=bounds, method="L-BFGS-B"
            )
            if best is None or res.fun < best.fun:
                best = res
    return Params(*best.x)
