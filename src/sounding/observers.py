"""Simulated observers with known parameters.

The whole pipeline is built against these first. If the analysis cannot recover
parameters we planted ourselves, it has no business being pointed at a real
model. The same simulator later drives the power analysis.
"""

from __future__ import annotations

import numpy as np

from sounding.psychometric import Params, p_detect


def simulate_responses(x, signal, params: Params, seed: int = 0) -> np.ndarray:
    """Draw 0/1 detection responses for each trial."""
    rng = np.random.default_rng(seed)
    return (rng.random(len(np.asarray(x))) < p_detect(x, signal, params)).astype(int)
