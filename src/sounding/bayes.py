"""Hierarchical Bayesian psychometric model (needs the ``bayes`` extra).

Thresholds are partially pooled across models and vulnerability types:

    alpha[m, v] = mu_alpha + model_offset[m] + type_offset[v]

so a model tested on few trials borrows strength from the rest, and each
threshold comes with a full posterior rather than a point estimate.
"""

from __future__ import annotations

import numpy as np


def build_model(x, signal, y, model_idx, type_idx, n_models: int, n_types: int):
    """Build the PyMC model.

    ``type_idx`` may hold any valid index on catch trials; it is unused there.
    """
    import pymc as pm

    x = np.asarray(x, dtype=float)
    signal = np.asarray(signal, dtype=bool)
    x_safe = np.where(signal, x, 0.0)

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=float(x[signal].mean()), sigma=2.0)
        sigma_model = pm.HalfNormal("sigma_model", 1.0)
        sigma_type = pm.HalfNormal("sigma_type", 1.0)
        z_model = pm.Normal("z_model", 0.0, 1.0, shape=n_models)
        z_type = pm.Normal("z_type", 0.0, 1.0, shape=n_types)
        alpha = pm.Deterministic(
            "alpha", mu_alpha + sigma_model * z_model[:, None] + sigma_type * z_type[None, :]
        )
        beta = pm.LogNormal("beta", mu=np.log(0.5), sigma=0.5, shape=n_models)
        gamma = pm.Beta("gamma", 1.0, 9.0, shape=n_models)
        lam = pm.Beta("lam", 1.0, 19.0, shape=n_models)

        a = alpha[model_idx, type_idx]
        curve = gamma[model_idx] + (1 - gamma[model_idx] - lam[model_idx]) * pm.math.invlogit(
            (x_safe - a) / beta[model_idx]
        )
        p = pm.math.switch(signal, curve, gamma[model_idx])
        pm.Bernoulli("y", p=p, observed=np.asarray(y, dtype=int))
    return model
