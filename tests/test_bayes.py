import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from sounding.bayes import build_model  # noqa: E402
from sounding.design import VULNERABILITY_TYPES, build_design  # noqa: E402
from sounding.observers import simulate_responses  # noqa: E402
from sounding.psychometric import Params  # noqa: E402


@pytest.mark.slow
def test_hierarchical_model_recovers_two_observers():
    trials = build_design(replicates=12, seed=0)
    x = np.array([t.level for t in trials], dtype=float)
    signal = np.array([t.signal for t in trials])
    type_idx = np.array(
        [VULNERABILITY_TYPES.index(t.vulnerability_type) if t.signal else 0 for t in trials]
    )
    truths = [Params(2.0, 0.5, 0.05, 0.02), Params(3.0, 0.5, 0.05, 0.02)]
    ys = [simulate_responses(x, signal, p, seed=i) for i, p in enumerate(truths)]
    n = len(trials)
    model = build_model(
        x=np.tile(x, 2),
        signal=np.tile(signal, 2),
        y=np.concatenate(ys),
        model_idx=np.repeat([0, 1], n),
        type_idx=np.tile(type_idx, 2),
        n_models=2,
        n_types=len(VULNERABILITY_TYPES),
    )
    with model:
        idata = pm.sample(500, tune=500, chains=2, random_seed=1, progressbar=False)
    alpha = idata.posterior["alpha"].mean(("chain", "draw")).values.mean(axis=1)
    assert alpha[0] == pytest.approx(2.0, abs=0.35)
    assert alpha[1] == pytest.approx(3.0, abs=0.35)
    assert alpha[1] - alpha[0] > 0.5
