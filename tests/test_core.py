import json
from pathlib import Path

import numpy as np
import pytest

from sounding.design import CONTEXTS, CUE_LEVELS, VULNERABILITY_TYPES, build_design
from sounding.observers import simulate_responses
from sounding.provenance import DataPolicyError, find_pii, validate_record
from sounding.psychometric import Params, fit_mle, p_detect
from sounding.scaling import fit_bradley_terry, simulate_comparisons
from sounding.sdt import dprime_criterion
from sounding.stimuli import load_bank, render

REPO = Path(__file__).resolve().parents[1]


def test_design_is_fully_crossed_and_reproducible():
    trials = build_design(replicates=2, catch_per_context=5, seed=1)
    signal = [t for t in trials if t.signal]
    assert len(signal) == 2 * len(VULNERABILITY_TYPES) * len(CUE_LEVELS) * len(CONTEXTS)
    assert len(trials) - len(signal) == 2 * 5 * len(CONTEXTS)
    assert len({t.trial_id for t in trials}) == len(trials)
    assert trials == build_design(replicates=2, catch_per_context=5, seed=1)


def test_every_cell_of_the_seed_bank_is_filled():
    cells = {(c["type"], c["level"]) for c in load_bank()["cues"]}
    assert cells == {(v, lv) for v in VULNERABILITY_TYPES for lv in CUE_LEVELS}


def test_rendered_stimuli_carry_provenance_and_pass_policy():
    records = render(build_design(seed=3), seed=3)
    for r in records:
        assert r["provenance"]["kind"] == "synthetic-template"
        assert "{cue}" not in r["text"]
        validate_record(r)  # also run inside render; asserted here on purpose


def test_policy_rejects_unsourced_and_personal_data():
    ok = {"text": "I have some savings.", "provenance": {"kind": "synthetic-handwritten"}}
    validate_record(ok)
    with pytest.raises(DataPolicyError):
        validate_record({"text": "hello"})
    with pytest.raises(DataPolicyError):
        validate_record({"text": "hello", "provenance": {"kind": "customer-transcript"}})
    with pytest.raises(DataPolicyError):
        validate_record(
            {**ok, "provenance": {"kind": "synthetic-llm", "sources": ["internal-wiki"]}}
        )
    for leaked in ("mail me at jo@example.com", "my account is 12345678", "I live at GL5 1AA"):
        with pytest.raises(DataPolicyError):
            validate_record({**ok, "text": leaked})


def test_no_data_file_in_the_repo_contains_personal_data_patterns():
    """Repo-wide guard: runs in CI on every push."""
    files = list((REPO / "src" / "sounding" / "data").rglob("*.json"))
    files += list((REPO / "data").rglob("*.json*")) if (REPO / "data").exists() else []
    assert files
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not find_pii(line), f"{path}: {line[:60]}"
    assert json.loads(files[0].read_text(encoding="utf-8"))["provenance"].startswith("synthetic")


def test_catch_trials_sit_at_the_false_alarm_rate():
    p = Params(alpha=2.5, beta=0.5, gamma=0.1, lam=0.02)
    out = p_detect([0, 100, 2.5], [False, True, True], p)
    assert out[0] == pytest.approx(0.1)
    assert out[1] == pytest.approx(0.98)
    assert out[2] == pytest.approx(0.1 + 0.88 * 0.5)


def _simulated(truth, seed):
    trials = build_design(replicates=40, seed=seed)
    x = np.array([t.level for t in trials], dtype=float)
    signal = np.array([t.signal for t in trials])
    return x, signal, simulate_responses(x, signal, truth, seed=seed)


def test_mle_recovers_planted_parameters():
    truth = Params(alpha=2.5, beta=0.5, gamma=0.05, lam=0.02)
    fit = fit_mle(*_simulated(truth, seed=11))
    assert fit.alpha == pytest.approx(truth.alpha, abs=0.2)
    assert fit.gamma == pytest.approx(truth.gamma, abs=0.03)


def test_sensitivity_gain_and_caution_are_told_apart():
    """The core claim: a threshold shift and a false-alarm shift are separable."""
    base = Params(alpha=2.5, beta=0.5, gamma=0.05, lam=0.02)
    sharper = fit_mle(*_simulated(base.shifted(d_alpha=-0.8), seed=5))
    jumpier = fit_mle(*_simulated(base.shifted(d_gamma=0.20), seed=6))
    assert sharper.alpha < 2.0 and sharper.gamma < 0.12
    assert jumpier.alpha > 2.2 and jumpier.gamma > 0.18


def test_fit_refuses_data_without_catch_trials():
    with pytest.raises(ValueError):
        fit_mle([1, 2, 3], [True, True, True], [0, 1, 1])


def test_dprime_and_criterion():
    d, c = dprime_criterion(50, 50, 50, 50)
    assert d == pytest.approx(0, abs=1e-9) and c == pytest.approx(0, abs=1e-9)
    d_good, _ = dprime_criterion(90, 10, 10, 90)
    assert d_good > 2
    _, c_liberal = dprime_criterion(95, 5, 60, 40)
    assert c_liberal < 0
    assert np.isfinite(dprime_criterion(10, 0, 0, 10)[0])


def test_bradley_terry_recovers_the_ordering_and_scale():
    truth = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    est = fit_bradley_terry(simulate_comparisons(truth, n_per_pair=400, seed=2))
    assert list(np.argsort(est)) == list(np.argsort(truth))
    assert est == pytest.approx(truth, abs=0.25)
    with pytest.raises(ValueError):
        fit_bradley_terry(np.array([[0, 5], [0, 0]]))
