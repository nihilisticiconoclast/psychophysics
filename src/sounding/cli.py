"""Command line entry point: ``sounding stimuli`` and ``sounding simulate``."""

from __future__ import annotations

import argparse
import os

import numpy as np

from sounding.design import build_design
from sounding.observers import simulate_responses
from sounding.psychometric import Params, fit_mle
from sounding.sdt import dprime_criterion
from sounding.stimuli import render, write_jsonl


def _stimuli(args) -> None:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = write_jsonl(render(build_design(args.replicates, seed=args.seed), seed=args.seed), args.out)
    print(f"wrote {n} synthetic stimuli to {args.out}")


def _simulate(args) -> None:
    """Plant two prompt effects in simulated observers and try to recover them."""
    trials = build_design(args.replicates, seed=args.seed)
    x = np.array([t.level for t in trials], dtype=float)
    signal = np.array([t.signal for t in trials])
    base = Params(alpha=2.5, beta=0.5, gamma=0.05, lam=0.02)
    observers = {
        "baseline": base,
        "prompt improves sensitivity (threshold -0.8)": base.shifted(d_alpha=-0.8),
        "prompt only adds caution (false alarms +0.20)": base.shifted(d_gamma=0.20),
    }
    n_sig, n_catch = int(signal.sum()), int((~signal).sum())
    print(f"{len(trials)} trials per observer ({n_sig} signal, {n_catch} catch)\n")
    print(f"{'observer':<48}{'alpha':>7}{'gamma':>7}{'d-prime':>9}{'crit':>7}")
    for i, (name, truth) in enumerate(observers.items()):
        y = simulate_responses(x, signal, truth, seed=args.seed + i)
        fit = fit_mle(x, signal, y)
        d, c = dprime_criterion(
            int(y[signal].sum()), int((1 - y[signal]).sum()),
            int(y[~signal].sum()), int((1 - y[~signal]).sum()),
        )  # fmt: skip
        print(f"{name:<48}{fit.alpha:>7.2f}{fit.gamma:>7.2f}{d:>9.2f}{c:>7.2f}")
        print(f"{'  planted':<48}{truth.alpha:>7.2f}{truth.gamma:>7.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sounding", description=__doc__)
    sub = parser.add_subparsers(required=True)
    s = sub.add_parser("stimuli", help="render the synthetic stimulus set to JSONL")
    s.add_argument("--out", default="data/generated/stimuli.jsonl")
    s.add_argument("--replicates", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_stimuli)
    m = sub.add_parser("simulate", help="parameter-recovery demo on simulated observers")
    m.add_argument("--replicates", type=int, default=20)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=_simulate)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
