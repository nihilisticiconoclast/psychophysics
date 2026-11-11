"""The factorial trial design.

Signal trials cross vulnerability type x cue level x financial context. Catch
trials carry a neutral filler in place of a cue, so a "detection" on one is a
false alarm. Without them the false-alarm rate, and so the whole
sensitivity-versus-criterion question, cannot be estimated.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

VULNERABILITY_TYPES: tuple[str, ...] = ("health", "life_event", "resilience", "capability")
CONTEXTS: tuple[str, ...] = ("pension_withdrawal", "debt", "investment")
CUE_LEVELS: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class Trial:
    trial_id: str
    signal: bool
    vulnerability_type: str | None  # None on catch trials
    level: int  # 0 on catch trials
    context: str
    replicate: int

    def as_dict(self) -> dict:
        return asdict(self)


def build_design(replicates: int = 1, catch_per_context: int = 8, seed: int = 0) -> list[Trial]:
    """Build a shuffled, fully crossed design.

    Per replicate: every (type, level, context) cell once, plus
    ``catch_per_context`` catch trials in each context.
    """
    if replicates < 1 or catch_per_context < 1:
        raise ValueError("replicates and catch_per_context must be at least 1")
    trials: list[Trial] = []
    for rep in range(replicates):
        for context in CONTEXTS:
            for vtype in VULNERABILITY_TYPES:
                for level in CUE_LEVELS:
                    trials.append(
                        Trial(
                            f"s-{vtype}-{level}-{context}-{rep}", True, vtype, level, context, rep
                        )
                    )
            for k in range(catch_per_context):
                trials.append(Trial(f"c-{k}-{context}-{rep}", False, None, 0, context, rep))
    random.Random(seed).shuffle(trials)
    return trials
