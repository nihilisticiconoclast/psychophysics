"""Render trials into synthetic customer messages.

Messages are assembled from the handwritten seed bank shipped with the package.
Every record is passed through the data-policy guard before it is returned.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from importlib import resources
from typing import Any

from sounding.design import Trial
from sounding.provenance import validate_record


def load_bank() -> dict[str, Any]:
    with resources.files("sounding.data").joinpath("cue_bank.json").open(encoding="utf-8") as f:
        return json.load(f)


def render(trials: Iterable[Trial], seed: int = 0) -> list[dict[str, Any]]:
    """Turn trials into stimulus records with text, factors and provenance."""
    bank = load_bank()
    rng = random.Random(seed)
    cues: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cue in bank["cues"]:
        cues.setdefault((cue["type"], cue["level"]), []).append(cue)

    records = []
    for trial in trials:
        if trial.signal:
            part = rng.choice(cues[(trial.vulnerability_type, trial.level)])
        else:
            part = rng.choice(bank["neutral_fillers"])
        text = bank["contexts"][trial.context].format(cue=part["text"])
        record = {
            **trial.as_dict(),
            "stimulus_id": hashlib.sha256(text.encode()).hexdigest()[:12],
            "text": text,
            "part_id": part["id"],
            "duration": part.get("duration"),
            "provenance": {
                "kind": "synthetic-template",
                "parts": bank["provenance"],
                "sources": ["FCA-FG21/1"],
            },
        }
        validate_record(record)
        records.append(record)
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n
