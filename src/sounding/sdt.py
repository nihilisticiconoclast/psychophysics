"""Signal detection summaries: sensitivity (d') and criterion (c)."""

from __future__ import annotations

from scipy.stats import norm


def dprime_criterion(hits: int, misses: int, false_alarms: int, correct_rejections: int):
    """Return ``(d_prime, criterion)`` using the log-linear correction.

    The correction (add 0.5 to each count) keeps rates of exactly 0 or 1 finite.
    Higher d' means better discrimination of cue from no cue. Negative c means a
    liberal observer, one inclined to say "vulnerable" either way.
    """
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    z_hit, z_fa = norm.ppf(hit_rate), norm.ppf(fa_rate)
    return float(z_hit - z_fa), float(-0.5 * (z_hit + z_fa))
