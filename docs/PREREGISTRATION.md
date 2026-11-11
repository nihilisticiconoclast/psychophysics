# Pre-registration (DRAFT, not frozen)

Freeze this file, with a tagged commit, before the first run against a real
model. Items marked **DECIDE** are open choices for the project owner.

## Hypotheses

- **H1 (threshold).** Detection thresholds differ between models and between
  vulnerability types.
- **H2 (sensitivity or caution).** Under the `care` system prompt, relative to
  `baseline`, the false-alarm rate rises. Whether the threshold also falls is
  the open question; both outcomes are reported.
- **H3 (knowing versus doing).** Thresholds for spontaneous adaptation are
  higher than thresholds for explicit detection.
- **H4 (duration).** Temporary circumstances have higher thresholds than
  enduring ones at matched scaled intensity.

## Design

- Signal trials: vulnerability type (4) x cue (several per cell) x context (3).
- Catch trials: neutral filler in place of the cue, same contexts.
- Conditions: task (explicit, spontaneous) x system prompt (baseline, care).
- **DECIDE:** models to test, and the versions to pin.
- **DECIDE:** trials per cell, set by simulated power analysis.

## Measurement

- Intensity (x-axis): Bradley-Terry scale values from human pairwise
  judgements. The provisional 1-4 levels in the seed bank are not used for any
  reported result.
- Explicit detection: one-word YES/NO answer.
- Spontaneous adaptation: LLM judge, blind to condition and cue labels. The
  judge must reach an agreed reliability against human labels before use.
  **DECIDE:** the reliability statistic and its threshold.
- Unparseable answers are excluded and their count reported per cell.

## Analysis

- Primary: hierarchical Bayesian psychometric model (`sounding.bayes`), with
  thresholds partially pooled across models and types.
- H2 is assessed on the posterior for the change in threshold and the change in
  false-alarm rate between prompt conditions.
- Judge error is propagated into the model rather than ignored.
- **DECIDE:** the smallest threshold difference that counts as meaningful.

## Stopping and exclusion rules

- If pilot detection is at ceiling at the faintest cues, add fainter cues
  before the full run; do not drop the model.
- No cue is removed after seeing model responses to it.
