# Sounding

**The psychophysics of vulnerability detection in language models.**

How faint does a sign of customer vulnerability have to be before a language
model notices it and changes how it responds? Sounding treats the model as an
observer in a psychophysics experiment and measures that threshold.

**Project page:** <https://nihilisticiconoclast.github.io/psychophysics/> (an interactive
psychometric curve, on simulated observers).

> **Data policy.** This project uses **synthetic stimuli and public sources
> only**. No customer data, no employer data, no real people. The policy is in
> [docs/DATA_POLICY.md](docs/DATA_POLICY.md) and is enforced in code and CI.

## The four questions

1. **Threshold.** How subtle a cue can each model detect, by vulnerability type
   (health, life events, resilience, capability)?
2. **Sensitivity or caution.** Does a "take care with vulnerable customers"
   system prompt make a model more perceptive, or only more willing to raise
   false alarms?
3. **Knowing versus doing.** Does a model that spots a cue when asked outright
   also adapt its reply when nobody asks?
4. **Temporary versus enduring.** Are temporary circumstances harder to detect?

## The model

```
P(detect | cue at intensity x) = gamma + (1 - gamma - lambda) * logistic((x - alpha) / beta)
P(detect | no cue)             = gamma
```

`alpha` is the detection threshold, `beta` the slope, `gamma` the false-alarm
rate and `lambda` the lapse rate. A system prompt that lowers `alpha` has
improved perception. One that only raises `gamma` has bought caution with false
alarms.

## Status

Working now, all on simulated observers and a mock model:

| Piece | Module |
| --- | --- |
| Factorial design with catch trials | `sounding.design` |
| Synthetic stimulus rendering from a handwritten seed bank | `sounding.stimuli` |
| Data-policy guard (provenance + personal-data scan) | `sounding.provenance` |
| Psychometric function and maximum-likelihood fit | `sounding.psychometric` |
| Simulated observers for parameter recovery | `sounding.observers` |
| d-prime and criterion | `sounding.sdt` |
| Bradley-Terry intensity scaling | `sounding.scaling` |
| Hierarchical Bayesian model (PyMC) | `sounding.bayes` |
| Inspect tasks: explicit detection and spontaneous adaptation | `sounding.evals.tasks` |

**No real model has been evaluated yet, and there are no results.** The seed
bank has one cue per cell, its intensity levels are provisional, and the judge
rubric is unvalidated. See the roadmap.

## Quick start

```bash
uv venv && uv pip install -e ".[dev,eval,bayes]"
uv run pytest -m "not slow"     # fast suite, includes the data-policy scan
uv run pytest -m slow           # MCMC parameter recovery, about 30 seconds
uv run sounding simulate        # plant two prompt effects, recover them
uv run sounding stimuli         # render the stimulus set to data/generated/
```

Against a real model, once you have chosen one and set its API key:

```bash
inspect eval src/sounding/evals/tasks.py@explicit_detection --model <provider/model>
```

## Roadmap

- [x] Scaffold, simulation-first pipeline, policy guard, CI
- [ ] Literature review and novelty check
- [ ] Freeze [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) before any real-model run
- [ ] Expand the seed bank to several cues per cell
- [ ] Human pairwise scaling of cue intensity; replace provisional levels
- [ ] Validate the adaptation judge against human labels
- [ ] Power analysis by simulation to set trial counts
- [ ] Pilot: one model, one vulnerability type; check for ceiling effects
- [ ] Full experiment, hierarchical fit, write-up and interactive site
