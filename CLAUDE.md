# Sounding: instructions for Claude Code

- **Synthetic and public data only.** Read `docs/DATA_POLICY.md` before adding
  any stimulus or data file. Never add real customer text, employer material,
  or details of real people, and never weaken `sounding.provenance` or its tests.
- Every new stimulus needs a provenance block and must pass `validate_record`.
- Simulation first: new analysis code must recover planted parameters in a test
  before it is used on model output.
- Do not report anything as a result unless it came from a run made after
  `docs/PREREGISTRATION.md` was frozen.
- Before committing: `uv run ruff check . && uv run ruff format --check . && uv run pytest -m "not slow"`.
