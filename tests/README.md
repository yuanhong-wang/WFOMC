# Test layout

Tests are organized by responsibility, not by the historical source layout:

- `unit/` checks one typed contract or implementation boundary at a time;
- `integration/` parses complete problems and exercises one or more algorithms;
- `WFOMC_RUN_SLOW=1` enables known slow algorithm combinations.

All modules use the `test_*.py` naming convention. New tests should stay close
to the domain they protect and should use local helpers unless a fixture is
genuinely shared by several files.

Unit tests protect behavior at typed boundaries. They should not freeze source
file names, package layout, removed attributes, or exact `__all__` contents.
Use one table-driven test for variants of the same contract, and put
cross-algorithm/model agreement in `integration/` instead of repeating it in
`unit/`.

## Running tests

Use a hard timeout so a solver regression cannot stall a development run:

```bash
/opt/homebrew/bin/timeout 30s uv run pytest -q tests/unit
/opt/homebrew/bin/timeout 120s uv run pytest -q
```
