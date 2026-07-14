# Test layout

Tests are organized by responsibility, not by the historical source layout:

- `unit/` checks one typed contract or implementation boundary at a time;
- `integration/` parses complete problems and exercises one or more algorithms;
- `WFOMC_RUN_SLOW=1` enables known slow legacy algorithm combinations.

All modules use the `test_*.py` naming convention. New tests should stay close
to the domain they protect and should use local helpers unless a fixture is
genuinely shared by several files.

Unit tests protect behavior at typed boundaries. They should not freeze source
file names, package layout, removed legacy attributes, or exact `__all__`
contents. Use one table-driven test for variants of the same contract, and put
cross-algorithm/model agreement in `integration/` instead of repeating it in
`unit/`.

## Pre-refactor migration

The repository at `/Users/lucien/Sync/repos/wfoMC` is a read-only behavioral
reference. Its ten test modules were migrated as follows:

| Pre-refactor module | Current authoritative coverage |
|---|---|
| `tests/wfomc_test.py` | `integration/test_algorithm_consistency.py`, `integration/test_math_answers.py`, `integration/test_solver_results.py` |
| `tests/propositional_test.py` | `integration/test_propositional.py` |
| `tests/test_formula_models.py` | `unit/test_qf_boolean.py` |
| `tests/test_ganak_count.py` | `unit/test_ganak.py` |
| `tests/test_incremental3_regressions.py` | `integration/test_incremental3_regressions.py`, `unit/test_incremental3_counting_native.py`, `unit/test_normal_form.py`, `unit/test_cardinality.py` |
| `tests/unary_evidence/test_algorithm_matrix.py` | `integration/test_unary_evidence.py`, `unit/test_cell_graph_data.py`, `unit/test_cell_graph_semantics.py` |
| `tests/unary_evidence/test_cell_evidence_allocation.py` | `unit/test_cell_graph_evidence.py` |
| `tests/unary_evidence/test_linear_order.py` | `integration/test_linear_order.py` |
| `tests/unary_evidence/test_required_unary_preds.py` | `unit/test_cell_graph_evidence.py` |
| `tests/unary_evidence/test_unary_evidence_partition.py` | `unit/test_evidence_partition.py` |

Assertions tied to deleted legacy classes were translated to the current typed
boundary. For example, internal two-table model storage is now covered through
the aggregated `PairFactor` contract, and embedded counting bodies are covered
through normal-form definitions instead of an obsolete rejection assertion.

## Running tests

Use a hard timeout so a solver regression cannot stall a development run:

```bash
/opt/homebrew/bin/timeout 30s uv run pytest -q tests/unit
/opt/homebrew/bin/timeout 120s uv run pytest -q
```
