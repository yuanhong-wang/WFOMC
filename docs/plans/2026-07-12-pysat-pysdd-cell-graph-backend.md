# PySAT + PySDD Cell-Graph Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace production cell and pair-model enumeration with PySAT projected cell enumeration and PySDD batch exact pair-factor evaluation while preserving every algorithm result.

**Architecture:** Typed Tseitin CNF belongs to `fol`; PySAT and PySDD are confined to `cell_graph/enumerate_cells.py` and `cell_graph/compute_pair_factors.py`. `cell_graph/build.py` orchestrates these capabilities and returns pure `Cell`, `PairFactor`, and `CellGraphData` values; algorithms never retain solver objects.

**Tech Stack:** Python 3.11, python-sat/CaDiCaL, PySDD, python-flint, pytest.

---

### Task 1: Deterministic typed Tseitin CNF

**Files:**
- Create: `src/wfomc/fol/cnf.py`
- Modify: `src/wfomc/fol/semantics.py`
- Modify: `src/wfomc/fol/grounding.py`
- Modify: `src/wfomc/fol/dsl.py`
- Test: `tests/unit/test_fol_cnf.py`
- Test: `tests/unit/test_fol_grounding.py`

**Steps:**

1. Add failing tests for `TseitinCNF` atom mappings, auxiliary variables, and stable grounding variable order.
2. Run `uv run pytest tests/unit/test_fol_cnf.py tests/unit/test_fol_grounding.py -q` and confirm failure.
3. Move Tseitin encoding from `semantics.py` into `fol/cnf.py` and return typed metadata.
4. Migrate every caller to `encode_tseitin()` and remove the obsolete `to_cnf_clauses()` / DSL compatibility exports.
5. Sort free variables deterministically in `ground_on_tuple()`.
6. Run the two tests and confirm pass under multiple `PYTHONHASHSEED` values.

### Task 2: PySAT projected cell enumeration

**Files:**
- Create: `src/wfomc/cell_graph/enumerate_cells.py`
- Modify: `src/wfomc/cell_graph/build.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/unit/test_cell_graph_enumerate_cells.py`

**Steps:**

1. Add differential tests comparing projected PySAT cells with `model_literals()` on formulas containing Tseitin auxiliaries, contradictions, free predicates, and overlapping profile formulas.
2. Add `python-sat` and `pysdd` as project dependencies and update the lockfile.
3. Implement `is_satisfiable(cnf)` and `enumerate_cells(cnf, predicates)` in `cell_graph/enumerate_cells.py` using CaDiCaL.
4. Block only original cell atom variables, never Tseitin auxiliaries.
5. Replace `_build_cells()` and branch satisfiability checks in `build.py` with typed CNF + PySAT.
6. Run cell and grounding tests.

### Task 3: Compact pair-factor model and exact SDD evaluator

**Files:**
- Create: `src/wfomc/cell_graph/compute_pair_factors.py`
- Modify: `src/wfomc/cell_graph/model.py`
- Modify: `src/wfomc/cell_graph/__init__.py`
- Test: `tests/unit/test_cell_graph_compute_pair_factors.py`

**Steps:**

1. Add tests for exact FMPQ scalar factors, counting-mask factors, false pair conditions, and Tseitin auxiliaries.
2. Replace `TwoTable` with immutable `PairFactor(total_weight, counting_weights)`.
3. Implement exact sparse-factor addition/multiplication using the supplied `ArithmeticContext`.
4. Implement SDD compilation, vtree-scope smoothing, and one batch projection over condition atoms plus requested counting atoms.
5. Ensure unary/diagonal condition atoms have neutral pair weights, off-diagonal original atoms use predicate weights, and auxiliaries use `(1, 1)`.
6. Release every referenced SDD root after materializing pure factors.
7. Run pair-factor tests against native reference enumeration.

### Task 4: Integrate pair factors into cell-graph construction

**Files:**
- Modify: `src/wfomc/cell_graph/build.py`
- Modify: `tests/unit/test_cell_graph_data.py`
- Modify: `tests/unit/test_cell_graph_structure.py`
- Modify: `tests/unary_evidence/test_required_unary_preds.py`

**Steps:**

1. Replace `_build_two_tables()` with `_build_pair_factors()`.
2. Keep the unary-only direct evaluation path without invoking PySDD.
3. Replace fresh profile selector models with the equivalent disjunction of profile formulas so hidden selector assignments cannot multiply WMC values.
4. Build predecessor pair-factor matrices through the same function.
5. Rename `two_tables` to `pair_factors` and `predecessor_two_tables` to `predecessor_pair_factors` everywhere.
6. Preserve `CellGraphData.pair_weights()` as the scalar projection used by ordinary algorithms.
7. Run all cell-graph, evidence, standard, fast, incremental, and recursive tests.

### Task 5: Incremental3 counting projection

**Files:**
- Modify: `src/wfomc/algo/incremental3/input.py`
- Modify: `src/wfomc/cell_graph/build.py`
- Test: `tests/unit/test_incremental3_counting_native.py`
- Test: `tests/test_incremental3_regressions.py`

**Steps:**

1. Pass `state.ext_preds + state.cnt_preds` as `projected_binary_preds` when building the graph.
2. Preserve the existing bit convention: reverse/`ba` is bit `2*i`, forward/`ab` is bit `2*i+1`.
3. Convert each `PairFactor.counting_weights` entry directly into incremental3 forward/reverse deltas.
4. Remove the loop over `state.binary_evidence` and all conditional model filtering.
5. Run incremental3 differential and regression tests.

### Task 6: Cleanup, architecture checks, and benchmarks

**Files:**
- Modify: `tests/unit/test_cell_graph_structure.py`
- Modify: `benchmarks/README.md`
- Consolidate: `benchmarks/cell_graph_backend_spike.py`
- Consolidate: `benchmarks/cell_graph_real_formula_spike.py`
- Create: `docs/adr/0019-pysat-pysdd-cell-graph-backend.md`

**Steps:**

1. Assert the production package contains exactly `build.py`, `data.py`, `enumerate_cells.py`, `compute_pair_factors.py`, and `evidence.py` plus `__init__.py`.
2. Assert only `enumerate_cells.py` imports PySAT and only `compute_pair_factors.py` imports PySDD.
3. Assert production `cell_graph` does not reference `model_literals`, `TwoTable`, or `.models`.
4. Consolidate the two experimental scripts into one maintained cell-graph backend benchmark and retain the experiment documents as historical evidence.
5. Record the dependency and data-boundary decision in ADR 0019.
6. Run `uv run ruff check src tests benchmarks` if Ruff is available.
7. Run `uv run pytest -q` and require the full suite to pass.
8. Run the solver matrix and the consolidated benchmark once as smoke verification.

No git staging or commit is part of this execution because the worktree already contains the user's broader architecture refactor.
