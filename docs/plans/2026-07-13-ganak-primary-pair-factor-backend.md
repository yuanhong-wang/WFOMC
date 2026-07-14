# Ganak Primary Pair-Factor Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Use one exact Ganak polynomial WMC as the primary non-enumerating cell-graph pair-factor backend, while preserving the current exact PySDD evaluator as an automatic backup.

**Architecture:** The bounded PySAT path remains the first choice for small local pair theories. When PySAT is skipped or exceeds its model budget, `compute_pair_factors()` calls a dedicated Ganak adapter that marks condition and incremental3 projection bits with polynomial variables and returns the existing sparse factor map. Unsupported arithmetic, a missing Ganak binary, timeout, process failure, or output-parse failure falls back to the current in-process PySDD traversal without changing `CellGraphData` or algorithm inputs.

**Tech Stack:** Python 3.11, python-sat/CaDiCaL, Ganak exact rational/polynomial WMC, python-flint, PySDD, pytest.

---

### Task 1: Specify backend selection and exact polynomial behavior

**Files:**
- Create: `tests/unit/test_cell_graph_ganak_pair_factors.py`
- Modify: `tests/unit/test_cell_graph_compute_pair_factors.py`

**Steps:**

1. Add a failing test that forces the PySAT budget to zero, supplies a successful Ganak factor, and requires that `_SddCircuit` is not constructed.
2. Change the existing forced-PySDD test so Ganak raises `GanakError`; require an exact PySDD result.
3. Add a shared-signature test requiring one Ganak call and reuse of one immutable `PairFactor` for equivalent cell pairs.
4. Add direct Ganak-adapter tests for exact `FMPQ`, one-symbol `FMPQ_POLY`, and multi-symbol `FMPQ_MPOLY` coefficient recovery.
5. Add a rounded-arithmetic test requiring the Ganak adapter to report unsupported and leave computation to PySDD.
6. Run the focused tests with a hard timeout and confirm the new selection tests fail before implementation.

### Task 2: Extract and reuse the generic Ganak boundary

**Files:**
- Create: `src/wfomc/ganak.py`
- Delete: `src/wfomc/algo/propositional/ganak.py`
- Modify: `src/wfomc/errors.py`
- Modify: `src/wfomc/algo/propositional/counting.py`
- Modify: `tests/unit/test_ganak.py`
- Modify: `tests/integration/test_propositional.py`

**Steps:**

1. Move binary discovery, DIMACS serialization, deterministic subprocess execution, exact output parsing, pinned-commit metadata, and `ganak_count()` to root-level `wfomc.ganak`.
2. Move `GanakError` into the shared `wfomc.errors` hierarchy as an `ExternalToolError` subclass.
3. Update the propositional algorithm, tests, benchmark scripts, and installer to import the root adapter.
4. Delete the algorithm-owned Ganak module so `cell_graph` never imports `algo.propositional`.
5. Run Ganak and propositional tests with a hard timeout.

### Task 3: Implement one-shot Ganak pair-factor evaluation

**Files:**
- Create: `src/wfomc/cell_graph/compute_pair_factors_ganak.py`
- Use: `src/wfomc/ganak.py`

**Steps:**

1. Expose one narrow function accepting `TseitinCNF`, literal-weight pairs, the existing variable-to-factor-bit projection, `ArithmeticContext`, and a timeout.
2. Support exact `FMPQ`, `FMPQ_POLY`, and `FMPQ_MPOLY` contexts; return `None` for rounded or otherwise unsupported arithmetic.
3. Convert existing exact weight variables and fresh condition/projection markers into one FLINT multivariate-polynomial context.
4. Invoke repository-pinned Ganak once in deterministic mode through `ganak_count()`; use rational mode when no polynomial variables are required and polynomial mode otherwise.
5. Split Ganak's polynomial terms back into the existing `dict[int, ArithmeticValue]`, preserving the branch-owned arithmetic type and rejecting non-multilinear marker output.
6. Run the new adapter tests with a hard timeout.

### Task 4: Make Ganak primary and PySDD backup

**Files:**
- Modify: `src/wfomc/cell_graph/compute_pair_factors.py`
- Modify: `tests/unit/test_cell_graph_compute_pair_factors.py`

**Steps:**

1. Keep the bounded PySAT fast path unchanged.
2. When PySAT returns no complete factor, call the Ganak pair adapter with a fixed 30-second safety timeout.
3. On unsupported arithmetic or `GanakError`, log the fallback reason and execute the existing exact `_SddCircuit` path.
4. Report `backend=ganak` or `backend=pysdd-backup` plus backend-specific timings in the existing INFO diagnostic.
5. Keep free off-diagonal multiplication, condition-factor materialization, projected-bit order, and immutable `PairFactor` reuse common to all backends.
6. Run all cell-graph, incremental3, arithmetic-context, and logging tests with hard timeouts.

### Task 5: Repair the optional Ganak installation boundary

**Files:**
- Modify: `scripts/tools/install_ganak.py`
- Modify: `tests/unit/test_ganak.py`

**Steps:**

1. Replace the deleted legacy Ganak import in the installer with the shared `wfomc.ganak` module.
2. Add an import-level regression test for the installer entry point and its pinned commit.
3. Confirm a missing binary remains non-fatal for cell-graph algorithms because PySDD is installed and retained as backup.
4. Run Ganak adapter and installer tests with a hard timeout.

### Task 6: Record the architecture decision

**Files:**
- Create: `docs/adr/0024-ganak-primary-pair-factor-backend.md`
- Modify: `docs/adr/0019-pysat-pysdd-cell-graph-backend.md`
- Modify: `docs/adr/0023-bounded-pysat-and-metadata-predicate-universe.md`
- Modify: `docs/README.md`

**Steps:**

1. Record the routing order `bounded PySAT -> Ganak -> PySDD backup`.
2. Record exact-arithmetic support, optional external-binary behavior, timeout policy, and shared-signature polynomial semantics.
3. Link the benchmark evidence in `docs/experiments/cell-graph-pysdd-ganak-polynomial-2026-07-13.md`.
4. State that factor-entry count and cell count are not sufficient backend selectors; Ganak is selected as the robust non-enumerating default based on observed structural PySDD blow-ups.

### Task 7: Complete regression verification

**Files:**
- Verify only; no production files added in this task.

**Steps:**

1. Run focused Ganak and cell-graph tests with `/opt/homebrew/bin/timeout`.
2. Run representative books, predecessor, and 3Markov models with INFO logging; require unchanged exact results and `backend=ganak` when the PySAT path is forced or naturally bypassed.
3. Run the complete pytest suite with a hard timeout.
4. Run four `PYTHONHASHSEED` variants of the semantic/grounding tests.
5. Run `ruff`, `compileall`, `uv build`, and `git diff --check`, all with hard timeouts where applicable.
6. Do not stage or commit: this worktree contains the user's larger active refactor.
