# Cell-Graph Small-Model and Free-Atom Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Accelerate small pair formulas without regressing large formulas, represent the predicate universe as metadata instead of Tseitin tautologies, and reuse pair factors for equivalent cell-condition signatures.

**Architecture:** Cell enumeration continues to use projected PySAT and large pair formulas continue to use one exact PySDD batch traversal. `compute_pair_factors()` first tries an exact, globally bounded PySAT enumeration over the distinct condition signatures; missing cell and off-diagonal atoms are expanded analytically from the predicate metadata rather than inserted into the formula. The public `CellGraphData` contract does not change.

**Tech Stack:** Python 3.11, python-sat/CaDiCaL, PySDD, python-flint, pytest.

---

### Task 1: Specify free predicate and compact-factor semantics

**Files:**
- Modify: `tests/unit/test_cell_graph_enumerate_cells.py`
- Modify: `tests/unit/test_cell_graph_compute_pair_factors.py`
- Modify: `tests/unit/test_fol_grounding.py`

**Steps:**

1. Add a cell-enumeration test whose predicate universe contains an atom absent from the CNF and require both truth values.
2. Add pair-factor tests for absent ordinary and projected off-diagonal atoms, including symbolic weights.
3. Add a test that cell pairs with the same condition signature share one immutable `PairFactor` instance.
4. Change the grounding expectation so `ground_on_tuple()` returns only atoms that occur after substitution.
5. Run the focused tests with a hard timeout and confirm the new assertions fail before implementation.

### Task 2: Replace formula tautologies with predicate metadata

**Files:**
- Modify: `src/wfomc/fol/grounding.py`
- Modify: `src/wfomc/cell_graph/build.py`
- Modify: `src/wfomc/cell_graph/enumerate_cells.py`

**Steps:**

1. Remove implicit binary-orientation tautologies from `ground_on_tuple()`.
2. Remove `_include_ground_atom_universe()` and ground the diagonal/pair formulas without adding `A | ~A`.
3. Teach `enumerate_cells()` to enumerate present predicate variables with PySAT and expand missing predicate bits explicitly in stable predicate order.
4. Run grounding, cell enumeration, nullary-branch, evidence-profile, and cell-graph semantic tests.

### Task 3: Add bounded PySAT pair-factor enumeration

**Files:**
- Modify: `src/wfomc/cell_graph/compute_pair_factors.py`
- Modify: `tests/unit/test_cell_graph_compute_pair_factors.py`
- Modify: `tests/unit/test_cell_graph_data.py`

**Steps:**

1. Compute distinct condition masks before backend selection.
2. Try exact PySAT enumeration with assumptions, blocking only original atom assignments and enforcing one global model budget.
3. Aggregate the same combined condition/counting masks returned by the SDD evaluator.
4. If the budget is exceeded, discard the partial enumeration and execute the existing PySDD batch path.
5. Multiply either backend's result by the exact factor for absent off-diagonal atoms; projected atoms contribute their existing incremental3 bit positions.
6. Log backend selection, enumerated models, condition signatures, and backend-specific timings.
7. Add an explicit forced-fallback test so the PySDD path remains covered.

### Task 4: Reuse compact condition-signature factors

**Files:**
- Modify: `src/wfomc/cell_graph/compute_pair_factors.py`
- Modify: `tests/unit/test_cell_graph_compute_pair_factors.py`

**Steps:**

1. Materialize one `PairFactor` per distinct valid condition mask.
2. Populate the cell-pair matrix with references to those immutable factors.
3. Preserve the existing tuple matrix API required by algorithms.
4. Run all pair-factor and incremental3 counting tests.

### Task 5: Document and verify the decision

**Files:**
- Create: `docs/adr/0023-bounded-pysat-and-metadata-predicate-universe.md`
- Modify: `docs/adr/0018-cell-graph-predicate-universe.md`

**Steps:**

1. Record why the predicate universe remains authoritative metadata while formula tautologies are removed.
2. Record the bounded PySAT/PySDD fallback rule and why opt27's per-pair CNF storage is not adopted.
3. Run the restored books model with `incremental3 -v`; require result `4` and `backend=pysat-enum`.
4. Run focused tests, the full test suite, package build, and repository diff checks, all with hard timeouts.

No staging or commit is part of this execution because the worktree contains the user's larger refactor.
