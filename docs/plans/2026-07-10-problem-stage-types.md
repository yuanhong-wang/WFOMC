# Problem Stage Types Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Separate public source problems, logical reduction state, and compiled quantifier-free problems without adding algorithm-specific wrapper layers.

**Architecture:** Keep public `Problem`; add internal `ReducedProblem` with a stable `C2NormalForm` field and `CompiledProblem` with a quantifier-free formula plus compiled weights. Reductions operate only on `ReducedProblem`; algorithm preparation derives any special state (notably incremental3 counting state) from the reduced normal form, then combines it with `CompiledProblem` into the existing `AlgoInput` classes.

**Tech Stack:** Python 3.11 dataclasses and typing, FLINT-backed weights, pytest, Ruff.

---

### Task 1: Define stable stage contracts

**Files:**
- Modify: `src/wfomc/problem.py`
- Modify: `tests/unit/reduction/test_core_contract.py`

**Steps:**

1. Add tests asserting `Problem.sentence` is a typed `Formula`, `ReducedProblem.normal_form` is a `C2NormalForm`, and `CompiledProblem.sentence` is a typed quantifier-free `Formula`.
2. Add `ReducedProblem` and `CompiledProblem` as flat frozen dataclasses; do not use inheritance or nested context objects.
3. Add explicit `begin_reduction(Problem) -> ReducedProblem` in `reduction/core.py`; it normalizes and validates once.
4. Run the reduction contract tests and Ruff.

### Task 2: Type the reduction pipeline

**Files:**
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/reduction/reduce_unary_evidence.py`
- Modify: `src/wfomc/reduction/reduce_counting_quantifiers.py`
- Modify: `src/wfomc/reduction/reduce_existential_quantifiers.py`
- Modify: `src/wfomc/reduction/reduce_cardinality_constraints.py`
- Modify: `src/wfomc/reduction/normal_form.py`
- Modify: `src/wfomc/reduction/__init__.py`
- Test: `tests/unit/reduction/`

**Steps:**

1. Make `apply_reductions` convert the source `Problem` once, then pass only `ReducedProblem` branches.
2. Remove `reduce_to_c2_normal_form` from algorithm reduction sequences.
3. Change each reduction to update `normal_form`, raw weights, evidence, and constraints within `ReducedProblem`.
4. Keep decoder composition on branch objects; never copy decoders into problem or algorithm input fields.
5. Add tests that every reduction branch still contains `ReducedProblem` with `C2NormalForm`.
6. Run reduction and engine compile tests.

### Task 3: Compile reduced branches explicitly

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/weights.py`
- Test: `tests/unit/test_engine_compile.py`
- Test: `tests/unit/test_weight_backend_planning.py`

**Steps:**

1. Replace `materialize_problem(Problem) -> Problem` with `compile_reduced_problem(ReducedProblem) -> (CompiledProblem, FeatureSet)`.
2. Extract the QF universal body without changing the reduced object.
3. Compile raw weights and cardinality marker weights into the new compiled object.
4. Re-run feature analysis from the compiled QF formula.
5. Test that raw and compiled weights live on different stage types and the reduced normal form remains unchanged.

### Task 4: Migrate algorithm-owned input construction

**Files:**
- Modify: `src/wfomc/algo/*/spec.py`
- Modify: algorithm input builders under `src/wfomc/algo/*/input.py`
- Modify: `src/wfomc/algo/cell_graph/cache.py`
- Test: `tests/unit/test_engine_compile.py`
- Test: `tests/unary_evidence/test_algorithm_matrix.py`

**Steps:**

1. Make all algorithm builders accept `CompiledProblem` rather than a phase-ambiguous `Problem`.
2. For incremental3, compile `CountingState` and unary masks from `ReducedProblem.normal_form` before building `CountingDPInput`.
3. Keep cell graphs, predecessor tables, ground CNF, and evidence allocations algorithm-owned.
4. Remove `object` annotations that represented these three problem stages.
5. Run algorithm matrix and order/counting regression tests.

### Task 5: Remove transition artifacts and verify

**Files:**
- Modify: `src/wfomc/problem.py`
- Modify: `src/wfomc/reduction/normal_form.py`
- Modify: current architecture documentation if behavior descriptions changed

**Steps:**

1. Remove `profile_capacity_constraint` from public `Problem`; keep it only on reduced/compiled internal stages.
2. Delete obsolete cross-stage helpers and any `replace(problem, sentence=C2NormalForm(...))` path.
3. Confirm `NormalFormReductionView` is no longer required; delete it if all consumers use `C2NormalForm` directly.
4. Run `uv run ruff check src tests benchmarks`.
5. Run `uv run pytest -q`.
6. Run `uv build`, CLI smoke, solver-matrix smoke, and `git diff --check`.
