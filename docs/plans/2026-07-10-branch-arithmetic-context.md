# Branch Arithmetic Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make one backend-bound `ArithmeticContext` the sole numeric factory for each prepared branch from weight compilation through solving and decoding.

**Architecture:** Logical reductions keep raw values. After an algorithm finishes logical reduction for one branch, compilation collects that branch's complete solver symbol set, creates one immutable context, and stores the same instance on `CompiledProblem` and its `AlgoInput`. Cell-graph construction, solver kernels, and decoders receive that context explicitly; different branches and algorithms may use different contexts.

**Tech Stack:** Python dataclasses, python-flint numeric backends, pytest, Ruff.

---

### Task 1: Establish branch ownership

**Files:**
- Modify: `src/wfomc/problem.py`
- Modify: `src/wfomc/algo/core.py`
- Modify: `tests/unit/test_engine_compile.py`

**Steps:**

1. Add tests asserting `CompiledProblem.arithmetic is AlgoInput.arithmetic` and distinct prepared branches may own distinct contexts.
2. Add `arithmetic: ArithmeticContext` to `CompiledProblem` and `AlgoInput`.
3. Split branch planning from weight compilation so `compile_reduced_problem()` creates the context exactly once after reductions.
4. Run engine compilation tests.

### Task 2: Thread context through materialization

**Files:**
- Modify: `src/wfomc/algo/*/input.py`
- Modify: `src/wfomc/algo/cell_graph/cache.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Modify: `src/wfomc/cardinality.py`

**Steps:**

1. Pass `CompiledProblem.arithmetic` into every algorithm input.
2. Replace default cell-graph weights, graph weights, and cardinality marker constants with context-created values.
3. Add exact and rounded materialization tests that assert every numeric value is compatible with the selected backend.
4. Run cell-graph, cardinality, and compilation tests.

### Task 3: Thread context through solver kernels

**Files:**
- Modify: `src/wfomc/algo/standard/solve.py`
- Modify: `src/wfomc/algo/fast/solve.py`
- Modify: `src/wfomc/algo/fast/operations.py`
- Modify: `src/wfomc/algo/incremental/solve.py`
- Modify: `src/wfomc/algo/incremental3/solve.py`
- Modify: `src/wfomc/algo/incremental3/counting_kernel.py`
- Modify: `src/wfomc/algo/recursive/solve.py`

**Steps:**

1. Replace direct `Rational(0, 1)` and `Rational(1, 1)` construction with the branch context.
2. Coerce multinomial and factorial coefficients through the context before multiplication.
3. Pass context explicitly into helper kernels that create accumulators.
4. Run each algorithm's focused tests after migration.

### Task 4: Thread context through decoding and results

**Files:**
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/reduction/reduce_cardinality_constraints.py`
- Modify: `src/wfomc/engine/orchestration.py`
- Modify: `src/wfomc/result.py`

**Steps:**

1. Pass `arithmetic` to branch decoders from engine orchestration.
2. Make correction-factor and cardinality decoding use the supplied context rather than hard-coded FLINT scalars.
3. Extend result handling only for backend values actually produced by supported rounded runs.
4. Add decoder and result regression tests.

### Task 5: Enable supported rounded backends

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/weights.py`
- Modify: `tests/unit/test_weight_backend_planning.py`
- Modify: `tests/unit/test_arithmetic_context.py`
- Modify: algorithm matrix tests

**Steps:**

1. Remove the blanket rounded-arithmetic rejection after migrated exact tests pass.
2. Enable scalar FLOAT/ARB and single-symbol ARB_POLY only where the complete pipeline supports them.
3. Keep explicit early rejection for float symbolic weights, rounded multivariate weights, and exact-only external serializers.
4. Compare exact and rounded results across directly runnable algorithms with tolerances appropriate to each backend.
5. Run Ruff, full pytest, build, CLI smoke, and solver-matrix smoke.

### Task 6: Record the invariant

**Files:**
- Create: `docs/adr/0011-branch-owned-arithmetic-context.md`
- Modify: `docs/architecture-review-2026-07-10.md`
- Modify: `docs/README.md`

**Steps:**

1. Document one context per prepared branch and the distinction between solver symbols and output symbols.
2. Record genuinely unsupported backend/symbol combinations.
3. Update the architecture review only after end-to-end tests pass.
