# Benchmark Case Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a typed catalog containing every benchmark case family and size
grid from `new_WFOMC_with_notes` without migrating its historical solver
baselines.

**Architecture:** `benchmarks/cases.py` expands reusable formula definitions
into immutable `BenchmarkCase` values that build current `Problem` objects.
Tests protect suite membership, metadata, constraints, and deterministic
ordering; no benchmark runner or production runtime API is added.

**Tech Stack:** Python 3.11, dataclasses, current WFOMC typed
FOL/problem/cardinality APIs, pytest.

---

### Task 1: Specify the catalog contract

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `tests/unit/test_benchmark_cases.py`

**Steps:**

1. Add failing tests requiring 4 core-smoke, 16 core-main, 49
   core-exhaustive, 47 C2, 11 cardinality, and 54 unary cases.
2. Require globally unique stable keys and deterministic ordering.
3. Require each case to build a typed `Problem` with its declared domain size.
4. Require constraints and weight keys to reference declared predicates.

### Task 2: Port the core transformed-FO2 families

**Files:**
- Create: `benchmarks/cases.py`
- Test: `tests/unit/test_benchmark_cases.py`

**Steps:**

1. Implement the `BenchmarkCase` and internal formula-definition records.
2. Port row/column, k-regular, k-coloured, derangement, k-matching,
   total-mapping, and no-isolated-digraph definitions.
3. Expand the original smoke, main, and exhaustive size grids exactly.
4. Run focused catalog tests.

### Task 3: Port C2, cardinality, and unary variants

**Files:**
- Modify: `benchmarks/cases.py`
- Modify: `tests/unit/test_benchmark_cases.py`

**Steps:**

1. Add direct C2 regular/coloured/directed cases and hand-cardinality cases.
2. Add properly-coloured and directed binary-cardinality cases with reporting
   divisors.
3. Add unary unconstrained/exact/interval grids and the larger structure/gate
   workloads as case metadata, without migrating old optimization modes.
4. Verify representative small exact results through current algorithms.

### Task 4: Document and verify

**Files:**
- Modify: `docs/plans/2026-07-13-benchmark-case-migration-design.md`

**Steps:**

1. Document suite names/counts, case metadata, problem construction, and the
   explicit exclusion of historical baselines/runners in the design note and
   module interface documentation.
2. Run focused and full pytest with hard timeouts.
3. Run four `PYTHONHASHSEED` catalog-order checks.
4. Run compileall, `uv build`, and `git diff --check`.
5. Record that Ruff/mypy are unavailable rather than adding new tooling to the
   project for this migration.
6. Do not stage or commit because the worktree contains the user's larger
   active refactor.
