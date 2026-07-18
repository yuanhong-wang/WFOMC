# Algorithm Capabilities and Typed Boundaries Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make algorithm maturity/external requirements explicit, hide non-runnable algorithms from the CLI, and replace phase-boundary `object` annotations with real contracts.

**Architecture:** Extend the flat `AlgoSpec` contract with an `AlgoMaturity` enum and string requirements; CLI choices derive from registered specs and expose only stable/beta algorithms. Tighten engine, reduction, and algorithm-input builder signatures while retaining `object` only for genuinely polymorphic FLINT values and opaque external-engine payloads.

**Tech Stack:** Python dataclasses/enums/protocol typing, pytest, Ruff.

---

### Task 1: Add maturity and requirement metadata

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/algo/*/spec.py`
- Test: `tests/unit/test_engine_compile.py`

**Steps:**

1. Add failing registry tests for stable, beta, and unavailable specs.
2. Add `AlgoMaturity` and `AlgoSpec.external_requirements`.
3. Mark propositional beta with Ganak and treewidth unavailable.
4. Run registry tests and Ruff.

### Task 2: Derive CLI choices from capabilities

**Files:**
- Modify: `src/wfomc/cli.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `README.md`

**Steps:**

1. Add a failing test that treewidth is absent from CLI choices.
2. Filter CLI algorithms to stable/beta specs.
3. Keep propositional visible and document its Ganak requirement.
4. Run CLI tests and smoke help.

### Task 3: Tighten architecture boundary annotations

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/reduction/reduce_*.py`
- Modify: `src/wfomc/algo/*/input.py`
- Modify: `src/wfomc/algo/*/spec.py`
- Modify: `src/wfomc/engine/orchestration.py`

**Steps:**

1. Type `AlgoInput`, `PreparedBranch`, spec prepare functions, and source/cache helpers.
2. Type reduction options/results with `AlgoOptions`, `ReducedProblem`, and branch contracts.
3. Type input builders with `CompiledProblem`, `FeatureSet`, and algorithm-specific state classes.
4. Remove ignored `**_: object` parameters where no caller needs them.
5. Keep opaque numeric and external-engine values explicitly documented rather than inventing fake types.

### Task 4: Verify contracts and public behavior

**Files:**
- Modify: `docs/adr/0008-explicit-problem-stage-types.md`
- Modify: `docs/architecture-review-2026-07-10.md`

**Steps:**

1. Run Ruff and the full test suite.
2. Build package and run CLI help/standard/incremental3 smoke tests.
3. Run solver-matrix smoke.
4. Run `git diff --check` and audit remaining `object` annotations to ensure they are implementation values rather than stage boundaries.
