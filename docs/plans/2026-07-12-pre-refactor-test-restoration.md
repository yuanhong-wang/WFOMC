# Pre-Refactor Test Restoration and Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the behavior coverage from `/Users/lucien/Sync/repos/wfoMC/tests`, run it against the refactored API, and compare exact results and runtime with the pre-refactor repository.

**Architecture:** The baseline repository remains read-only and supplies test intent plus timing/reference results. Tests already migrated in the current worktree are retained; only missing behavior suites are restored, with imports and setup translated to the typed `Problem` / `AlgoName` / `solve` API. A deterministic parity report records identical model inputs, exact output comparisons, skips, failures, and wall-clock timings.

**Tech Stack:** Python 3.11, pytest, uv, python-flint exact arithmetic, GANAK, `/usr/bin/time`.

**Status:** Completed on 2026-07-12. All diagnostic and verification runs after
the initial inventory used hard shell timeouts. Results are recorded in
`docs/experiments/pre-refactor-test-parity-2026-07-12.md`.

---

### Task 1: Record the untouched baseline

**Files:**
- Read: `/Users/lucien/Sync/repos/wfoMC/tests/**`
- Create: `docs/experiments/pre-refactor-test-parity-2026-07-12.md`

**Steps:**

1. Verify the baseline worktree is clean and both repositories contain identical model files.
2. Run the untouched baseline suite with `PYTHONHASHSEED=0` and record pass/fail/skip counts and wall time.
3. Classify environment failures separately from semantic assertion failures.
4. Run a comparable baseline subset or explicit model/algorithm matrix that excludes the broken optional `pynauty.Graph` path.

### Task 2: Restore broad solver and MATH behavior tests

**Files:**
- Restore and migrate: `tests/wfomc_test.py`
- Test: existing `tests/test_incremental3_regressions.py`

**Steps:**

1. Recreate the pre-refactor model-family matrix with `AlgoName` and `solve`.
2. Preserve public `WFOMCResult`, symbolic-term, unsatisfiable-result, and MATH-answer assertions.
3. Run the restored file and classify unsupported capability combinations rather than weakening expected exact results.
4. Fix production regressions if any exact-result assertion differs.

### Task 3: Restore evidence and order behavior tests

**Files:**
- Restore and migrate: `tests/unary_evidence/test_linear_order.py`
- Restore and migrate: `tests/unary_evidence/test_unary_evidence_partition.py`

**Steps:**

1. Translate old `UnaryEvidencePartition` assertions to the authoritative `ProfileCapacityConstraint` reduction output.
2. Preserve deterministic grouping, negative literals, empty evidence, size validation, and consistency checks.
3. Restore the linear-order/evidence cross-algorithm result assertions through `AlgoOptions` and `EvidenceStrategy`.
4. Run evidence and order tests.

### Task 4: Restore propositional cross-checks

**Files:**
- Restore and migrate: `tests/propositional_test.py`

**Steps:**

1. Translate GANAK discovery and model feature checks to current `FeatureSet` / `AlgoName` APIs.
2. Compare the propositional counter with the appropriate lifted algorithm over the original model corpus.
3. Keep explicit skips only for unsupported `PREDk`, unavailable GANAK, or opt-in slow cases.
4. Run the restored propositional suite and investigate every exact mismatch.

### Task 5: Compare timing and publish evidence

**Files:**
- Complete: `docs/experiments/pre-refactor-test-parity-2026-07-12.md`
- Modify: `docs/README.md`

**Steps:**

1. Run restored semantic suites in both repositories where API/environment paths are comparable.
2. Run the full refactored suite three times and record median wall time.
3. Record baseline limitations, exact result agreement, restored test count, current pass/skip/fail counts, and runtime comparison.
4. Run Ruff checks, solver smoke matrix, and package build.

No baseline files, git staging, or commits are modified by this plan.
