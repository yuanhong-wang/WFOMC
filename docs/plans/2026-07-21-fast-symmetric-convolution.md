# Fast Symmetric-Clique Convolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Share BP-DP's twisted binomial convolution method with Fast and use repeated squaring for homogeneous symmetric cliques.

**Architecture:** Introduce an algorithm-neutral convolver over `ArithmeticContext`, then make Fast operations cache full clique messages instead of evaluating one recursive D-term at a time. Keep boundary-profile files untouched so the later `bp-dp` rebase is conflict-light.

**Tech Stack:** Python 3.11, python-flint, pytest, existing typed WFOMC engine.

---

### Task 1: Shared convolution primitive

**Files:**
- Create: `src/wfomc/algo/symmetric_clique.py`
- Create: `tests/unit/test_symmetric_clique_convolution.py`

1. Add failing coefficient, associativity, symbolic-arithmetic, and combine-count tests.
2. Implement cached binomial coefficients and interaction powers.
3. Implement dense convolution, binary power, and grouped product.
4. Run the focused tests and commit.

### Task 2: Fast integration

**Files:**
- Modify: `src/wfomc/algo/fast/weights.py`
- Modify: `src/wfomc/algo/fast/operations.py`
- Create: `tests/unit/test_fast_symmetric_convolution.py`

1. Add tests comparing cached clique messages with the legacy D-term recurrence.
2. Add per-domain message caches to both operation implementations.
3. Build local rows with Fast's existing weight-placement semantics.
4. Route ordinary and evidence J-terms through grouped convolution products.
5. Run Fast unit/integration tests and commit.

### Task 3: Verify and merge

**Files:**
- No additional production files.

1. Run compile checks, focused tests, full pytest, and Fast benchmark smoke.
2. Confirm `bp-dp`-owned paths are untouched.
3. Switch to `devel` and merge with `--ff-only`.
4. Confirm `devel` is clean and record the resulting commit.
