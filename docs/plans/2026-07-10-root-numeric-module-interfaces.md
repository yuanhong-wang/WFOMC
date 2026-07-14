# Root Numeric Module Interfaces Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give root numeric modules single, explicit responsibilities, delete the obsolete `utils` polynomial/rational compatibility layer, and expose the remaining combinatorial helpers directly from `multinomial.py`.

**Architecture:** `arithmetic.py` owns backend identity, backend selection, the complete compiled-value union, and `ArithmeticContext`. `weights.py` owns user weight options, raw/compiled weight mappings, symbol discovery, and compilation through a supplied context. `result.py` owns public result inspection and returns standard-library `Fraction` for exact constants. Dependencies flow `weights -> arithmetic`; `arithmetic` does not import `weights`.

**Tech Stack:** Python dataclasses/type aliases, `fractions.Fraction`, python-flint, pytest, Ruff.

---

### Task 1: Fix the arithmetic root interface

**Files:**
- Modify: `src/wfomc/arithmetic.py`
- Modify: `src/wfomc/weights.py`
- Test: `tests/unit/test_arithmetic_context.py`
- Test: `tests/unit/test_weights.py`

**Steps:**

1. Add interface tests for each module's `__all__` and the one-way dependency.
2. Move `ArithmeticBackend` and backend selection to `arithmetic.py`.
3. Add a complete `ArithmeticValue` union covering scalar, polynomial, and multivariate FLINT values plus float.
4. Remove `WeightPlan`; construct `ArithmeticContext` directly from backend, solver symbols, and output symbols.
5. Remove unused `build_weight_plan`, `build_arithmetic_context`, and legacy ring conversion APIs.

### Task 2: Remove the custom rational compatibility type

**Files:**
- Modify: parser transformers, reductions, arithmetic, results, and tests
- Delete: `src/wfomc/utils/rational.py`

**Steps:**

1. Replace raw/reduction rational values with `fractions.Fraction`.
2. Make `ArithmeticContext.coerce()` accept `Fraction` directly.
3. Make `WFOMCResult.constant_value()` and exact terms return `Fraction`.
4. Preserve comparison with exact FLINT values in `WFOMCResult.__eq__`.
5. Delete the custom subclass and all exports/imports.

### Task 3: Delete the polynomial utility bucket

**Files:**
- Modify: `src/wfomc/algo/propositional/counting.py`
- Modify: numeric type annotations throughout algorithms/cell graph
- Create: `src/wfomc/multinomial.py`
- Delete: `src/wfomc/utils/__init__.py`
- Delete: `src/wfomc/utils/polynomial_flint.py`

**Steps:**

1. Move the only live context-alignment helper beside the Ganak caller.
2. Replace the incomplete `RingElement` alias with `ArithmeticValue`.
3. Delete the identity `expand()` call and unused filtering/sampling helpers.
4. Replace test-only `create_vars()` with direct FLINT context construction.
5. Move the three live combinatorial helpers to root `multinomial.py`, update imports,
   and delete the now-empty `utils` package.
6. Delete `polynomial_flint.py` and remove its exports.

### Task 4: Document and verify root boundaries

**Files:**
- Create: `docs/adr/0012-root-numeric-module-interfaces.md`
- Modify: `docs/README.md`
- Modify: `docs/architecture-review-2026-07-10.md`

**Steps:**

1. Record public interfaces and allowed dependency direction.
2. Audit root modules for undocumented re-exports and old utility imports.
3. Run Ruff, full pytest, build, CLI smoke, and solver-matrix smoke.
