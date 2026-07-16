# Dual Propositional Modes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Preserve direct source grounding as `propositional` and restore the former reduction-first implementation as `propositional-reduced`.

**Architecture:** Register a second public `AlgoName` and `AlgoSpec`. The new spec reuses the existing Ganak solver and `GroundCNFInput`, but prepares that input through normalization, logical reductions, `CompiledProblem`, and `ground_qf_formula`; the existing direct spec remains unchanged.

**Tech Stack:** Python 3.11, pytest, python-flint, Ganak, the existing WFOMC algorithm registry.

---

### Task 1: Specify the public algorithm contract

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/unit/test_engine_compile.py`

**Step 1: Write failing registry and CLI tests**

Assert that `AlgoName.PROPOSITIONAL_REDUCED` has value `propositional-reduced`, resolves to a beta Ganak-backed spec, and appears in the CLI choices.

**Step 2: Write a failing preparation test**

Compile one source counting-quantifier problem with both modes. Assert that `propositional` retains the source `Problem`, while `propositional-reduced` returns a `ReducedProblem` whose compiled CNF input identifies the reduced algorithm.

**Step 3: Run the focused tests and verify failure**

Run: `uv run pytest tests/unit/test_cli.py tests/unit/test_engine_compile.py -q`

Expected: failure because `PROPOSITIONAL_REDUCED` is not registered.

### Task 2: Restore reduction-first preparation

**Files:**
- Create: `src/wfomc/algo/propositional/reduced_input.py`
- Create: `src/wfomc/algo/propositional/reduced_spec.py`
- Modify: `src/wfomc/algo/core.py`

**Step 1: Restore the reduced input builder**

Build `GroundCNFInput` from `CompiledProblem` with `ground_qf_formula`, ground unary-evidence unit clauses when requested, add order clauses, and assign compiled predicate weights.

**Step 2: Restore the reduction pipeline**

Apply unary-evidence, counting-quantifier, existential, and cardinality reductions; compile each branch; build one ground-CNF input per branch; retain the composed decoder.

**Step 3: Register the new spec**

Map `AlgoName.PROPOSITIONAL_REDUCED` to the new spec while leaving `AlgoName.PROPOSITIONAL` mapped to direct grounding.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_cli.py tests/unit/test_engine_compile.py tests/unit/test_fol_grounding.py -q`

Expected: all pass.

### Task 3: Cross-check semantics and document the two modes

**Files:**
- Modify: `tests/integration/test_propositional.py`
- Modify: `README.md`
- Modify: `docs/README.md`

**Step 1: Add integration comparisons**

For representative ordinary, existential, counting, and simple global-cardinality inputs supported by both paths, assert direct and reduced propositional results are equal.

**Step 2: Document selection semantics**

Describe `propositional` as direct source grounding and `propositional-reduced` as normalization/reduction followed by quantifier-free grounding. Note that both use Ganak and that the reduced path inherits reduction feature limits.

**Step 3: Run propositional and full test suites**

Run: `uv run pytest tests/integration/test_propositional.py -q`

Then run: `uv run pytest -q`

Expected: all tests pass, apart from existing environment-dependent skips.
