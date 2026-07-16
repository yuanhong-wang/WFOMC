# Direct Propositional Grounding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the propositional algorithm ground the public source `Problem` directly, without normalization or logical reductions, and count the resulting model-preserving CNF with Ganak.

**Architecture:** `wfomc.fol.grounding` recursively expands finite-domain ordinary and counting quantifiers into a closed quantifier-free formula, then the existing bidirectional Tseitin encoder produces CNF. The propositional input builder appends ground evidence, order semantics, and direct combination clauses for simple global `|P| <=/=/>= k` constraints; the branch keeps the source problem and an identity decoder.

**Tech Stack:** Python 3.11, typed FOL IR, itertools combination/product grounding, bidirectional Tseitin CNF, Ganak exact WMC, python-flint, pytest.

---

### Task 1: Specify direct finite-domain formula grounding

**Files:**
- Modify: `tests/unit/test_fol_grounding.py`
- Modify: `src/wfomc/fol/grounding.py`

**Steps:**

1. Add failing tests for nested universal/existential quantifiers, equality, every counting comparator, empty-domain semantics, and formulas with more than two variables.
2. Add failing truth-table tests for auxiliary-free at-most, at-least, and exactly-k ground clauses, including boundary values.
3. Implement recursive source-formula grounding with environment-aware variable substitution.
4. Implement counting quantifier expansion using direct finite-domain cardinality formulas.
5. Implement direct combination CNF helpers for global cardinality constraints.
6. Run `tests/unit/test_fol_grounding.py`.

### Task 2: Keep Tseitin expansion model-preserving and compact

**Files:**
- Modify: `src/wfomc/fol/cnf.py`
- Modify: `tests/unit/test_fol_cnf.py`

**Steps:**

1. Add a failing test showing repeated grounded subformulas should share one uniquely defined auxiliary variable.
2. Memoize non-atomic formula encodings while retaining bidirectional definitions and the asserted root.
3. Exhaustively verify one CNF extension per satisfying assignment of original atoms.
4. Run the focused CNF tests.

### Task 3: Build propositional input directly from `Problem`

**Files:**
- Modify: `src/wfomc/algo/propositional/input.py`
- Modify: `src/wfomc/algo/propositional/spec.py`
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/reduction/core.py`
- Modify: `tests/unit/test_engine_compile.py`

**Steps:**

1. Add a failing compile test requiring the prepared propositional branch to retain the original source problem and counting quantifiers.
2. Add tests requiring unary and binary evidence to become unit clauses.
3. Compile source weights into a branch arithmetic context without internal reduction markers.
4. Ground the source sentence once, allocate all predicate/evidence/cardinality atoms, and append direct clauses.
5. Replace the propositional reduction chain with one source branch and `identity_decoder`.
6. Add a `ground` existential strategy and binary-evidence capability declaration for the direct algorithm.
7. Run engine compile and option-resolution tests.

### Task 4: Support scoped global cardinality constraints

**Files:**
- Modify: `src/wfomc/algo/propositional/input.py`
- Modify: `tests/unit/test_engine_compile.py`
- Modify: `tests/integration/test_propositional.py`

**Steps:**

1. Add compile/count tests for unary and binary `|P| <= k`, `|P| = k`, and `|P| >= k` constraints.
2. Ensure predicates mentioned only by a cardinality constraint receive their full ground atom universe and declared weights.
3. Reject non-unit coefficients, multi-predicate linear expressions, and unsupported global comparators with `UnsupportedFeatureError`.
4. Run focused unit and Ganak integration tests.

### Task 5: Cross-check direct semantics

**Files:**
- Modify: `tests/integration/test_propositional.py`

**Steps:**

1. Add the counting-quantifier model corpus to propositional/reference cross-checks.
2. Add direct tests for nested/embedded counts, modulo counts, symbolic weights, and ground evidence.
3. Verify exact agreement with incremental3 or another supported lifted algorithm on small domains.
4. Run all propositional integration tests with a hard timeout.

### Task 6: Complete regression verification

**Files:**
- Modify: `docs/README.md`

**Steps:**

1. Record the direct propositional path and its intentionally exponential global-cardinality clause counts.
2. Run focused grounding, CNF, engine, CLI, arithmetic, and propositional tests.
3. Run the complete pytest suite with a hard timeout.
4. Run `compileall`, `git diff --check`, and inspect the final diff for accidental changes.
