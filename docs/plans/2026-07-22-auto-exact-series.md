# Automatic Exact Arithmetic and Truncated-Series Plan

**Goal:** Select the smallest exact symbolic backend automatically and use
true truncated univariate series for domain-bounded cardinality markers,
without making compiled branches or BP trees domain-specific.

**Architecture:** Domain-free compilation selects `fmpq` for zero symbols,
`fmpq_poly` for one symbol, and `fmpq_mpoly` for multiple symbols. A compiled
branch records whether this selection came from the automatic policy. During
concrete-domain instantiation only, an automatically selected univariate
polynomial with a finite marker degree limit is rebound to `fmpq_series` at
precision `limit + 1`. Compiled weights and reusable algorithm templates stay
domain-independent; only concrete arithmetic values and execution caches use
the series precision.

**Tech Stack:** Python 3.11, python-flint, pytest, Ruff.

---

### Task 1: Expose automatic exact backend selection

**Files:**

- Modify: `src/wfomc/options.py`
- Modify: `src/wfomc/cli.py`
- Modify: `src/wfomc/arithmetic.py`
- Test: `tests/unit/test_weights.py`
- Test: `tests/unit/test_cli.py`

**Steps:**

1. Add `auto` as the default exact symbolic backend policy.
2. Route zero, one, and multiple exact symbols to scalar, univariate, and
   multivariate FLINT backends respectively.
3. Preserve explicit `fmpq_poly` validation and explicit `fmpq_mpoly` behavior.

### Task 2: Add a bounded univariate series execution backend

**Files:**

- Modify: `src/wfomc/arithmetic.py`
- Modify: `src/wfomc/weights.py`
- Test: `tests/unit/test_arithmetic_context.py`
- Test: `tests/unit/test_weights.py`

**Steps:**

1. Add internal `FMPQ_SERIES` arithmetic backed by `flint.fmpq_series`.
2. Require exactly one symbolic variable and a finite degree limit.
3. Mint constants and the generator at precision `limit + 1`.
4. Coerce cached `fmpq_poly` and one-variable `fmpq_mpoly` values into the
   bounded series ring.
5. Verify multiplication and exponentiation discard high-degree terms during
   arithmetic, rather than after constructing the full product.

### Task 3: Switch backend only at domain instantiation

**Files:**

- Modify: `src/wfomc/engine/compilation.py`
- Modify: `src/wfomc/reduction/cardinality.py`
- Modify: `src/wfomc/result.py`
- Test: `tests/unit/test_engine_compile.py`
- Test: `tests/integration/test_boundary_profile.py`

**Steps:**

1. Mark auto-selected domain-free arithmetic as series-eligible.
2. Upgrade only eligible, finitely bounded, one-symbol concrete contexts to
   `FMPQ_SERIES` after evaluating the domain expression.
3. Decode cardinality coefficients and project constants without leaking the
   internal series backend through the public result.
4. Confirm the same compiled branch is instantiated at different series
   precisions for different domain sizes.

### Task 4: Keep exact algorithm compatibility

**Files:**

- Modify as needed: `src/wfomc/algo/propositional/`
- Test: existing algorithm integration suites

**Steps:**

1. Make exact backend capability checks recognize the internal series form.
2. Convert series weights into the shared univariate Ganak polynomial context
   where propositional counting requires polynomial serialization.
3. Coerce the returned polynomial back into the concrete series context.

### Task 5: Verify correctness and performance path

**Steps:**

1. Run focused arithmetic, weight, compilation, cardinality, CLI, and BP tests.
2. Run the complete test suite and Ruff.
3. Run representative cardinality cases on Fast, Incremental3, and BP-DP and
   compare results with explicit `fmpq_mpoly`.
4. Run `git diff --check` and inspect the final diff without touching unrelated
   benchmark cleanup changes already present in the worktree.
