# Arithmetic Context Backend Enforcement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make user-selected `WeightOptions` control every numeric value created by WFOMC, including compiled weights, algorithm temporaries, coefficients, defaults, and decode factors.

**Architecture:** Introduce an explicit `ArithmeticContext` built from `WeightPlan`. Engine/materialization constructs one arithmetic context per reduced problem branch and passes it into every algorithm input. Algorithms must create `zero`, `one`, integers, rationals, multinomial coefficients, and default weights only through this context. Delete legacy arithmetic selection APIs instead of preserving compatibility.

**Tech Stack:** Python 3.11, dataclasses, python-flint (`fmpz`, `fmpq`, `arb`, `fmpz_poly`, `fmpq_poly`, `arb_poly`, `fmpz_mpoly`, `fmpq_mpoly`), existing `wfomc.weights`, `wfomc.algo`, `wfomc.engine`, typed `wfomc.problem.Problem`.

---

## Decision

The current code only applies `WeightOptions` at the raw-weight compilation boundary. That is not enough. Algorithms still directly create `fmpq(0, 1)`, `fmpq(1, 1)`, `Rational(...)`, and `float(...)`-adjacent values, so rounded backends or future integer/arb backends are not enforced globally.

Target invariant:

```text
If a value participates in WFOMC arithmetic, it must be created or coerced by ArithmeticContext.
```

This includes:

- raw predicate weights;
- default missing predicate weights;
- graph weights;
- cell weights;
- pair weights;
- nullary weights;
- multinomial coefficients;
- evidence assignment coefficients;
- counting-DP dynamic-programming totals;
- decode repeat factors;
- cardinality encoding coefficients;
- solver result accumulators.

## No Legacy Compatibility

Do not preserve the old arithmetic API.

Delete:

```python
AlgoOptions.arithmetic_backend
SymbolicWeightPlan
choose_arithmetic_backend
ArithmeticBackend.RATIONAL
ArithmeticBackend.FLINT_POLY
ArithmeticBackend.FLINT_MPOLY
ArithmeticBackend.TRUNCATED_SERIES
ArithmeticBackend.PY_INT
ArithmeticBackend.PY_FLOAT  # replace with FLOAT
```

Keep only the backend names used by the new `WeightPlan`:

```python
class ArithmeticBackend(Enum):
    FMPZ = "fmpz"
    FMPQ = "fmpq"
    FLOAT = "float"
    ARB = "arb"
    FMPZ_POLY = "fmpz_poly"
    FMPQ_POLY = "fmpq_poly"
    ARB_POLY = "arb_poly"
    FMPZ_MPOLY = "fmpz_mpoly"
    FMPQ_MPOLY = "fmpq_mpoly"
```

Keep `ARB_MPOLY` out until python-flint supports it, or define it only as an unsupported error target.

Delete tests that exist only to pin legacy `arithmetic_backend` behavior. Rewrite tests around `WeightOptions`, `WeightPlan`, and `ArithmeticContext`.

## Target Public Options

`AlgoOptions` should expose:

```python
@dataclass(frozen=True)
class AlgoOptions:
    evidence_strategy: EvidenceStrategy | str | None = None
    linear_order_encoding: LinearOrderEncoding | str | None = None
    weight_options: WeightOptions = field(default_factory=WeightOptions)
```

No `arithmetic_backend` field.

`WeightOptions` should stay user-level:

```python
@dataclass(frozen=True)
class WeightOptions:
    precision: Literal["exact", "round"] = "exact"
    rounded_backend: Literal["float", "arb"] = "arb"
```

`WeightPlan` should be internal/resolved:

```python
@dataclass(frozen=True)
class WeightPlan:
    options: WeightOptions
    symbolic_variables: tuple[str, ...]
    backend: ArithmeticBackend
```

## ArithmeticContext Contract

Create `src/wfomc/arithmetic.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from wfomc.weights import ArithmeticBackend, WeightPlan


@dataclass(frozen=True)
class ArithmeticContext:
    plan: WeightPlan

    @property
    def backend(self) -> ArithmeticBackend:
        return self.plan.backend

    def zero(self):
        return self.from_int(0)

    def one(self):
        return self.from_int(1)

    def neg_one(self):
        return self.from_int(-1)

    def from_int(self, value: int):
        ...

    def from_fraction(self, numerator: int, denominator: int = 1):
        ...

    def coerce(self, value):
        ...

    def is_zero(self, value) -> bool:
        return value == self.zero()

    def sum(self, values):
        total = self.zero()
        for value in values:
            total += self.coerce(value)
        return total
```

Initial backend behavior:

```python
FMPZ:
    from_int -> fmpz
    from_fraction with denominator != 1 -> ValueError

FMPQ:
    from_int -> fmpq(value)
    from_fraction -> fmpq(numerator, denominator)

FLOAT:
    from_int -> float(value)
    from_fraction -> numerator / denominator

ARB:
    from_int -> arb(value)
    from_fraction -> arb(numerator) / arb(denominator)

FMPZ_POLY:
    constants -> fmpz_poly([value])

FMPQ_POLY:
    constants -> fmpq_poly([fmpq(...)])

ARB_POLY:
    constants -> arb_poly([arb(...)])

FMPZ_MPOLY / FMPQ_MPOLY:
    constants -> matching mpoly context constant
```

For unsupported or not-yet-wired backends, fail early:

```python
raise UnsupportedArithmeticBackend(...)
```

Do not silently fall back to `fmpq`.

## Build Arithmetic From Problem

Add to `src/wfomc/weights.py`:

```python
def build_weight_plan(
    problem: Problem,
    options: WeightOptions,
) -> WeightPlan:
    symbols = collect_symbolic_weight_variables(problem)
    backend = choose_weight_backend(
        options,
        symbolic_variables=symbols,
        raw_weights=problem.weights,
    )
    return WeightPlan(options=options, symbolic_variables=symbols, backend=backend)
```

Add:

```python
def build_arithmetic_context(problem: Problem, options: WeightOptions) -> ArithmeticContext:
    return ArithmeticContext(build_weight_plan(problem, options))
```

Avoid circular imports by placing `ArithmeticContext` in `wfomc.arithmetic` and importing it only inside functions if necessary.

## MaterializationContext Change

Extend `src/wfomc/engine/orchestration.py`:

```python
@dataclass(frozen=True)
class MaterializationContext:
    runtime: RuntimeContext
    normal_form: C2NormalForm
    features: FeatureSet
    options: AlgoOptions
```

Do not put one global `ArithmeticContext` here if branches may later have branch-specific symbolic variables. Instead, materializers should build it from the branch problem:

```python
arith = build_arithmetic_context(branch.problem, ctx.options.weight_options)
```

Then attach it to the algorithm input:

```python
algo_input.arithmetic = arith
```

If all materializers need the same helper, add in `src/wfomc/algo/core.py`:

```python
def arithmetic_for_branch(problem: Problem, options: AlgoOptions) -> ArithmeticContext:
    return build_arithmetic_context(problem, options.weight_options)
```

## AlgoInput Contract

Each concrete algo input dataclass should carry:

```python
arithmetic: ArithmeticContext
```

Files to update:

```text
src/wfomc/algo/standard/input.py
src/wfomc/algo/fast/input.py
src/wfomc/algo/incremental/input.py
src/wfomc/algo/incremental3/input.py
src/wfomc/algo/propositional/input.py
src/wfomc/algo/recursive/input.py
src/wfomc/algo/tail_signature/input.py
```

Every builder in `src/wfomc/algo/cell_graph/inputs.py` should set the field.

Example:

```python
return BasicCellGraphInput(
    ...,
    arithmetic=reduced.arithmetic,
)
```

If `ReducedProblem` remains as materializer-internal bundle temporarily, attach:

```python
arithmetic: ArithmeticContext
```

to it. This is not a new compiled artifact; it is the branch arithmetic policy used while the backend bundle still exists.

## Weight Compilation

Replace:

```python
compiled = compile_weight_mapping(reduced.weight_map(), plan)
```

with:

```python
arith = ArithmeticContext(plan)
compiled = compile_weight_mapping(reduced.weight_map(), arith)
```

Target signature:

```python
def compile_weight_mapping(
    weights: Mapping[object, tuple[object, object]],
    arithmetic: ArithmeticContext,
) -> dict[object, tuple[object, object]]:
    return {
        predicate: (arithmetic.coerce(positive), arithmetic.coerce(negative))
        for predicate, (positive, negative) in weights.items()
    }
```

Backend-specific symbolic parsing should also live behind `ArithmeticContext.coerce(...)`, not as ad hoc code in `compile_weight_mapping`.

## Default Weights

Replace every hard-coded default:

```python
from flint import fmpq
return weights.get(predicate, (fmpq(1, 1), fmpq(1, 1)))
```

with:

```python
one = arithmetic.one()
return weights.get(predicate, (one, one))
```

Files:

```text
src/wfomc/algo/cell_graph/components.py
src/wfomc/algo/cell_graph/inputs.py
src/wfomc/algo/propositional/ganak.py
```

## Algorithm Accumulators

Replace all algorithm-local imports like:

```python
from flint import fmpq as Rational
```

with:

```python
arith = algo_input.arithmetic
result = arith.zero()
one = arith.one()
```

Files:

```text
src/wfomc/algo/standard/solve.py
src/wfomc/algo/fast/solve.py
src/wfomc/algo/incremental/solve.py
src/wfomc/algo/incremental3/solve.py
src/wfomc/algo/incremental3/counting_kernel.py
src/wfomc/algo/recursive/solve.py
src/wfomc/algo/tail_signature/solve.py
src/wfomc/algo/propositional/ganak.py
```

Example:

```python
def solve(algo_input, runtime=None):
    arith = algo_input.arithmetic
    result = arith.zero()
    for component in algo_input.components:
        result += _solve_component(component, arith=arith, ...)
    ...
```

Helper functions should receive `arith` explicitly:

```python
def _solve_component(component, *, arith: ArithmeticContext, ...):
    subtotal = arith.zero()
    weight = arith.one()
```

## Multinomial Coefficients

Current `MultinomialCoefficients.coef(...)` returns exact values independent of backend. Keep the combinatorial cache integer-based, but coerce at use sites:

```python
coefficient = arith.coerce(MultinomialCoefficients.coef(mu))
```

Do not make `MultinomialCoefficients` backend-aware unless profiling shows coercion is too expensive.

## Decode Pipeline

`DecodePipeline` currently carries arithmetic values such as `repeat_factor`.

Change:

```python
repeat_factor=_ring_rational(...)
```

to:

```python
repeat_factor=arith.coerce(...)
```

But `reduction` should remain pure and should not build arithmetic. Therefore:

1. Keep reduction-side repeat/correction metadata as raw exact Python values (`int`, `Fraction`, or typed plan metadata).
2. Convert to backend values in materialization when constructing the final decode callable.

Target:

```python
DecodePipeline(
    repeat_factor=arith.coerce(raw_repeat_factor),
    cardinality_encoding=compile_cardinality_encoding(..., arith),
)
```

This may require moving `DecodePipeline` construction out of `reduction/core.py` and into algo-owned materialization. If that is too much for the first commit, attach raw repeat metadata to `ReducedProblem` and compile it immediately after `compile_reduced_weights(...)`.

## Cardinality Encoding

Cardinality marker weights must use the same arithmetic context:

```python
weighted, encoding = apply_cardinality_weighting(
    compiled,
    reduced.cardinality_constraints,
    arithmetic=arith,
)
```

Update `src/wfomc/cardinality.py` so it does not create `fmpq`/polynomial values directly.

If symbolic variables are introduced by cardinality constraints, `build_weight_plan(...)` must see those variables before choosing backend.

## Propositional / Ganak Boundary

If `ganak` requires exact integer/rational textual weights, reject unsupported rounded backends before running:

```python
if algo_input.arithmetic.backend in {ArithmeticBackend.FLOAT, ArithmeticBackend.ARB, ArithmeticBackend.ARB_POLY}:
    raise PlanningError("propositional ganak backend requires exact weights")
```

Do not silently stringify rounded weights as exact rationals.

## Cache Keys

Engine and cell-graph cache keys must include:

```python
_object_key(options.weight_options)
_object_key(weight_plan.backend)
_object_key(weight_plan.symbolic_variables)
```

If `ArithmeticContext` is attached to `ReducedProblem`, include:

```python
_object_key(reduced.arithmetic.plan)
```

## Task 1: Delete Legacy Arithmetic Surface

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/weights.py`
- Modify: `tests/unit/test_weight_backend_planning.py`

**Step 1: Write failing tests**

Add tests asserting:

```python
assert not hasattr(AlgoOptions(), "arithmetic_backend")
assert "SymbolicWeightPlan" not in dir(wfomc.weights)
assert "choose_arithmetic_backend" not in dir(wfomc.weights)
```

**Step 2: Remove fields and exports**

Delete `AlgoOptions.arithmetic_backend`, `SymbolicWeightPlan`, and `choose_arithmetic_backend`.

**Step 3: Update option resolver**

Only resolve evidence and linear order. Preserve `weight_options`.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_weight_backend_planning.py tests/unit/test_algo_planning.py -q
```

Expected: pass after updating old assertions to `WeightOptions`.

## Task 2: Add ArithmeticContext

**Files:**
- Create: `src/wfomc/arithmetic.py`
- Modify: `src/wfomc/weights.py`
- Test: `tests/unit/test_arithmetic_context.py`

**Step 1: Write tests**

Cases:

```python
def test_fmpq_context_creates_fmpq_zero_one_fraction(): ...
def test_float_context_creates_float_zero_one_fraction(): ...
def test_arb_context_creates_arb_values(): ...
def test_fmpz_context_rejects_non_integer_fraction(): ...
def test_unsupported_arb_mpoly_fails_early(): ...
```

**Step 2: Implement `ArithmeticContext`**

Use backend-specific constructors.

**Step 3: Add `build_weight_plan(...)`**

Build symbols, choose backend, return `WeightPlan`.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_arithmetic_context.py tests/unit/test_weight_backend_planning.py -q
```

Expected: pass.

## Task 3: Compile Weights Through ArithmeticContext

**Files:**
- Modify: `src/wfomc/weights.py`
- Modify: `src/wfomc/algo/core.py`
- Test: `tests/unit/test_weights.py`

**Step 1: Write tests**

Assert compiled values match backend:

```python
exact -> fmpq
round/float -> float
round/arb -> arb
```

**Step 2: Change `compile_weight_mapping` signature**

Use:

```python
compile_weight_mapping(weights, arithmetic)
```

**Step 3: Update `compile_reduced_weights`**

Build `WeightPlan`, `ArithmeticContext`, compile weights, attach arithmetic to reduced/materialized input.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_weights.py tests/unit/test_weight_backend_planning.py -q
```

Expected: pass.

## Task 4: Thread ArithmeticContext Into Algo Inputs

**Files:**
- Modify: all `src/wfomc/algo/*/input.py`
- Modify: `src/wfomc/algo/cell_graph/inputs.py`
- Modify: `src/wfomc/algo/core.py`
- Test: `tests/unit/test_engine_compile.py`

**Step 1: Add tests**

For each major algo:

```python
artifacts = compile_problem(problem, algo=...)
assert artifacts.algo_input.arithmetic.plan.options == requested_weight_options
```

**Step 2: Add field to input dataclasses**

Every algo input gets:

```python
arithmetic: ArithmeticContext
```

**Step 3: Populate from materializers**

Every materializer passes the same arithmetic context used for compiled weights.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_engine_compile.py tests/unit/test_cell_graph_inputs.py -q
```

Expected: pass.

## Task 5: Replace Hard-Coded Constants in Cell Graph Materialization

**Files:**
- Modify: `src/wfomc/algo/cell_graph/components.py`
- Modify: `src/wfomc/algo/cell_graph/inputs.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py` if it creates constants directly
- Test: `tests/unit/test_cell_graph_inputs.py`

**Step 1: Search**

```bash
rg "fmpq|Rational\\(|float\\(|int\\(" src/wfomc/algo/cell_graph src/wfomc/cell_graph
```

**Step 2: Pass `arithmetic` into helper functions**

`_weight_getter(reduced)` becomes:

```python
def _weight_getter(reduced, arithmetic):
    one = arithmetic.one()
    ...
```

**Step 3: Replace zero/one defaults**

No direct `fmpq(1, 1)` for default weights.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_cell_graph_inputs.py tests/unit/test_cell_graph_construction.py -q
```

Expected: pass.

## Task 6: Replace Hard-Coded Constants in Solvers

**Files:**
- Modify: `src/wfomc/algo/standard/solve.py`
- Modify: `src/wfomc/algo/fast/solve.py`
- Modify: `src/wfomc/algo/incremental/solve.py`
- Modify: `src/wfomc/algo/incremental3/solve.py`
- Modify: `src/wfomc/algo/incremental3/counting_kernel.py`
- Modify: `src/wfomc/algo/recursive/solve.py`
- Modify: `src/wfomc/algo/tail_signature/solve.py`

**Step 1: Search**

```bash
rg "from flint import fmpq|Rational\\(|fmpq\\(" src/wfomc/algo
```

**Step 2: Thread `arith` into helper functions**

No helper should import `fmpq` just to create `0` or `1`.

**Step 3: Coerce multinomial coefficients**

Use:

```python
coefficient = arith.coerce(MultinomialCoefficients.coef(mu))
```

**Step 4: Run algorithm suites**

```bash
PYTHONPATH=src:. pytest tests/unit/test_engine_compile.py tests/unit/test_incremental3_counting_native.py tests/unit/test_tail_signature_adapter.py -q
```

Expected: pass.

## Task 7: Compile Decode and Cardinality With ArithmeticContext

**Files:**
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/reduction/decode.py`
- Modify: `src/wfomc/cardinality.py`
- Modify: `src/wfomc/algo/core.py`
- Test: `tests/unit/test_cardinality.py`
- Test: `tests/unit/test_cardinality_execution.py`

**Step 1: Make reduction carry raw exact metadata**

Reduction should not call `fmpq`.

**Step 2: Compile decode in materialization**

Use arithmetic context for repeat factors and cardinality encoding.

**Step 3: Make cardinality weighting backend-aware**

Pass `arithmetic` into cardinality marker construction.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_cardinality.py tests/unit/test_cardinality_execution.py tests/unit/test_reduction.py -q
```

Expected: pass.

## Task 8: Backend Enforcement Tests

**Files:**
- Create: `tests/unit/test_arithmetic_backend_enforcement.py`

**Step 1: Add round/float end-to-end test**

Use a small non-symbolic problem:

```python
result = solve(
    problem,
    options=AlgoOptions(
        weight_options=WeightOptions(precision="round", rounded_backend="float")
    ),
)
assert isinstance(result.value, float)
```

**Step 2: Add exact end-to-end test**

```python
assert isinstance(result.value, fmpq)
```

**Step 3: Add unsupported symbolic float test**

```python
with pytest.raises(UnsupportedArithmeticBackend):
    compile_problem(symbolic_problem, options=round_float)
```

**Step 4: Add source-level guard**

For solver modules, forbid direct constructors:

```python
for path in Path("src/wfomc/algo").rglob("*.py"):
    assert "from flint import fmpq" not in path.read_text()
```

Allow exceptions only in explicit boundary modules:

```text
src/wfomc/arithmetic.py
src/wfomc/weights.py
src/wfomc/algo/propositional/ganak.py  # if exact-only boundary remains
```

**Step 5: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_arithmetic_backend_enforcement.py -q
```

Expected: pass.

## Task 9: Delete Legacy Arithmetic Tests and Framework Residue

**Files:**
- Delete or rewrite: tests that import legacy arithmetic fields
- Delete: `tests/framework/*` if those tests only pin deleted legacy framework behavior
- Delete: production modules only needed by deleted framework tests

**Step 1: Remove old test expectations**

Delete assertions involving:

```python
options.arithmetic_backend
ArithmeticBackend.RATIONAL
ArithmeticBackend.FLINT_POLY
ArithmeticBackend.FLINT_MPOLY
SymbolicWeightPlan
choose_arithmetic_backend
```

**Step 2: Remove old framework import blockers**

Since legacy code is not preserved, delete or rewrite `tests/framework`.

**Step 3: Run unit suite**

```bash
PYTHONPATH=src:. pytest tests/unit -q
```

Expected: pass.

## Acceptance Criteria

The implementation is complete when:

1. `AlgoOptions` has no `arithmetic_backend`.
2. `weights.py` has no `SymbolicWeightPlan` or `choose_arithmetic_backend`.
3. Every algo input carries an `ArithmeticContext`.
4. Every solver accumulator uses `algo_input.arithmetic`.
5. No solver module imports `fmpq` just to create `0` or `1`.
6. Default predicate weights use `arithmetic.one()`.
7. Multinomial coefficients are coerced through `ArithmeticContext`.
8. Cardinality marker weights use the same arithmetic context as predicate weights.
9. Decode repeat factors use the same arithmetic context.
10. `round/float` non-symbolic problems produce float results end to end.
11. Unsupported symbolic rounded backends fail before solving.
12. Exact problems still produce exact FLINT results.
13. Legacy arithmetic tests and framework residues are deleted or rewritten.
14. The suite passes:

```bash
PYTHONPATH=src:. pytest tests/unit -q
```

If full `pytest -q` still collects deleted legacy framework tests, delete those tests or remove their package entry points. Do not preserve legacy code just to keep old framework tests alive.

## Expected Complexity

This is medium-to-large, not a tiny patch.

The hard part is not adding `ArithmeticContext`; that is straightforward. The hard part is finding every implicit numeric constructor in algorithms and making the arithmetic policy explicit without changing counts.

Recommended execution order:

1. Add `ArithmeticContext` and tests.
2. Compile weights through it.
3. Thread it into algo inputs.
4. Replace constants in one algorithm at a time.
5. Move decode/cardinality last, because those touch result semantics.

Do not attempt all algorithms in one commit.
