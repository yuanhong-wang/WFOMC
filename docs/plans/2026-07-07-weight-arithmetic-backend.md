# Weight Arithmetic Backend Implementation Plan

> Superseded by `docs/plans/2026-07-07-reduction-engine-algo-consolidated.md`.
> This file is retained only as historical context. Follow the consolidated plan for current decisions.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign `wfomc.weights` so users can choose exact vs rounded arithmetic, while WFOMC automatically selects scalar, polynomial, or multivariate polynomial backends from the number of symbolic variables.

**Architecture:** Treat weight handling as arithmetic planning and compilation, not reduction. The weight layer should inspect raw problem weights and symbolic variables introduced by symbolic weights/cardinality constraints, choose a backend from precision mode plus symbolic arity, and compile weights into the matching Python/FLINT objects. Unsupported combinations, especially rounded multivariate symbolic arithmetic, should fail early with clear errors.

**Tech Stack:** Python 3.11, python-flint (`fmpz`, `fmpq`, `arb`, `fmpz_poly`, `fmpq_poly`, `arb_poly`, `fmpz_mpoly`, `fmpq_mpoly`), existing `wfomc.weights`, existing `wfomc.cardinality`.

---

## Design Decision

Weights have two independent dimensions:

```text
precision mode:
  exact | round

symbolic dimension:
  none | one variable | multiple variables
```

The user chooses precision:

```text
exact:
  use exact integer/rational arithmetic

round:
  use approximate arithmetic, either float or arb
```

WFOMC chooses symbolic backend automatically from the number of variables:

```text
0 symbolic variables -> scalar arithmetic
1 symbolic variable  -> polynomial arithmetic
2+ symbolic variables -> multivariate polynomial arithmetic
```

## Backend Matrix

| Precision | 0 symbolic vars | 1 symbolic var | 2+ symbolic vars |
|---|---|---|---|
| `exact` | `fmpz` or `fmpq` | `fmpz_poly` or `fmpq_poly` | `fmpz_mpoly` or `fmpq_mpoly` |
| `round` + `float` | Python `float` | unsupported initially | unsupported |
| `round` + `arb` | `arb` | `arb_poly` | unsupported because python-flint has no `arb_mpoly` |

Integer-preserving variants are preferred when possible:

- all scalar exact weights integer -> `fmpz`;
- any scalar exact rational weight -> `fmpq`;
- univariate exact with integer coefficients only -> `fmpz_poly`;
- univariate exact with rational coefficients -> `fmpq_poly`;
- multivariate exact with integer coefficients only -> `fmpz_mpoly`;
- multivariate exact with rational coefficients -> `fmpq_mpoly`.

If this distinction is too much for the first implementation, default exact scalar/poly/mpoly to rational-capable backends (`fmpq`, `fmpq_poly`, `fmpq_mpoly`) and optimize to integer-capable backends later.

## Public Options

Add:

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WeightOptions:
    precision: Literal["exact", "round"] = "exact"
    rounded_backend: Literal["float", "arb"] = "arb"
```

Default should be exact.

Rationale:

- WFOMC usually wants exact model counts by default.
- Rounded arithmetic should be explicit.
- `arb` is the preferred rounded backend when available because it carries ball arithmetic semantics.

## Backend Enum

Replace or extend the current `ArithmeticBackend`:

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
    ARB_MPOLY = "arb_mpoly"
```

Keep `ARB_MPOLY` in the enum only as a named unsupported target:

```python
raise UnsupportedWeightBackend(
    "rounded multivariate symbolic weights require arb_mpoly, "
    "which is not supported by python-flint"
)
```

## Plan Object

Add:

```python
@dataclass(frozen=True)
class WeightPlan:
    options: WeightOptions
    symbolic_variables: tuple[str, ...]
    backend: ArithmeticBackend
```

This object should be built by engine/materialization, not by pure reduction.

## Backend Selection

Add:

```python
def choose_weight_backend(
    options: WeightOptions,
    *,
    symbolic_variables: tuple[str, ...],
    raw_weights: object,
) -> ArithmeticBackend:
    ...
```

Initial behavior:

```python
def choose_weight_backend(options, *, symbolic_variables, raw_weights):
    n_symbols = len(symbolic_variables)

    if options.precision == "exact":
        if n_symbols == 0:
            return ArithmeticBackend.FMPQ
        if n_symbols == 1:
            return ArithmeticBackend.FMPQ_POLY
        return ArithmeticBackend.FMPQ_MPOLY

    if options.precision == "round":
        if n_symbols == 0:
            if options.rounded_backend == "float":
                return ArithmeticBackend.FLOAT
            return ArithmeticBackend.ARB
        if n_symbols == 1:
            if options.rounded_backend == "float":
                raise UnsupportedWeightBackend(
                    "rounded univariate symbolic weights require arb_poly; "
                    "float polynomial weights are not supported"
                )
            return ArithmeticBackend.ARB_POLY
        raise UnsupportedWeightBackend(
            "rounded multivariate symbolic weights require arb_mpoly, "
            "which is not supported by python-flint"
        )

    raise ValueError(f"Unsupported weight precision: {options.precision}")
```

Later optimization:

- inspect `raw_weights` and cardinality symbolic coefficients;
- choose `FMPZ` instead of `FMPQ` when all exact values are integral;
- choose `FMPZ_POLY` / `FMPZ_MPOLY` when all coefficients are integral.

## Symbolic Variable Discovery

WFOMC should discover symbolic variables automatically.

Sources:

1. symbolic weights in the problem;
2. symbolic variables introduced by cardinality constraints or cardinality encodings;
3. future symbolic parameters introduced by reductions.

Add helper:

```python
def collect_symbolic_weight_variables(problem: object, cardinality_constraints: object | None = None) -> tuple[str, ...]:
    ...
```

Ordering must be stable:

```python
return tuple(sorted(variables))
```

If callers need custom variable order later, add an override to `WeightOptions`.

## Weight Compilation

Rename the current vague conversion function:

Current:

```python
convert_weight_mapping_to_ring_elements(weights)
```

Target:

```python
def compile_weight_mapping(
    weights: Mapping[object, tuple[object, object]],
    plan: WeightPlan,
) -> dict[object, tuple[object, object]]:
    ...
```

Behavior:

- `FMPQ`: convert exact scalar values to `fmpq`;
- `FMPZ`: convert exact integer scalar values to `fmpz`;
- `FLOAT`: convert scalar values to `float`;
- `ARB`: convert scalar values to `arb`;
- `FMPQ_POLY`: parse symbolic expressions as univariate rational polynomials;
- `FMPZ_POLY`: parse symbolic expressions as univariate integer polynomials;
- `ARB_POLY`: parse symbolic expressions as univariate arb polynomials;
- `FMPQ_MPOLY`: parse symbolic expressions as multivariate rational polynomials;
- `FMPZ_MPOLY`: parse symbolic expressions as multivariate integer polynomials;
- `ARB_MPOLY`: raise unsupported.

Initial implementation may preserve the current `to_ringelements()` path for exact scalar/polynomial behavior, but it should be wrapped behind the explicit `WeightPlan`.

## Engine Boundary

Pure `reduction` must not call weight compilation.

Allowed:

```text
engine/materialization:
  WeightOptions + problem/cardinality metadata
    -> WeightPlan
    -> compiled solver weights
```

Not allowed:

```text
reduction:
  convert raw weights to flint/ring values
```

So this line should eventually leave `reduction/core.py`:

```python
ring_weights = convert_weight_mapping_to_ring_elements(state.weights)
```

and become engine/materialization logic:

```python
symbolic_variables = collect_symbolic_weight_variables(problem, cardinality_constraints)
weight_plan = build_weight_plan(weight_options, symbolic_variables, problem.weights)
solver_weights = compile_weight_mapping(problem.weights, weight_plan)
```

## User-Facing Examples

Exact scalar:

```python
options = WeightOptions(precision="exact")
# backend: fmpq by default, fmpz if integer-only optimization is enabled
```

Rounded scalar:

```python
options = WeightOptions(precision="round", rounded_backend="arb")
# backend: arb
```

Exact univariate symbolic:

```python
weights = {P: ("x + 1", 1)}
options = WeightOptions(precision="exact")
# symbolic_variables = ("x",)
# backend: fmpq_poly
```

Exact multivariate symbolic:

```python
weights = {P: ("x + y", 1)}
options = WeightOptions(precision="exact")
# symbolic_variables = ("x", "y")
# backend: fmpq_mpoly
```

Rounded univariate symbolic:

```python
weights = {P: ("x + 1", 1)}
options = WeightOptions(precision="round", rounded_backend="arb")
# backend: arb_poly
```

Rounded multivariate symbolic:

```python
weights = {P: ("x + y", 1)}
options = WeightOptions(precision="round", rounded_backend="arb")
# raises UnsupportedWeightBackend until python-flint supports arb_mpoly
```

## Tests

Create or update:

```text
tests/unit/test_weight_backend_planning.py
```

Test cases:

```python
def test_exact_scalar_uses_fmpq_by_default():
    plan = build_weight_plan(WeightOptions(precision="exact"), (), {P: (1, 2)})
    assert plan.backend is ArithmeticBackend.FMPQ


def test_round_scalar_arb_uses_arb():
    plan = build_weight_plan(
        WeightOptions(precision="round", rounded_backend="arb"),
        (),
        {P: (1, 2)},
    )
    assert plan.backend is ArithmeticBackend.ARB


def test_exact_univariate_symbolic_uses_fmpq_poly():
    plan = build_weight_plan(WeightOptions(precision="exact"), ("x",), {P: ("x", 1)})
    assert plan.backend is ArithmeticBackend.FMPQ_POLY


def test_exact_multivariate_symbolic_uses_fmpq_mpoly():
    plan = build_weight_plan(
        WeightOptions(precision="exact"),
        ("x", "y"),
        {P: ("x + y", 1)},
    )
    assert plan.backend is ArithmeticBackend.FMPQ_MPOLY


def test_round_univariate_symbolic_uses_arb_poly():
    plan = build_weight_plan(
        WeightOptions(precision="round", rounded_backend="arb"),
        ("x",),
        {P: ("x", 1)},
    )
    assert plan.backend is ArithmeticBackend.ARB_POLY


def test_round_multivariate_symbolic_is_unsupported():
    with pytest.raises(UnsupportedWeightBackend):
        build_weight_plan(
            WeightOptions(precision="round", rounded_backend="arb"),
            ("x", "y"),
            {P: ("x + y", 1)},
        )
```

Add compile tests later once expression parsing and polynomial construction are implemented.

## Migration Steps

### Task 1: Introduce Options and Backend Selection

Modify `src/wfomc/weights.py`:

- add `WeightOptions`;
- replace/extend `ArithmeticBackend`;
- add `WeightPlan`;
- add `UnsupportedWeightBackend`;
- add `choose_weight_backend`;
- add `build_weight_plan`.

Keep `convert_weight_mapping_to_ring_elements()` temporarily for existing callers.

### Task 2: Add Backend Planning Tests

Add tests listed above.

Run:

```bash
uv run pytest tests/unit/test_weight_backend_planning.py -q
```

Expected: PASS.

### Task 3: Move Engine Call Sites to Explicit Plan

Where materialization currently calls:

```python
convert_weight_mapping_to_ring_elements(weights)
```

replace with:

```python
plan = build_weight_plan(options.weight_options, symbolic_variables, weights)
compiled_weights = compile_weight_mapping(weights, plan)
```

Do not do this inside pure `reduction`.

### Task 4: Implement Scalar Compilation

Implement `compile_weight_mapping()` for:

- `FMPZ`;
- `FMPQ`;
- `FLOAT`;
- `ARB`.

### Task 5: Implement Symbolic Exact Compilation

Implement:

- `FMPQ_POLY`;
- `FMPQ_MPOLY`.

Add integer-specialized backends only after rational-capable versions are stable.

### Task 6: Implement Rounded Univariate Symbolic Compilation

Implement:

- `ARB_POLY`.

Keep `ARB_MPOLY` unsupported.

## Acceptance Criteria

- Users can choose exact vs rounded arithmetic.
- WFOMC automatically chooses scalar/poly/mpoly from symbolic variable count.
- Rounded multivariate symbolic weights fail with a clear unsupported-backend error.
- `weights.py` owns arithmetic backend planning and weight compilation.
- Pure `reduction` no longer converts weights to ring elements.
- Existing exact scalar behavior remains compatible.
