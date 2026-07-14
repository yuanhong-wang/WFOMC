# ADR-0012: Explicit Root Numeric Module Interfaces

## Status

Accepted

## Context

Arithmetic ownership was split between `arithmetic.py`, `weights.py`, and a
generic `utils/polynomial_flint.py`. The utility module mixed an incomplete
numeric type alias, exact conversion, Ganak context alignment, an identity
`expand` function, and unused random/filter helpers. A custom `Rational`
subclass existed mainly to permit direct mixing of raw fractions with FLINT,
which branch-owned `ArithmeticContext` now forbids.

`WeightPlan` also duplicated fields already required by `ArithmeticContext` and
made `arithmetic.py` depend on `weights.py` for its own backend identity.

## Decision

The root numeric modules have these complete public interfaces:

- `arithmetic.py`: `ArithmeticBackend`, `ArithmeticValue`,
  `choose_arithmetic_backend`, and `ArithmeticContext`;
- `weights.py`: `WeightOptions`, raw/compiled weight mappings, symbol discovery,
  and `compile_weight_mapping`;
- `result.py`: `WFOMCResult`;
- `multinomial.py`: the three shared combinatorial helpers. A one-file `utils`
  package adds no useful boundary, so it is removed.

`ArithmeticContext` directly stores backend, solver symbols, and output symbols;
there is no `WeightPlan`. Raw exact values use `fractions.Fraction`. The
Ganak-only polynomial alignment code lives beside its caller. The obsolete
`polynomial_flint.py` and `rational.py` files are deleted without compatibility
shims.

## Dependency Rule

`weights.py` may depend on `arithmetic.py` to compile values. `arithmetic.py`
must not import `weights.py` at runtime; `WeightOptions` is referenced only for
static typing in backend selection. Algorithms depend on `ArithmeticValue`, not
an exact-only `RingElement` alias.

## Consequences

- Root files expose one concept each and publish explicit `__all__` lists.
- Rounded values are represented honestly by the shared numeric union.
- Raw/reduction values cannot accidentally rely on custom FLINT mixed operators.
- Adding a backend changes the arithmetic owner and its tests, not a utility
  bucket or weight-compilation compatibility layer.
