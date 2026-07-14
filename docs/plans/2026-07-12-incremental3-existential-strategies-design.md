# Incremental3 existential strategies

## Decision

Existential sections in `C2NormalForm` must be consumed explicitly before an
incremental3 input is materialized. `AlgoOptions.existential_strategy` selects
one of two exact reductions:

- `counting` (incremental3 default): lower `forall X: exists Y: phi(X,Y)` to a
  row-count section `count Y phi(X,Y) >= 1`, and lower a global existential to
  a global `count >= 1` section. A fresh definition predicate makes a composite
  body atomic without changing its meaning.
- `skolem`: use the weighted Skolem reduction shared by the other lifted
  algorithms. Every existential section receives a fresh predicate with
  weights `(1, -1)`.

Other algorithms support only `skolem` and reject an explicit `counting`
request. This prevents a shared option from being silently ignored.

## Data flow

The incremental3 reduction order is:

1. reduce unary evidence;
2. reduce existential sections according to the resolved option;
3. prepare remaining counting sections for native counting DP;
4. reduce global cardinality constraints.

The native `>= 1` row state reuses the existing saturating existential bit in
`CountingState`. Global `>= 1` uses a unary lower-bound mask. The old behavior
that scanned formulas for binary predicates is no longer on the incremental3
production path.

## Verification

Tests cover atomic and composite row existentials, global existentials,
multiple existential sections, fresh-name collisions, CLI selection, exact
agreement between both strategies, and rejection by unsupported algorithms.
The solver-matrix benchmark contains paired counting/Skolem cases and can be
run under an external hard timeout.
