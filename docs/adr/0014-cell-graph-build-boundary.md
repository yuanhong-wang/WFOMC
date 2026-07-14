# ADR-0014: Cell-Graph Build Boundary

## Status

Accepted; package placement amended by ADR-0020.

## Context

Cell-graph preparation had three representations: a mutable `CellGraph`, an
immutable-looking `CellGraphData` snapshot, and fast-specific live subclasses.
A module-global cache returned either data snapshots or fast live objects based
on an `optimized` flag. Equivalent branches with different
`ArithmeticContext` instances could therefore reuse the first branch's graph
and `TwoTable` context.

The package also retained a 255-line formula wrapper module, a one-function
utility module, test-only data facade methods, and a second nullary-weight
representation even though nullary assignments were already represented by
the branch `graph_weight`.

## Decision

- `cell_graph/build.py` contains a private, single-use `_CellGraphBuilder` and
  exposes only `build_cell_graphs`.
- `cell_graph/data.py` owns `Cell`, `PairFactor`, plain `CellGraphData`, and
  the algorithm-facing `CellGraphComponent` projection.
- Every build uses the supplied branch `ArithmeticContext`; there is no global
  cell-graph cache. Prepared algorithm inputs remain cached by `RuntimeContext`.
- `CellGraphData` is the only shared output. Algorithms derive their own
  `CellGraphComponent` variants.
- Fast algorithms compose over `CellGraphData` for temporary clique analysis;
  they do not inherit from the base builder.
- FOL operations are called from `wfomc.fol` directly. Cell-graph-specific
  literal-universe and selector helpers remain private in the builder.
- Nullary assignments contribute only to branch `graph_weight`.
- Every branch retains the original non-nullary predicate universe; branch
  simplification must not remove free predicate interpretations.

## Consequences

- Branch arithmetic contexts cannot leak through process-global graph state.
- The ordinary and fast paths share one grounding and nullary-branch builder.
- The cell-graph package does not import engine, problem, or algorithm modules;
  its explicit `evidence.py` boundary consumes reduced evidence contracts.
- Cross-algorithm graph reuse is intentionally removed. If measurements later
  justify it, reuse must be added to the instance-scoped `RuntimeCache`.
- `CellGraphComponent` keeps its established name while its module is made
  explicit (`cell_graph/data.py`, as amended by ADR-0020).
