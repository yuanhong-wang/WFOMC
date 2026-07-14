# ADR-0020: Unified Cell-Graph Package

## Status

Accepted

## Context

Cell-graph construction lived in `wfomc.cell_graph`, while the shared
`CellGraphComponent` contract and unary-evidence materialization lived in
`wfomc.algo.cell_graph`. The latter was not an algorithm, and every cell-graph
algorithm had to import both packages to prepare one input.

## Decision

- `wfomc.cell_graph` owns the complete shared cell-graph subsystem.
- `data.py` owns both construction and algorithm-facing immutable data:
  `Cell`, `PairFactor`, `CellGraphData`, `CellGraphComponent`, and
  `PairWeightMatrix`.
- `evidence.py` owns cell/profile compatibility and materialization.
- `wfomc.algo.cell_graph` is deleted without compatibility aliases.
- Construction modules remain independent of algorithms, the engine, and
  problem orchestration. Only `cell_graph/evidence.py` may import reduced
  evidence contracts.
- `CellGraphData` remains the exact builder output. `CellGraphComponent`
  remains the algorithm-facing scalar/materialized projection; the classes are
  not merged because they represent different stages.

## Consequences

- Algorithms import one cell-graph package instead of two similarly named
  packages.
- `wfomc.algo` contains only actual algorithm implementations and their shared
  core contract.
- The root package has a slightly larger public interface, while the small
  component contract does not require a one-class module.
- This amends ADR-0014 only with respect to package placement and the evidence
  dependency boundary.
