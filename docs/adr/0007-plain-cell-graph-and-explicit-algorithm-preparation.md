# ADR-0007: Plain Cell Graph Data and Explicit Algorithm Preparation

## Status

Accepted

## Context

The framework previously built active `CellGraph` subclasses, converted them
through a shared algorithm adapter, and then copied their fields into concrete
algorithm inputs. The engine also assembled algorithms from separate reduction,
materialization, and run callables. Following one solve required jumping across
many modules, while shared cell-graph code imported concrete algorithm types.

## Decision

The shared cell-graph boundary is `CellGraphData`: cells, cell weights, and
conditional two-tables. Algorithms explicitly derive their own ordered,
counting, fast, tail-signature, or CNF inputs.

`AlgoSpec` exposes option resolution, `prepare`, and `solve`. Each algorithm's
spec lists its reduction sequence and input construction in execution order.
The engine caches prepared branches and only coordinates analyze, prepare,
solve, decode, and branch aggregation.

## Consequences

### Positive

- Standard, recursive, incremental, incremental3, tail-signature, and
  propositional solvers no longer depend on a live cell graph.
- Shared code no longer imports concrete algorithm input types.
- There is one authoritative component collection instead of first-component
  mirror fields.
- Unsupported binary evidence and rounded arithmetic fail before solving.

### Negative

- Algorithm spec files repeat their short reduction/preparation sequence.
- Fast/fastv2 retain specialized clique analysis in their own package.

ADR-0014 later removed their live builder subclasses; the analysis now composes
over the same `CellGraphData` used by the other cell-graph algorithms.

### Neutral

- `Cell` and `TwoTable` remain small domain objects because incremental3 needs
  conditional two-table queries.

## Alternatives Considered

- Split the existing active object into more mixins and interfaces: rejected
  because it adds files without making the execution path direct.
- Give every algorithm an independent cell-graph builder: rejected because
  1-type/2-type grounding is genuinely shared and correctness-sensitive.
- Introduce a problem-type hierarchy: rejected. ADR-0008 later introduced three
  flat stage dataclasses after concrete runtime representation drift remained;
  it does not use inheritance or phase-generic wrappers.

## References

- `docs/architecture-review-2026-07-10.md`
- `docs/plans/2026-07-10-readable-algorithm-cell-graph-simplification.md`
