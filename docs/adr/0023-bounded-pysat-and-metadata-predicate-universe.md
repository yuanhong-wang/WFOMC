# ADR 0023: Bound pair enumeration and keep predicate universes as metadata

## Status

Accepted — 2026-07-13; amended by ADR 0024 — 2026-07-13

## Context

The batch PySDD pair backend removed unbounded pair-model materialization, but
its compilation cost dominated formulas with only a handful of local models.
The books-arrangement model spent about 146 ms building a two-cell pair table;
the three-predecessor Markov model spent about 58.6 seconds compiling four
small but SDD-unfriendly pair formulas.

Cell-graph construction also preserved predicates removed by simplification by
conjoining ground tautologies such as `R(a,b) | ~R(a,b)`. This preserved the
right vocabulary but unnecessarily enlarged Tseitin CNF and SDD structure.

## Decision

- Compute the distinct valid cell-condition signatures before selecting the
  pair backend.
- When there are at most 64 signatures, try PySAT enumeration under those
  assumptions. Block original atom assignments only and stop after 128 models
  across the complete pair build.
- If the budget is exceeded, discard partial factors and enter ADR 0024's
  Ganak/PySDD non-enumerating cascade. PySAT is therefore a bounded fast path,
  not an unbounded production model enumerator.
- Aggregate condition bits and incremental3 projection bits into the same
  sparse exact factor for either backend.
- Preserve the predicate universe as metadata. Expand missing diagonal cell
  bits after projected SAT enumeration and multiply absent off-diagonal atoms
  into the pair factor analytically.
- Materialize one immutable `PairFactor` per valid condition signature and
  reuse it in the existing cell-pair tuple matrix.
- Do not adopt opt27's per-cell-pair CNF copies or its custom recursive WMC
  fallback. Large formulas continue to use one batch factor computation.

## Consequences

Small local theories avoid knowledge-compilation startup while retaining exact
rational and symbolic weights. Large or model-rich theories retain bounded
memory and the batch SDD scaling established by ADR 0019. Algorithm inputs and
the public `CellGraphData` shape do not change.

The model and signature limits are conservative implementation constants, not
semantic limits. Exceeding either changes only the selected backend. ADR 0024
routes this case through exact Ganak polynomial WMC before the existing PySDD
backup. Runtime INFO logs report the selected backend and backend-specific
timings.

Removing formula tautologies makes named constants no longer fail through an
accidental duplicate-predicate check. Cell-graph construction now rejects
named constants explicitly until lifted named-constant semantics are
supported.
