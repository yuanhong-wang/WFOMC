# ADR-0013: Evidence Stage Boundaries

## Status

Accepted

## Context

The evidence package mixed raw user input, algorithm option selection, unary
profile reduction, CCS encoding, cell compatibility, and solver coefficient
generation in one `unary.py` module. It also maintained three equivalent
representations: `EvidencePartition`, `UnaryEvidencePartition`, and
`ProfileCapacityConstraint`.

## Decision

Evidence is represented by three stage-specific contracts:

- `UnaryEvidence` and `BinaryEvidence` are raw public input in
  `evidence/data.py`;
- `ProfileCapacityConstraint` is the reduced unary-profile artifact in
  `evidence/profile.py`;
- `CellEvidenceAllocation` is algorithm materialization data in
  `cell_graph/evidence.py`.

Unary profile construction and CCS encoding belong to
`reduction/unary_evidence.py`. `EvidenceStrategy` belongs to the neutral
`options.py` module. Cell graphs and algorithm input builders
consume `ProfileCapacityConstraint` directly; no partition adapter is retained.

Only raw evidence input types are exported from `wfomc.evidence`.

## Consequences

- Each conversion has one owner and one output type.
- The duplicate partition classes and conversion helpers are deleted.
- Cell-allocation counting no longer makes the evidence domain package depend
  on algorithm arithmetic or multinomial helpers.
- CCS changes are localized to reduction; cell compatibility changes are
  localized to cell-graph materialization.
- `ProfileCapacityConstraint` is internal pipeline data rather than top-level
  public API.
