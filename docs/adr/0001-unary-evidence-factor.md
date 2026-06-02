# ADR 0001: Shared Unary Evidence Cells

## Status

Accepted.

## Context

Unary evidence was previously handled by separate implementations:

- auxiliary predicates plus `PartitionConstraint`
- direct evidence-consistent configuration enumeration
- a threaded Incremental DP

These paths represented the same fixed evidence-profile capacities, but each
owned its own grouping, compatibility checks, normalization, and algorithm
branches. `PartitionConstraint` also expanded the formula vocabulary and forced
cell and two-table construction over evidence auxiliary predicates.

## Decision

Unary evidence has one semantic representation:

- `EvidenceProfile`: an evidence-induced partial unary assignment and its fixed
  size
- `UnaryEvidencePartition`: an immutable, deterministically ordered partition
  of domain elements
- `CellEvidenceAllocation`: compatibility between one cell graph and those
  evidence profiles

The ownership chain is:

```text
UnaryEvidencePlan
  -> UnaryEvidencePartition
     -> EvidenceProfile
     -> CellEvidenceAllocation (per cell graph)
```

For cell `i`, evidence profile `e`, evidence-profile size `s_e`, and allocation
`x[i,e]`:

```text
x[i,e] = 0 when cell i is incompatible with evidence profile e
sum_i x[i,e] = s_e
n_i = sum_e x[i,e]
```

Algorithms choose an elimination order for the same allocation relation:

| Algorithm | Strategy |
| --- | --- |
| Standard | absolute configuration coefficients |
| Incremental3 | absolute or cell-multinomial-relative configuration coefficients |
| Incremental | threaded remaining evidence-profile capacities |
| Fastv2 | virtual compatible `(base cell, evidence profile)` nodes |
| Fast, Recursive | classic auxiliary-predicate CCS fallback |

The public selector is `UnaryEvidenceStrategy`:

- `AUTO` is the default and chooses the mapping above.
- `CCS` forces the modular classic encoding for any algorithm.

Evidence predicates that do not occur in the sentence are passed to cell graph
construction as required unary predicates. They are assigned in one-element and
two-element grounded models, weighted exactly once in cell weights, and ignored
by two-table weighting.

There is no `UnaryEvidenceEncoding`, internal PC strategy, or
`PartitionConstraint`.

## Constraints

Grouping named evidence into profile sizes relies on domain exchangeability.
Unary evidence combined with named constants in the sentence is rejected until
distinguished constants are supported explicitly.

All evidence coefficients and normalizations use exact `Rational` arithmetic.

## Consequences

- Evidence-profile grouping and literal compatibility are implemented once.
- Incremental3 linear-order handling requests a relative coefficient basis from
  the allocation instead of correcting coefficients locally.
- Fastv2 delegates virtual evidence-node weights and relations to a smaller base
  cell graph.
- Classic CCS remains an independent correctness reference for algorithms that
  have not migrated.
