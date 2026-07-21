# ADR-0025: Native Boundary-Profile Dynamic Program

## Status

Accepted

## Context

The cell graph already materializes the WFOMC master sum as local cell
contributions and symmetric pair interactions. FastV2 evaluates that sum with
clique and independent-set formulas, while the experimental tail-signature
package delegates evaluation to an external engine. The Boundary-Profile
framework generalizes tail signatures from a linear order to a binary
decomposition of the cell set and needs framework-owned arithmetic,
cardinality-marker truncation, plan search, and resource diagnostics.

The paper defines messages containing `W_i(n) / n!`. A literal implementation
would introduce rational denominators even when all input weights are integral.
FLINT polynomial and Arb interaction values are also unhashable, so they cannot
serve directly as profile or planner-cache keys.

## Decision

Implement Boundary-Profile as a separate native algorithm package and expose it
as beta through both the CLI and Python API after correctness, resource-bound
benchmark, and package-build verification.
Extract the master-sum component materializer into `wfomc.algo.master_sum` and
reuse it from both Boundary-Profile and the external tail-signature adapter.

Store factorial-scaled messages `G_t(c) = |c|! F_t(c)`. Binary joins therefore
use an integer binomial coefficient and require only addition, multiplication,
and powers through `ArithmeticContext`.

Search several deterministic decomposition candidates: a tail-signature
caterpillar, greedy agglomeration, component/symmetric-clique structure, exact
subset splits for small cell sets, and independent-set root closures. Select by
a domain-size-aware state/join-work estimate. Intern interaction values into
integer labels before constructing boundary profiles.

Register Boundary-Profile as `AlgoName.BOUNDARY_PROFILE` and the CLI value
`boundary-profile`. Beta maturity communicates that its decomposition planner
and performance envelope may still evolve without hiding a verified native
implementation from users.

## Consequences

### Positive

- Cardinality markers, CCS evidence, truncation, and numeric backend selection
  remain owned by existing framework layers.
- Factorial scaling avoids unnecessary rational growth and is valid over the
  arithmetic operations already exposed by the framework.
- Independent cells, symmetric repeated blocks, disconnected components, and
  tail-signature orderings are candidate decompositions rather than unrelated
  solver-specific shortcuts.
- The implementation can be benchmarked independently against FastV2,
  Incremental3, and the external tail-signature engine.

### Negative

- Decomposition planning adds preprocessing work and uses heuristic estimates
  beyond the exact small-cell search threshold.
- Wide boundary messages can still require polynomial space in the domain size
  with a large exponent and may hit external resource limits.
- The native kernel and specialized FastV2 formulas remain separate code paths.

### Neutral

- False-negative equality during approximate-value interning only prevents a
  profile merge; values are never merged without a positive equality check.
- Articulation separators require a separately justified separator state and
  are not part of the base disjoint-child recurrence.

## Alternatives Considered

**Replace FastV2.** Rejected because FastV2 has mature closed forms and evidence
handling whose costs differ substantially from generic BP joins.

**Add Boundary-Profile modes to the external tail-signature adapter.** Rejected
because core correctness, arithmetic, planning, and memory behavior would remain
outside the framework.

**Embed the DP in Incremental3.** Rejected because Incremental3 counting states
are transition-mutated element states rather than fixed master-sum interaction
profiles.

## References

- `docs/plans/2026-07-16-boundary-profile-dp-design.md`
- `reorder_wfomc/draft.tex`, “A Boundary-Profile Framework for WFOMC”
- `new_WFOMC_with_notes/src/wfomc/tail_signature_opt27.py`
