# Cardinality-Complete Core Benchmark Cases

## Goal

Replace the relation and reduction kernels in the core benchmark suites with
the complete finite-model problems named by the historical benchmark labels.
Keep every existing domain-size grid and do not retain compatibility aliases.

## Problem families

- `permutations`: one outgoing and one incoming `P` edge per element.
- `derangements`: permutations with no fixed point.
- `endofunctions`: one outgoing `F` edge per element.
- `undirected-k-regular`: simple undirected graphs of degree exactly `k`.
- `k-edge-disjoint-perfect-matchings`: `k` labelled, pairwise edge-disjoint
  perfect matchings.
- `loopless-digraph-without-isolates`: use the intended existential sentence,
  correcting the polarity error in the literal historical Skolem matrix.

Properly coloured graph cases are already complete and remain unchanged.

## Encoding

Keep the historical weighted-Skolem FO2 matrices and attach the concrete
binary cardinality constraints required by each domain.  This directly tests
the FO2-cardinality path and preserves the original benchmark encodings.  The
right-hand sides scale with the selected domain, so these cases are distinct
source `Problem` objects; cross-domain template reuse is outside this change.

## Catalog contract

Cardinality-complete families use the `fo2-cardinality-reduction` variant in
their keys.
The previous kernel keys disappear.  Unary-cardinality stress-test families
remain separate, but the no-isolates Skolem formula is corrected there as well.
No historical CSV artifact is rewritten.

## Verification

Tests cover catalog inventory, removal of old keys, corrected no-isolates
polarity, and exact small-domain counts.  The focused catalog/runner tests and
a three-algorithm smoke comparison must pass.
