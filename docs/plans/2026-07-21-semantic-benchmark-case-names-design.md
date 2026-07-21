# Semantic Benchmark Case Names Design

## Goal

Replace misleading historical benchmark identifiers with names that describe
the formulas actually measured.  This is a direct migration: old keys are not
accepted as aliases, and historical result artifacts are not rewritten.

## Naming model

Keys use semantic problem names.  When two encodings represent the same
mathematical problem, the problem remains the family and the encoding becomes
the variant.  Thus direct C2 and FO2-cardinality reductions group naturally:

```text
c2/undirected-3-regular/direct-c2/n10
c2/undirected-3-regular/fo2-cardinality-reduction/n10
```

Synthetic weighted kernels say `kernel` explicitly.  Historical `matchings`
becomes `edge-disjoint-edge-covers`, and unary cases name both the base problem
and the constrained auxiliary predicate.

## Migration boundary

Update the canonical catalog, all current tests, comparison-group identifiers,
and current benchmark documentation.  Do not add compatibility lookup, and do
not modify committed historical CSV/experiment reports.  Suite membership,
domain grids, formulas, weights, correction divisors, and algorithms remain
unchanged.

## Verification

Tests assert the complete semantic family inventory, new concrete keys, removal
of old keys, direct/reduced equivalence grouping, and unchanged suite sizes.
Run focused benchmark tests, then the full test suite and benchmark smoke tests.
