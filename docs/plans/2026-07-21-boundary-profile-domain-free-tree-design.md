# Domain-Free Boundary-Profile Tree Design

## Goal

Build each Boundary-Profile decomposition tree once per structural cell-graph
variant and reuse it for every concrete domain size. Keep all domain-sized
numeric tables, execution state, and estimates outside the reusable input
template.

## Current Problem

The engine already separates domain-free compilation and algorithm input
templates from concrete-domain execution. Boundary-Profile currently violates
that boundary: `BoundaryProfileInputTemplate.instantiate()` builds local
`W_i(k)` tables and calls the planner, so solving the same compiled problem for
two domain sizes searches for and compiles the BP tree twice.

The existing standalone `wfomc.algo.master_sum` module is also no longer a
shared abstraction. Tail-signature was removed, and only Boundary-Profile
consumes its `W/R` representation. The useful conversion should be owned by the
Boundary-Profile input package.

## Architecture

`BoundaryProfileInputTemplate` owns immutable, domain-free component templates.
Each component template contains base cell weights, the scalar pair-interaction
matrix, the nullary graph weight, and one `BoundaryProfileTree` selected when
the input template is built. The tree contains only topology, cell membership,
boundary partitions, projections, and cross-interaction coordinates. It does
not contain a concrete domain size, local `W_i(k)` tables, domain-limited
arithmetic values, or domain-specific estimates.

Instantiating a template for a domain rebinds the cached base weights into the
concrete arithmetic context, materializes `W_i(0..n)`, binds the tree's
cross-interaction coordinates to the concrete pair matrix, and computes
diagnostic estimates for `n`. This produces a concrete
`BoundaryProfileComponent` and `BoundaryProfilePlan` without searching for or
rebuilding a tree.

The engine may still create more than one input template when its structural
input key changes. For example, CCS evidence can change whether an unmarked
cell profile exists. Each such structural variant receives one tree, and every
domain mapped to that variant reuses it.

## Planning Option

Add a public `BoundaryProfileOptions` dataclass with one field:

```python
tree_reference_domain_size: int | None = None
```

It is exposed as `AlgoOptions.boundary_profile_options` and through the CLI as
`--bp-tree-reference-domain-size N`.

- `None` selects a domain-independent structural cost. Candidate trees are
  compared by peak join width, BP width, non-unit cross interactions, balance,
  and a deterministic tree key.
- A non-negative integer selects the tree using the existing domain-size cost
  model evaluated at that reference size. The chosen tree is still built once
  in the template and reused for every actual domain.

The option is part of the compilation and input-template cache key. Negative
reference sizes fail validation. CLI use with a non-Boundary-Profile algorithm
fails as an algorithm-scoped option error.

## Structural Classification

Independent and symmetric blocks must not be inferred from a finite
`W_i(0..n)` prefix. That makes classification accidentally depend on small
domains. Instead:

- a cell has an exponential local table when its diagonal pair weight is one
  or its base cell weight is zero;
- two cells have identical local tables for every domain when their base cell
  weights and diagonal pair weights are equal;
- interaction signatures and clique checks use the domain-free pair matrix.

This makes node kinds stable across all domain sizes.

## Testing

Tests will first demonstrate the current failure, then verify:

1. one input template and one identical tree object are reused across multiple
   domains of the same structural variant;
2. concrete `W_i(k)` tables and estimates differ with domain size while the
   tree does not;
3. structural and reference-domain planning both return correct counts;
4. distinct reference sizes produce distinct compilation/cache entries;
5. the Python API validates negative sizes and the CLI validates scope;
6. Boundary-Profile remains equal to FastV2 on integration models;
7. no production or test code imports the deleted `master_sum.py` module.
