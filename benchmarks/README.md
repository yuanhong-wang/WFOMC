# WFOMC benchmarks

`domain_series_performance.py` is the authoritative runner for comparing the
current implementations of boundary-profile, FastV2, and Incremental3.

## Protocols

- `cold`: each measured solve gets a fresh `RuntimeContext`; compilation and
  input-template construction are included in `solver_time_s`.
- `compile-once`: each repetition compiles one domain-free problem, then solves
  its increasing domains with the same runtime.  The first domain includes
  template/tree construction and later domains measure cache reuse.

The protocols answer different questions and must not be combined into one
speed ratio.

## Commands

Bounded catalog comparison:

```console
uv run python benchmarks/domain_series_performance.py \
  --protocol compile-once \
  --sources catalog \
  --suite core-main \
  --repetitions 3 \
  --timeout 30 \
  --memory-gib 4 \
  --out benchmarks/results/current_algorithms
```

Complete current inventory:

```console
uv run python benchmarks/domain_series_performance.py \
  --protocol compile-once \
  --sources all \
  --suite all \
  --repetitions 3 \
  --timeout 30 \
  --memory-gib 4 \
  --out benchmarks/results/current_algorithms-full
```

Use a distinct output directory for cold results.  Every directory contains a
`manifest.json`; resume is allowed only when the complete run identity matches.
Use `--no-resume` to intentionally replace an existing run.

Statuses distinguish `unsupported`, `invalid`, `timeout`, `memory`, and genuine
solver/worker errors.  Speed aggregates contain only paired successful cases;
the status table reports solved-under-budget coverage separately.

## Catalog key names

Catalog keys describe the formula being measured rather than preserving names
from the historical scripts.  There is no compatibility lookup for old keys.
Representative keys are:

```text
core/bi-total-relation/n100
core/loopless-bi-total-relation/n100
core/3-neighbour-surjection-kernel/n60
core/3-edge-disjoint-edge-covers/n20
c2/undirected-3-regular/direct-c2/n20
c2/undirected-3-regular/fo2-cardinality-reduction/n20
unary/bi-total-relation/sx-cardinality/exact/n100
```

The C2 and reduction cases use one semantic family with different encoding
variants.  Names ending in `kernel` are synthetic weighted reduction kernels,
not standalone claims that the represented graph is regular.  `edge-covers`
means every vertex is incident to an edge in each layer; it does not mean a
matching.

## Historical runners

`cross_branch_performance.py` exists for branches with the legacy API.
`boundary_profile_performance.py` combines a newly measured BP result with saved
historical rows and is retained only for experiment reproduction.  Its timing
ratios are not a current head-to-head comparison.
