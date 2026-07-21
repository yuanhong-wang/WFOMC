# Boundary-Profile Initial Results — 2026-07-16

## Configuration

- Branch: `devel`
- Algorithm: native `boundary-profile`
- Reference: current `fastv2`
- Per-process limits: 30 seconds wall time and 4 GiB RSS
- Measurement worker: `benchmarks/cross_branch_worker.py`
- Repetitions: 1

These measurements are implementation smoke/performance checks, not a stable
cross-machine performance baseline.

## Catalog cases

| Case | Boundary-Profile | Peak RSS | FastV2 | Peak RSS | Reference result |
| --- | ---: | ---: | ---: | ---: | :---: |
| `core/3-coloured/n100` | 0.031 s | 39.5 MiB | 0.108 s | 52.3 MiB | yes |
| `core/total-mappings/n160` | 0.096 s | 40.3 MiB | 0.124 s | 54.5 MiB | yes |
| `core/4-matchings/n8` | 0.560 s | 63.3 MiB | 14.205 s | 116.3 MiB | yes |
| `c2/3-coloured-3-regular/n14` | 4.094 s | 208.7 MiB | 6.724 s | 61.9 MiB | yes |
| `unary/total-mappings-S/exact/n30` | 0.018 s | 34.8 MiB | 0.067 s | 52.2 MiB | yes |
| `core/4-matchings/n12` | 7.433 s | 370.3 MiB | not rerun | — | not compared |
| `c2/3-coloured-3-regular/n16` | 5.416 s | 222.8 MiB | not rerun | — | not compared |
| `core/3-regular/n100` | 0.598 s | 46.9 MiB | not rerun | — | not compared |
| `core/row-column/n150` | 0.115 s | 40.5 MiB | not rerun | — | not compared |

`core/4-matchings/n16` exceeded the 30-second limit with a sampled peak RSS of
about 2058 MiB. Its selected independent-tail plan has BP width 10, join width
11, about 9.4 million estimated materialized states, and about 160 million
estimated join-state pairs. This is the current practical boundary of the
sparse Python message representation under the configured limit.

## Model-file checks

The following original model files returned exactly the same result under
Boundary-Profile and FastV2:

- `models/4-colored-graph.wfomcs`
- `models/existential.wfomcs`
- `models/permutation-no-fix-sc2.wfomcs`
- `models/nonisolated_graph.wfomcs`
- `models/regular_graphs/3-regular-4-colored-graph.wfomcs`
- `models/regular_graphs/4-regular-2-colored-graph.wfomcs`
- `models/unary_evidence/employment.mln` using the algorithm's CCS preparation

The focused test suite additionally covers direct random master sums, binary
cardinality markers in `R`, default `fmpq_mpoly`, explicit `fmpq_poly`, float,
Arb, nullary branches, empty/unsatisfiable cell sets, disconnected projection,
symmetric balanced composition, and independent-root closing.

## Selected-plan observations

The catalog cases above selected `independent-tail`. This confirms that the
opt27/FastWFOMC independent-cell closing is active: the independent child table
is not materialized, and the root sums effective activities over the hard-tail
boundary state. Synthetic tests separately force and verify the balanced
`symmetric` node and neutral connected-component root joins.

The planner also constructs and compares tail-caterpillar,
greedy-agglomerative, component/symmetric, and (for at most 12 cells) exact
subset-cost candidates. Candidate estimates remain available on the compiled
`BoundaryProfilePlan` for diagnosis.
