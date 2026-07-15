# Cross-branch WFOMC performance comparison

- `devel`: `481230d668dd34051161f2ca41fa21f2f008af84`
- `modk`: `2ca2b381941d76f51956daa528717e329aff8b87`
- Limits: 30 seconds and 4 GiB RSS per measurement
- Rows: 1740; excluded model files: 68

## Status totals

| configuration | ok | timeout | memory | error | skipped |
|---|---:|---:|---:|---:|---:|
| devel/fastv2 | 290 | 15 | 0 | 58 | 72 |
| devel/incremental3 | 328 | 30 | 0 | 5 | 72 |
| modk/fastv2 | 292 | 13 | 0 | 58 | 72 |
| modk/incremental3 | 325 | 32 | 1 | 5 | 72 |

## Correctness

All groups with two or more successful configurations returned the same result.

## Aggregate paired branch comparison

Ratios are `modk solve time / devel solve time`; values above 1 favor `devel`.

| algorithm | paired workloads | geometric mean | median | devel faster |
|---|---:|---:|---:|---:|
| fastv2 | 286 | 0.517x | 0.465x | 13.6% |
| incremental3 | 323 | 0.719x | 0.655x | 17.0% |

## Successful-run peak RSS

| configuration | median | maximum |
|---|---:|---:|
| devel/fastv2 | 51.8 MiB | 664.7 MiB |
| devel/incremental3 | 45.0 MiB | 2081.3 MiB |
| modk/fastv2 | 158.9 MiB | 984.9 MiB |
| modk/incremental3 | 158.7 MiB | 2722.1 MiB |

## Largest commonly successful domain per series

| series | n | devel/fastv2 | devel/incremental3 | modk/fastv2 | modk/incremental3 |
|---|---:|---:|---:|---:|---:|
| c2/3-coloured-3-regular | 16 | 9.03 s | 976.6 ms | 4.48 s | 1.62 s |
| c2/3-regular | 50 | 5.08 s | 131.6 ms | 26.51 s | 84.6 ms |
| c2/3-regular-hand | 40 | 1.76 s | 6.64 s | 4.29 s | 13.74 s |
| c2/directed-3-in-3-out | 7 | 21.73 s | 48.5 ms | 4.62 s | 37.0 ms |
| core/2-coloured | 300 | 60.4 ms | 265.5 ms | 28.0 ms | 259.4 ms |
| core/2-matchings | 40 | 60.9 ms | 844.1 ms | 26.8 ms | 836.2 ms |
| core/2-regular | 100 | 71.1 ms | 1.16 s | 49.3 ms | 941.9 ms |
| core/3-coloured | 200 | 98.6 ms | 12.99 s | 83.7 ms | 8.94 s |
| core/3-matchings | 10 | 282.1 ms | 443.1 ms | 116.7 ms | 443.7 ms |
| core/3-regular | 60 | 156.1 ms | 4.71 s | 139.2 ms | 3.91 s |
| core/4-matchings | 8 | 21.85 s | 11.22 s | 12.34 s | 11.31 s |
| core/4-regular | 40 | 504.1 ms | 10.23 s | 447.9 ms | 9.20 s |
| core/derangements | 80 | 64.7 ms | 16.23 s | 25.8 ms | 12.58 s |
| core/no-isolated-digraph | 225 | 62.8 ms | 38.8 ms | 21.2 ms | 24.0 ms |
| core/row-column | 8 | 62.2 ms | 43.4 ms | 20.2 ms | 32.8 ms |
| core/total-mappings | 160 | 128.7 ms | 5.61 s | 84.7 ms | 4.86 s |
| models/2-colored-graph.wfomcs | 64 | 64.3 ms | 46.3 ms | 25.6 ms | 35.2 ms |
| models/3-colored-graph.wfomcs | 64 | 63.6 ms | 305.9 ms | 36.0 ms | 289.7 ms |
| models/4-colored-graph.wfomcs | 64 | 65.4 ms | 5.96 s | 45.8 ms | 5.18 s |
| models/cardinality_constraints_example.wfomcs | 64 | 62.7 ms | 56.2 ms | 18.7 ms | 26.8 ms |
| models/deskmate.mln | 32 | 441.6 ms | 35.7 ms | 1.05 s | 25.7 ms |
| models/employment.mln | 16 | 74.4 ms | 743.7 ms | 25.5 ms | 877.6 ms |
| models/existential.wfomcs | 16 | 60.0 ms | 1.59 s | 27.5 ms | 4.83 s |
| models/friends-smokes.mln | 32 | 68.5 ms | 46.1 ms | 31.6 ms | 32.6 ms |
| models/friends-smokes.wfomcs | 32 | 122.7 ms | 24.55 s | 85.4 ms | 21.98 s |
| models/function-no-fix-sc2.wfomcs | 64 | 74.7 ms | 150.8 ms | 289.2 ms | 134.7 ms |
| models/function-no-fix.wfomcs | 32 | 75.7 ms | 73.6 ms | 36.6 ms | 1.50 s |
| models/molecule.mln | 32 | 181.9 ms | 337.7 ms | 169.9 ms | 356.5 ms |
| models/nonisolated_graph.wfomcs | 64 | 63.0 ms | 1.28 s | 23.4 ms | 1.71 s |
| models/partition.wfomcs | 64 | 59.3 ms | 1.38 s | 36.9 ms | 313.8 ms |
| models/permutation-no-fix-sc2.wfomcs | 64 | 1.26 s | 36.9 ms | 27.19 s | 32.1 ms |
| models/permutation-no-fix.wfomcs | 16 | 72.1 ms | 7.02 s | 34.4 ms | 7.32 s |
| models/permutation.wfomcs | 32 | 433.6 ms | 38.3 ms | 1.18 s | 43.3 ms |
| models/regular_graphs/2-regular-directed-graph.wfomcs | 8 | 656.7 ms | 40.0 ms | 202.8 ms | 27.9 ms |
| models/regular_graphs/2-regular-graph-sc2.wfomcs | 64 | 436.0 ms | 44.9 ms | 2.67 s | 26.4 ms |
| models/regular_graphs/3-regular-2-colored-graph.wfomcs | 32 | 780.4 ms | 235.1 ms | 616.2 ms | 206.3 ms |
| models/regular_graphs/3-regular-3-colored-graph.wfomcs | 16 | 9.51 s | 1.06 s | 4.80 s | 1.72 s |
| models/regular_graphs/3-regular-4-colored-graph.wfomcs | 8 | 3.93 s | 102.5 ms | 1.69 s | 135.4 ms |
| models/regular_graphs/3-regular-directed-graph.wfomcs | 4 | 755.1 ms | 38.7 ms | 556.6 ms | 21.3 ms |
| models/regular_graphs/3-regular-graph-sc2.wfomcs | 32 | 501.3 ms | 57.3 ms | 1.09 s | 38.3 ms |
| models/regular_graphs/4-regular-2-colored-graph.wfomcs | 32 | 10.35 s | 1.74 s | 5.77 s | 2.04 s |
| models/regular_graphs/4-regular-3-colored-graph.wfomcs | 8 | 1.90 s | 49.3 ms | 955.4 ms | 42.8 ms |
| models/regular_graphs/4-regular-4-colored-graph.wfomcs | 4 | 571.5 ms | 43.4 ms | 481.2 ms | 45.7 ms |
| models/regular_graphs/4-regular-graph.wfomcs | 32 | 9.15 s | 308.6 ms | 16.23 s | 242.2 ms |
| models/unary_evidence/2-colored-graph.wfomcs | 64 | 62.4 ms | 43.6 ms | 26.1 ms | 32.8 ms |
| models/unary_evidence/employment.mln | 32 | 76.5 ms | 450.7 ms | 37.1 ms | 5.65 s |
| models/unary_evidence/evidence-only.wfomcs | 64 | 60.9 ms | 45.4 ms | 19.3 ms | 26.6 ms |
| models/unary_evidence/friends-smokes.mln | 16 | 139.2 ms | 466.8 ms | 98.8 ms | 355.2 ms |
| models/unary_evidence/impossible-evidence.wfomcs | 64 | 63.7 ms | 39.7 ms | 19.0 ms | 18.5 ms |
| models/unary_evidence/molecule.mln | 32 | 418.6 ms | 675.6 ms | 272.1 ms | 416.8 ms |
| models/unary_evidence/overlapping-profiles.wfomcs | 64 | 130.5 ms | 361.6 ms | 22.4 ms | 324.8 ms |
| models/unary_evidence/sanity_check.wfomcs | 64 | 91.0 ms | 84.1 ms | 23.2 ms | 30.5 ms |
| unary-cardinality/4-coloured-C1 [exact] | 40 | 81.2 ms | 2.00 s | 45.3 ms | 889.9 ms |
| unary-cardinality/4-coloured-C1 [interval] | 40 | 82.8 ms | 2.02 s | 44.6 ms | 896.1 ms |
| unary-cardinality/4-coloured-C1 [unconstrained] | 40 | 73.4 ms | 1.00 s | 44.8 ms | 944.8 ms |
| unary-cardinality/no-isolated-digraph-S [exact] | 40 | 63.6 ms | 35.6 ms | 21.7 ms | 19.6 ms |
| unary-cardinality/no-isolated-digraph-S [interval] | 40 | 59.5 ms | 70.4 ms | 19.4 ms | 20.0 ms |
| unary-cardinality/no-isolated-digraph-S [unconstrained] | 40 | 62.3 ms | 34.6 ms | 20.4 ms | 19.0 ms |
| unary-cardinality/row-column-Sx [exact] | 40 | 137.0 ms | 21.31 s | 52.1 ms | 9.58 s |
| unary-cardinality/row-column-Sx [interval] | 40 | 111.4 ms | 21.50 s | 33.0 ms | 9.89 s |
| unary-cardinality/row-column-Sx [unconstrained] | 40 | 63.0 ms | 9.36 s | 23.7 ms | 9.15 s |
| unary-cardinality/total-mappings-S [exact] | 200 | 7.13 s | 23.98 s | 228.0 ms | 11.55 s |
| unary-cardinality/total-mappings-S [interval] | 200 | 17.05 s | 24.16 s | 254.4 ms | 11.26 s |
| unary-cardinality/total-mappings-S [unconstrained] | 40 | 65.3 ms | 109.1 ms | 20.9 ms | 80.6 ms |
