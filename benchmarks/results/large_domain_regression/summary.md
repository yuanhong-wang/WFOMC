# Large-domain refactor regression benchmark

- Current: `85abc3303b016de7082cda0800e1b811c0416353+worktree.75e3e73dc49e`
- Baseline: clean pre-refactor `4addfae10b9827f66649b0494a30b3cccc5f738e`
- Limits: 30 seconds and 4 GiB process-tree RSS per measurement
- Matrix: 6 workloads × `fastv2`/`incremental3` × current/pre-refactor = 24 rows
- Domain sizes: 40, 50, 64, 80, 200, and 300

## Verdict

No large-domain correctness, status, resource-bound, or repeatable performance
regression was found.

- All 12 current/pre-refactor pairs completed successfully.
- All 12 pairs returned exactly the same result.
- No case hit the 30-second or 4-GiB limit.
- The single-run current/pre-refactor ratios had a `1.041x` geometric mean and
  `1.055x` median.
- The only large single-run ratio (`2.438x`) was a sub-second cold-start
  outlier. Three alternating fresh-process pairs reduced its median to
  `0.957x`.
- The two ten-second/high-memory cases that initially appeared 8–9% slower
  were slightly faster in alternating A/B confirmation.

## Cold-run results

| workload | domain | algorithm | pre-refactor | current | ratio |
|---|---:|---|---:|---:|---:|
| `c2/3-regular/n50` | 50 | fastv2 | 15.203s | 12.665s | 0.833x |
| `c2/3-regular/n50` | 50 | incremental3 | 0.199s | 0.485s | 2.438x |
| `c2/3-regular-hand/n40` | 40 | fastv2 | 4.975s | 5.343s | 1.074x |
| `c2/3-regular-hand/n40` | 40 | incremental3 | 12.914s | 14.097s | 1.092x |
| `core/derangements/n80` | 80 | fastv2 | 0.124s | 0.077s | 0.625x |
| `core/derangements/n80` | 80 | incremental3 | 18.962s | 17.035s | 0.898x |
| `unary-structure/total-mappings-S/exact/n200` | 200 | fastv2 | 1.018s | 1.082s | 1.063x |
| `unary-structure/total-mappings-S/exact/n200` | 200 | incremental3 | 14.516s | 15.723s | 1.083x |
| `models/unary_evidence/employment.mln/n64` | 64 | fastv2 | 0.228s | 0.195s | 0.858x |
| `models/unary_evidence/employment.mln/n64` | 64 | incremental3 | 7.279s | 7.616s | 1.046x |
| `core/2-coloured/n300` | 300 | fastv2 | 0.113s | 0.143s | 1.267x |
| `core/2-coloured/n300` | 300 | incremental3 | 0.331s | 0.307s | 0.926x |

The n=300 case is intentionally lightweight: it checks large-domain arithmetic
and result construction, while the n=40–200 cases exercise the expensive DP
and cell-graph paths.

## Alternating fresh-process confirmation

Each row used three current/pre-refactor pairs with alternating execution
order.

| workload / algorithm | paired ratios | paired median |
|---|---|---:|
| `c2/3-regular/n50` / incremental3 | 1.004x, 0.957x, 0.826x | 0.957x |
| `c2/3-regular-hand/n40` / incremental3 | 0.991x, 0.956x, 0.982x | 0.982x |
| `unary-structure/total-mappings-S/exact/n200` / incremental3 | 0.967x, 0.961x, 1.018x | 0.967x |

## Memory

The largest current RSS was 1479.4 MiB for
`total-mappings-S/exact/n200`, compared with 1496.3 MiB pre-refactor. Other
high-memory incremental3 comparisons were:

| workload | pre-refactor | current |
|---|---:|---:|
| `c2/3-regular-hand/n40` | 718.1 MiB | 710.3 MiB |
| `core/derangements/n80` | 874.9 MiB | 852.8 MiB |
| `models/unary_evidence/employment.mln/n64` | 711.1 MiB | 734.7 MiB |

No memory-limit transition or material memory growth was observed.

## Files

- `results.csv`: 24 raw bounded measurements
- `comparison.csv`: 12 paired current/pre-refactor rows
- `confirmation.csv`: 18 alternating confirmation measurements
