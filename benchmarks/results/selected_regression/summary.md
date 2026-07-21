# Selected refactor regression check

- Current: `85abc3303b016de7082cda0800e1b811c0416353+worktree.75e3e73dc49e`
- Saved comparison baseline: `4addfae10b9827f66649b0494a30b3cccc5f738e+dirty.da812fb6cb60`
- Fresh confirmation baseline: clean pre-refactor `4addfae10b9827f66649b0494a30b3cccc5f738e`
- Limits: 30 seconds and 4 GiB process-tree RSS per measurement
- Algorithms: `fastv2` and `incremental3`

## Verdict

No correctness, status, resource-limit, or clear repeatable performance
regression was found in the selected workloads.

- All 14 rows with a saved baseline kept the same status.
- All 12 baseline/current successful pairs returned exactly the same result.
- The existing `row-column-Sx interval n100` incremental3 timeout remained a
  timeout; no new timeout or memory-limit transition appeared.
- The 12 successful cold-run pairs had a current/baseline geometric mean of
  `0.936x` and median of `0.935x`.
- The two largest apparent short-workload slowdowns disappeared under fresh
  multi-process A/B confirmation.
- `facility-location` returned the expected result `120` with incremental3;
  fastv2 continued to reject its non-reducible counting sections as expected.

## Selected cold runs

Times are saved-baseline to current. A single cold run is intentionally noisy;
the apparent slowdowns are checked separately below.

| workload | fastv2 | incremental3 |
|---|---:|---:|
| `c2/3-coloured-3-regular/n12` | 2.755s → 3.263s | 0.174s → 0.237s |
| `c2/3-regular-hand/n30` | 1.030s → 1.135s | 2.985s → 2.804s |
| `unary-structure/row-column-Sx/interval/n100` | 0.805s → 0.441s | timeout → timeout |
| `unary/total-mappings-S/exact/n30` | 0.076s → 0.139s | 0.229s → 0.105s |
| `models/employment.mln/n16` | 0.075s → 0.067s | 0.711s → 0.743s |
| `models/linear_order/head-middle-tail.wfomcs/n32` | unsupported → unsupported | 0.096s → 0.090s |
| `models/unary_evidence/evidence-only.wfomcs/n32` | 0.068s → 0.057s | 0.045s → 0.036s |
| `models/counting_quantifiers/facility-location.wfomcs/n6` | expected reduction error | 0.056s, result 120 |

Peak RSS for successful baseline-comparable rows remained effectively flat:
the largest selected value changed from 212.0 MiB to 213.1 MiB. The known
incremental3 timeout used 1607 MiB before termination, below its earlier
2093 MiB observation and below the 4 GiB limit.

## Fresh-process confirmation

The first three rows compare environment medians. The last row used five
alternating pre-refactor/current pairs because this multi-second case showed
high system-level variance.

| workload / algorithm | pre-refactor | current | current / pre-refactor |
|---|---:|---:|---:|
| `c2/3-coloured-3-regular/n12` / incremental3 | 0.1676s | 0.1676s | 1.000x |
| `unary/total-mappings-S/exact/n30` / fastv2 | 0.0715s | 0.0695s | 0.971x |
| `c2/3-regular-hand/n30` / fastv2 | 1.1204s | 1.1067s | 0.988x |
| `c2/3-coloured-3-regular/n12` / fastv2 | 2.8761s | 3.0106s | 1.074x paired median |

The final case ranged from `0.953x` to `1.467x` across individual pairs; its
paired median was `1.074x`, while the earlier saved current run was faster than
pre-refactor. This does not establish a material code regression, but it is the
one selected case worth retaining as a performance sentinel.

## Files

- `results.csv`: 16 bounded current-worktree measurements
- `comparison.csv`: direct comparison with the saved refactor benchmark
- `confirmation.csv`: fresh-process pre-refactor/current repetitions
