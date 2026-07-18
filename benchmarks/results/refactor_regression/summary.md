# Domain-separated refactor regression benchmark

- Baseline: `4addfae10b9827f66649b0494a30b3cccc5f738e` (clean pre-refactor worktree)
- Current: `4addfae10b9827f66649b0494a30b3cccc5f738e+dirty.da812fb6cb60`
- Matrix: 435 identical saved inputs × `fastv2`/`incremental3` = 870 rows per version
- Limits: 30 seconds and 4 GiB process-tree RSS per measurement
- Truncation: once any configuration in a series hits time/memory, all larger domain sizes in that series are skipped

## Verdict

**No correctness, status, resource-bound, or material heavy-workload regression was found.**

- All 870 baseline/current keys matched.
- 0 baseline-success → current-failure transitions.
- All 618 paired successful results were identical; current `fastv2` and `incremental3` also agreed wherever both succeeded.
- Repeating every apparent >0.1 s slowdown in three fresh processes eliminated all of them; remaining repeatable cold overhead was about 3–5 ms per solve.
- Explicit compile-once probes produced one compilation/template miss followed by cache hits for later domains, with results matching the cold benchmark.

## Status totals

| algorithm | version | ok | timeout | memory | expected error | skipped |
|---|---|---:|---:|---:|---:|---:|
| fastv2 | baseline | 290 | 16 | 0 | 59 | 70 |
| fastv2 | current | 290 | 16 | 0 | 59 | 70 |
| incremental3 | baseline | 328 | 32 | 0 | 5 | 70 |
| incremental3 | current | 328 | 32 | 0 | 5 | 70 |

## Paired cold-solve performance

Ratios are current / pre-refactor solver time. These measurements start a fresh process for every domain size, so they intentionally do not reuse compilation.

| algorithm | paired ok | all geomean | baseline ≥0.1 s | baseline ≥1 s | median RSS before → after | max RSS before → after |
|---|---:|---:|---:|---:|---:|---:|
| fastv2 | 290 | 1.080x | 0.996x | 0.965x | 52.1 → 52.6 MiB | 678.3 → 669.6 MiB |
| incremental3 | 328 | 1.117x | 1.024x | 1.014x | 45.2 → 45.6 MiB | 1590.2 → 1616.5 MiB |

The all-pair ratio is dominated by very short solves: for baseline runs under 0.1 s, the median added time was 6.5 ms (`fastv2`) and 5.4 ms (`incremental3`). On ≥0.1 s and ≥1 s workloads, performance is effectively unchanged.

## Three-process confirmation of apparent slowdowns

| algorithm / case | baseline median | current median | ratio |
|---|---:|---:|---:|
| fastv2 / row-column-Sx interval n100 | 0.794 s | 0.510 s | 0.643x |
| fastv2 / row-column-Sx exact n100 | 0.429 s | 0.442 s | 1.031x |
| fastv2 / function-no-fix-sc2 n4 | 60.5 ms | 64.6 ms | 1.069x |
| fastv2 / permutation n8 | 66.1 ms | 69.2 ms | 1.047x |
| fastv2 / 2-regular-graph-sc2 n16 | 68.5 ms | 72.4 ms | 1.056x |
| fastv2 / partition n2 | 60.0 ms | 64.2 ms | 1.069x |
| incremental3 / BA_CC n32 | 44.4 ms | 49.3 ms | 1.111x |
| incremental3 / total-mappings-S exact n30 | 86.1 ms | 90.9 ms | 1.056x |
| incremental3 / employment n8 | 56.9 ms | 60.0 ms | 1.054x |
| incremental3 / c2 3-regular n40 | 84.6 ms | 89.2 ms | 1.054x |

## Compile-once cache probe

Using `compile_problem(schema)` once and then solving the compiled object for multiple `Domain` values:

- 6-domain model series: 1 `compiled_problems` miss, 1 `algo_input_templates` miss, then 5 template hits.
- 3-domain catalog series: 1 compilation/template miss, then 2 template hits.
- Every successful result matched the independent-process benchmark.
- Per-domain execution/result entries remain domain-specific, as designed.

## Files

- `results.csv`: current 870 raw rows
- `../pre_refactor_regression/results.csv`: clean `4addfae` raw rows
- `comparison.csv`: direct key-by-key A/B table
- `runtime.png`: per-series solve time as domain size grows
- `memory.png`: per-series peak RSS as domain size grows
- `ratio_by_domain.png`: current / baseline solve-time ratio scatter and domain medians

Historical note: comparing against the older `481230d` report shows four status differences, but all four also reproduce on clean `4addfae`; they predate this refactor (the intervening commits include the arithmetic-backend change).
