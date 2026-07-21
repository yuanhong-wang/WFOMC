# Boundary-profile performance comparison

- Boundary-profile commit: `4dd49a212354b6ed8d154a6456cd6ca725da59a8`
- Saved workload inventory commit: `481230d668dd34051161f2ca41fa21f2f008af84`
- Limits: 30 seconds and 4 GiB RSS per measurement
- The four historical configurations were reused from `../cross_branch/results.csv`; they were not rerun.

## Status totals

| configuration | ok | timeout | memory | error | skipped |
|---|---:|---:|---:|---:|---:|
| current/boundary-profile | 327 | 19 | 0 | 59 | 30 |
| devel/fastv2 | 290 | 15 | 0 | 58 | 72 |
| devel/incremental3 | 328 | 30 | 0 | 5 | 72 |
| modk/fastv2 | 292 | 13 | 0 | 58 | 72 |
| modk/incremental3 | 325 | 32 | 1 | 5 | 72 |

## Correctness

Boundary-profile matched the saved successful-result consensus on 296/296 comparable workloads.

## Paired solve-time comparison

Ratios are `historical solve time / boundary-profile solve time`; values above 1 favor boundary-profile.

| historical configuration | pairs | geometric mean | median | boundary-profile faster |
|---|---:|---:|---:|---:|
| devel/fastv2 | 288 | 2.968x | 3.745x | 86.5% |
| devel/incremental3 | 274 | 2.282x | 2.440x | 75.2% |
| modk/fastv2 | 291 | 1.567x | 1.544x | 85.6% |
| modk/incremental3 | 272 | 1.616x | 1.398x | 69.9% |

## Successful-run peak RSS

| configuration | median | maximum |
|---|---:|---:|
| current/boundary-profile | 36.9 MiB | 613.2 MiB |
| devel/fastv2 | 51.8 MiB | 664.7 MiB |
| devel/incremental3 | 45.0 MiB | 2081.3 MiB |
| modk/fastv2 | 158.9 MiB | 984.9 MiB |
| modk/incremental3 | 158.7 MiB | 2722.1 MiB |
