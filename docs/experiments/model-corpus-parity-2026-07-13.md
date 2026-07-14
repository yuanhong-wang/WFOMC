# Model corpus parity and runtime comparison (2026-07-13)

## Post-optimization follow-up

ADR 0023 implemented a bounded PySAT pair-factor fast path, removed
predicate-universe tautologies from grounded formulas, and reused factors by
condition signature. Exact regressions remain green (`352 passed, 1 skipped`).

Representative reruns on the same machine:

| Model | Result | Before cell graph | After cell graph | After total solve |
|---|---:|---:|---:|---:|
| books arrangement / incremental3 | 4 | 146.9 ms | 1.8 ms | 4.2 ms |
| predecessor / incremental | 732297646080000 | 575 ms | 1.8 ms | 5.5 ms |
| 3Markov_chain / incremental | exact rational unchanged | 58.6 s | 4.6 ms | 1.86 s |

The books pair phase enumerates 24 local models and takes about 0.7 ms. The
Markov base and three predecessor phases contain only 8 or 16 local models per
phase, so all four avoid SDD compilation. The remaining Markov runtime is the
incremental dynamic program rather than cell-graph construction.

## Scope and method

The current worktree and the read-only repository at
`/Users/lucien/Sync/repos/wfoMC` each contain 81 `.wfomcs` or `.mln` models
under `models/`. The moved `linear_order/predk` and
`linear_order/unary_evidence` directories were mapped to their old locations.
All 81 mapped files have identical SHA-256 content hashes.

Each case ran in an isolated Python process with `PYTHONHASHSEED=0`. The two
repositories used the same algorithm for each model:

- 39 `incremental3` cases for ordinary, regular, modulo-counting, and ordinary
  linear-order models;
- 31 `propositional` cases for MATH models using PREDk or circular predecessor;
- 8 `standard` unary-evidence cases;
- 3 `incremental` PREDk cases.

The main matrix used three independent processes per repository and model,
reported the median, and imposed a 15-second timeout on every process. The one
timeout was rerun once with a 90-second hard limit to obtain its exact result.
Times below measure parsing and solving after importing the package unless
explicitly described as process wall time.

## Exact-result parity

- 81/81 results match exactly.
- 80 cases completed within the 15-second per-process limit in both
  repositories.
- `linear_order/predk/3Markov_chain.mln` exceeded 15 seconds only in the
  current worktree. Its controlled rerun completed in 69.37 seconds. The old
  and current 20,660-character exact rational results have the same SHA-256:
  `8af707485c9235377cb0c2ede7adafcff6c2af141ba83be329883a13fd5a9f00`.
- No result was unstable across the three main-matrix trials.

## Runtime summary

For the 80 cases that completed in the main matrix:

| Metric (sum of per-case medians) | Old | Current | Current / old |
|---|---:|---:|---:|
| Parse | 3.510 s | 2.571 s | 0.733x |
| Solve | 35.180 s | 7.008 s | 0.199x |
| Process wall | 86.585 s | 16.755 s | 0.194x |

After adding the successful long-timeout PREDk rerun, aggregate solve time is
37.049 seconds old versus 76.374 seconds current, or 2.061x. This reversal is
entirely caused by `3Markov_chain`: without it, the current corpus solve total
is about 5.02x faster, mainly because the propositional MATH cases take only
9%–16% of their old solve time.

Across individual models, using a 5% tolerance, the current implementation is
faster on 39 and slower on 42. Most `incremental3` slowdowns are only about
10–30 milliseconds in absolute time even when their ratios are 1.3x–2x.

## Largest current slowdowns

| Model | Algorithm | Old | Current | Ratio |
|---|---|---:|---:|---:|
| `linear_order/predk/3Markov_chain.mln` | incremental | 1.869 s | 69.365 s | 37.11x |
| `linear_order/predk/predecessor.wfomcs` | incremental | 0.040 s | 0.551 s | 13.85x |
| `linear_order/unary_evidence/books-arragement.wfomcs` | incremental3 | 0.036 s | 0.169 s | 4.72x |
| `4-colored-graph.wfomcs` | incremental3 | 0.042 s | 0.135 s | 3.19x |
| `regular_graphs/4-regular-4-colored-graph.wfomcs` | incremental3 | 0.047 s | 0.130 s | 2.77x |
| `modk/m-odd-degree-graph-sc2.wfomcs` | incremental3 | 0.068 s | 0.180 s | 2.65x |

Logging on the long PREDk case locates the dominant cost in cell-graph
compilation: 15.66 seconds for the base pair factors and 42.62 seconds for the
predecessor variants, with total compilation at 58.29 seconds. This is a real
performance regression rather than a solver hang or result mismatch.

## Cell-graph runtime analysis

Runtime logging now reports grounding, each cell-enumeration profile, CNF
shape, base and predecessor pair-factor phases, and SDD compile/evaluate/
materialization times.

- `linear_order/predk/3Markov_chain.mln` has built-in predecessor orders 1, 2,
  and 3. It independently compiles four almost identical SDDs: base 15.94 s,
  PRED1 17.28 s, PRED2 12.38 s, and PRED3 12.95 s. SDD evaluation and output
  materialization remain in the millisecond range. Cell-graph compilation is
  58.60 s and the subsequent incremental DP takes about 11.61 s.
- `linear_order/predk/predecessor.wfomcs` uses built-in PRED1. Its base and
  PRED1 SDD compilations take 0.291 s and 0.260 s, accounting for almost all of
  its 0.575-second run.
- `linear_order/unary_evidence/books-arragement.wfomcs` does **not** use a
  built-in predecessor. `Pred` is case-sensitive and remains an ordinary
  binary predicate, so `predecessor_orders=()`. Its single 0.127-second SDD
  compilation carries two projected row-count relations (`Perm` and the
  reverse-relation marker `@c2_rel_0`) plus two unary-evidence cell profiles.

The first optimization to test is therefore one projected predecessor SDD:
compile the pair formula once with all PREDk orientation bits projected, then
derive the base and each order-specific table by selecting masks. This removes
the current base-plus-k independent compilation pattern without returning to
native enumeration. The books case needs a separate SDD/CNF optimization; it
will not benefit from predecessor-table sharing.
