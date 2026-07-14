# Pre-refactor test restoration and parity (2026-07-12)

## Scope and timeout policy

The tests were restored from the read-only repository at
`/Users/lucien/Sync/repos/wfoMC`. Model files were verified byte-identical at
the end of the comparison. Every run after the initial inventory was wrapped
in `/opt/homebrew/bin/timeout`: 10–30 seconds per isolated case, 60–120 seconds
per suite.

All ten pre-refactor test modules were migrated against the typed public API.
They now live under responsibility-based `tests/unit/` and
`tests/integration/` directories; the complete source-to-destination mapping is
recorded in `tests/README.md`. In particular:

- broad solver agreement is in `tests/integration/test_algorithm_consistency.py`;
- published answers are in `tests/integration/test_math_answers.py`;
- GANAK/lifted cross-checks are in `tests/integration/test_propositional.py`;
- linear-order and evidence matrices are in `tests/integration/`;
- formula, counting-kernel, evidence-allocation, and profile contracts are in
  focused `tests/unit/` modules.

## Suite results

| Repository/run | Result | Pytest time | Wall time |
|---|---|---:|---:|
| Untouched old repository, full suite | 118 passed, 50 failed, 1 skipped | 49.50 s | 50.99 s |
| Refactored repository, full suite (3-run median) | 450 passed, 1 skipped | 54.29 s | 54.57 s |
| Post-cleanup suite with full unary-evidence matrix | 473 passed, 1 skipped | 73.88 s | < 75 s |
| After unit-test consolidation | 342 passed, 1 skipped | 77.16 s | < 80 s |
| Migrated broad solver/MATH suites | 74 passed | 9.58 s | < 10 s |
| Migrated propositional suite | 34 passed, 1 skipped | 37.92 s | < 40 s |
| Migrated linear/evidence suites | 7 passed | 4.16 s | < 5 s |

All 50 old-repository failures have the same environmental cause: its
recursive path directly calls the unavailable `pynauty.Graph` extension. They
are not failed result assertions. The refactored cell-graph code has a fallback
and therefore does not reproduce those environment failures. Suite times are
not directly comparable because the refactored suite contains substantially
more tests and the restored default matrix avoids known slow legacy lowerings.
The three refactored wall times were 54.57, 54.49, and 55.06 seconds; every
trial had a 90- or 120-second hard timeout.

The post-cleanup suite is intentionally slower because it restores the full
eight-model, six-algorithm unary-evidence matrix under both automatic and CCS
strategies. Its final run used a 120-second hard timeout.

The later unit-test consolidation removed source-layout assertions, legacy
absence checks, repeated solver smoke cases, and duplicate parameter matrices.
It reduced the unit layer from 327 to 196 cases while leaving the integration
matrix unchanged.

## Exact-result parity matrix

Each row used the same model and algorithm in both repositories. Times are
one-shot process wall times and therefore include import/startup overhead.

| Case / algorithm | Result | Old | Refactored |
|---|---:|---:|---:|
| 2-colored / fastv2 | 330626 | 0.59 s | 0.24 s |
| existential / incremental3 | 15515568475732467854453889 | 0.60 s | 0.22 s |
| function-no-fix / incremental3 | 1024 | 0.57 s | 0.20 s |
| 2-regular graph / incremental3 | 465 | 0.57 s | 0.20 s |
| 0mod2 graph / incremental3 | 64 | 0.58 s | 0.20 s |
| head-middle-tail / incremental3 | 360 | 0.58 s | 0.20 s |
| predecessor / incremental | 732297646080000 | 0.61 s | 0.83 s |
| unary evidence / standard | 4 | 0.57 s | 0.17 s |
| books arrangement / incremental3 | 4 | 0.61 s | 0.44 s |
| BA / incremental3 | 24 | 0.62 s | 0.25 s |
| MATH 102 / propositional | 432 | 2.84 s | 0.39 s |
| MATH 33 / propositional | 7680 | 1.99 s | 0.36 s |

All 12 exact results match. Eleven cases are faster in the refactored process;
`predecessor/incremental` is 36% slower in this startup-inclusive measurement.

## Regressions found and fixed

The restored tests exposed five semantic/runtime-boundary bugs:

1. Multiple existential sections shared one weighted Skolem predicate. They
   now receive collision-free predicates.
2. Incremental3 inferred existential constraints by scanning binary
   predicates and could mishandle composite/global existentials. It now has
   explicit counting and Skolem strategies.
3. Exact row-count lowering reused `__sk`, `__aux`, and `__C` predicates within
   and across sections. All generated predicates are now fresh.
4. Linear-order results lost the domain factorial during the staged decoder
   refactor. The engine restores it before reduction decoders (and still skips
   it for propositional axioms mode).
5. FastV2 crashed on an unsatisfiable problem because independent-set analysis
   called NetworkX on an empty graph. Empty cell graphs now produce zero.

## Remaining performance difference

Some legacy lowering/algorithm combinations are still materially slower even
though native incremental3 or propositional paths are fast and exact:

| Case | Old | Refactored |
|---|---:|---:|
| BA / incremental | 0.70 s | >10 s (timeout) |
| permutation-no-fix-sc2 / fastv2 | 0.60 s | >10 s (timeout) |
| books arrangement / incremental | 0.65 s | >10 s (timeout) |

The restored tests retain these old combinations behind
`WFOMC_RUN_SLOW=1`. Default CI uses incremental3 for native counting/ordinary
linear-order cases and propositional for PRED/CIRCULAR MATH cases. Therefore
the refactor is result-compatible on the tested matrix, but it is not accurate
to claim zero performance impact for every legacy algorithm path.
