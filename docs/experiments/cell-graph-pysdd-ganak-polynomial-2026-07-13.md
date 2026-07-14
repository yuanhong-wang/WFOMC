# Cell-graph PySDD vs Ganak polynomial WMC

Date: 2026-07-13

## Question

Can one Ganak weighted model-counting call build the complete cell-graph
`PairFactor` table faster than the current one-traversal PySDD backend?

This is not a comparison with the small-model PySAT fast path. It compares the
two non-enumerating candidates for cases where explicit SAT model enumeration
is no longer attractive.

## Integration compared

Both backends consume the same `TseitinCNF`, exact predicate weights, condition
atoms, and incremental3 projection bits.

- PySDD constructs the current balanced-vtree SDD and evaluates it once with
  the exact batched factor algebra in `compute_pair_factors.py`.
- Ganak receives one exact polynomial WMC instance. A fresh polynomial variable
  marks every condition/projection bit. Grouping the result polynomial by those
  marker exponents recovers the complete factor table in one invocation.

For example, a term `3*x0*x4` contributes weight `3` to the factor entry whose
condition/projection mask has bits 0 and 4 set. Existing symbolic WFOMC weights
remain separate polynomial variables, so this also works for the internal
cardinality symbols in the books and predecessor models.

Ganak is invoked in deterministic exact mode (`--prob 0 --mode 3`). Its timing
includes polynomial-weight construction, DIMACS serialization, temporary-file
I/O, process startup, exact counting, output parsing, coefficient extraction,
and the free-off-diagonal factor. PySDD timing includes in-process SDD
construction, exact factor evaluation, and the same free factor.

For rows where both backends finish, every timed trial checks equality of every
nonzero mask and exact coefficient. A mismatch aborts the benchmark. The
200-cell Ganak result, for which PySDD times out, is checked independently
against exact PySAT enumeration of all 40,000 pair models.

## Environment

- Apple Silicon (`arm64`), macOS 15.7.4
- Python 3.11
- PySDD 1.0.6
- python-sat 1.9.dev5
- python-flint 0.8.0
- repository-pinned Ganak commit
  `82a1d1fb6f0d6fb4a46b825f84b29567728ae483`
- one Ganak thread; deterministic cache mode
- three or five repetitions as noted; tables report the median
- explicit outer timeouts on every benchmark command

Command:

```bash
/opt/homebrew/bin/timeout 300s uv run python \
  scripts/benchmark_pair_factor_backends.py \
  --trials 3 \
  --timeout 30 \
  --cell-counts '' \
  --filter 'books/pair-1,predecessor/pair-1,markov3/pair-1'
```

## Real pair-CNF results

`pair-1` is the base pair formula captured from each normal solver run.

| workload | vars | clauses | condition signatures | marker bits | factor entries | PySDD compile | PySDD evaluate | PySDD total | Ganak count | Ganak extract | Ganak total | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| books/pair-1 | 38 | 84 | 4 | 10 | 16 | 71.770 ms | 7.592 ms | 79.338 ms | 8.174 ms | 0.203 ms | 8.527 ms | 9.30x |
| predecessor/pair-1 | 44 | 106 | 4 | 6 | 4 | 240.894 ms | 2.424 ms | 243.322 ms | 8.050 ms | 0.099 ms | 8.334 ms | 29.20x |
| markov3/pair-1 | 62 | 140 | 16 | 8 | 16 | 13204.330 ms | 3.235 ms | 13207.570 ms | 8.797 ms | 0.120 ms | 9.118 ms | 1448.56x |

The speedup column is `PySDD total / Ganak total`; values greater than one
favor Ganak.

A fresh first Ganak call was also observed at about 26 ms before the operating
system's dynamic-library and page caches were warm. Even that cold result was
faster than PySDD on all three real CNFs. Ten subsequent Ganak runs had these
total-time medians: books 6.329 ms, predecessor 5.845 ms, and markov3 6.280 ms.

## Actual-cell scaling family

The earlier marker-only scaling test was not a cell-count experiment: its
3Markov source graph still had four cells. It has therefore been replaced by a
typed FO2 family whose cells and signatures are constructed through the real
grounding, Tseitin, and projected-cell-enumeration path.

For each type bit, the family contains

```text
P_i(X) <-> R_i(X,Y)
```

The diagonal grounding admits one cell per unary bit code. The pair grounding
depends on both the left and right codes. Consequently, `N` actual cells
produce exactly `N^2` actual cell pairs and distinct condition signatures. For
non-power-of-two counts, a compact Boolean `code < N` constraint selects
exactly the requested number of cells.

The 8--128 rows use five repetitions. The 256 row uses three repetitions.

| cells | cell pairs/signatures | vars | clauses | factor entries | PySDD compile | PySDD evaluate | PySDD total | Ganak count | Ganak extract | Ganak total | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 19 | 32 | 64 | 8.218 ms | 0.466 ms | 8.678 ms | 6.772 ms | 0.181 ms | 6.986 ms | 1.24x |
| 16 | 256 | 25 | 42 | 256 | 9.150 ms | 0.918 ms | 10.071 ms | 7.485 ms | 0.642 ms | 8.167 ms | 1.23x |
| 32 | 1,024 | 31 | 52 | 1,024 | 11.829 ms | 2.863 ms | 14.667 ms | 9.392 ms | 2.602 ms | 12.066 ms | 1.22x |
| 64 | 4,096 | 37 | 62 | 4,096 | 15.787 ms | 7.621 ms | 23.575 ms | 16.344 ms | 11.080 ms | 27.659 ms | 0.85x |
| 128 | 16,384 | 43 | 72 | 16,384 | 45.506 ms | 28.347 ms | 74.157 ms | 47.477 ms | 47.674 ms | 95.928 ms | 0.77x |
| 256 | 65,536 | 49 | 82 | 65,536 | 103.831 ms | 95.460 ms | 198.841 ms | 176.634 ms | 195.075 ms | 371.744 ms | 0.53x |

For this regular power-of-two family Ganak is slightly faster through 32 cells;
the current in-process PySDD path becomes faster from 64 cells onward because
materializing and parsing the polynomial coefficients dominates Ganak's cost.
At 256 cells PySDD is about 1.9 times faster.

### Exact 200-cell case

The 200-cell family has eight type bits plus `code < 200`. It produces exactly
40,000 actual cell pairs, condition signatures, and factor entries.

| cells | pairs/signatures | vars | clauses | PySDD total | Ganak count | Ganak extract | Ganak total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 40,000 | 65 | 130 | >120 s (timeout) | 112.447 ms | 124.444 ms | 235.941 ms |

The Ganak times are medians of five runs. Exact PySAT enumeration independently
produced the same 40,000 coefficients in 1.332 seconds. PySDD was run alone
under a 120-second hard timeout and did not finish.

The nearby 256-cell row completes in 199 ms with PySDD. Therefore the 200-cell
timeout is not caused by cell count alone: the non-power-of-two allowed-cell
constraint interacts badly with the current fixed SDD vtree/order, while
Ganak's search and component caching remain robust on this shape.

## Interpretation

For the three repository CNFs, one Ganak polynomial WMC is decisively faster
than the current PySDD implementation. The largest real difference is caused
by SDD construction: the markov3 SDD needs a median 13.2 seconds to compile,
while its exact batched evaluation takes only 3.2 ms. Ganak returns the
complete exact factor in about 9.1 ms total.

The actual-cell family gives a more nuanced result. With a regular Boolean
shape and a large factor output, PySDD's in-process traversal can outperform
Ganak's polynomial serialization and parsing. But the 200-cell case shows that
the current fixed SDD vtree can also suffer a catastrophic structural blow-up.
Neither cell count nor total model count is sufficient to select a backend.

This supports using Ganak as the primary non-enumerating fallback when the
PySAT path is expected to enumerate too many local models. It does not support
removing PySAT: on the current small pair theories, bounded PySAT enumeration
still avoids Ganak's process startup and is faster.

It also does not establish that every possible PySDD integration is slow. The
current implementation fixes a balanced vtree and numeric variable order.
Formula-aware vtree search could reduce SDD construction substantially. The
measured conclusion is specifically that Ganak beats the current production
PySDD backend on these workloads.

## Suggested routing direction

1. Keep PySAT for small, sparse pair theories.
2. When enumeration exceeds its local budget, prefer one Ganak polynomial WMC
   as the robust fallback; it is not always fastest, but avoided the current
   PySDD path's 200-cell structural blow-up.
3. Retain PySDD as an in-process fallback when Ganak is unavailable until the
   external dependency policy is settled.
4. Treat factor-entry count as a materialization-cost estimate, not as the sole
   backend selector. Formula structure and SDD variable order dominate some
   cases.
5. Before changing production routing, add a corpus experiment containing pair
   formulas with genuinely large hidden satisfying-model sets. This cell family
   deliberately has one pair model per condition signature and tests cell
   scaling instead.

## Reproduction artifact

The benchmark is
`scripts/benchmark_pair_factor_backends.py`. It does not mutate production
backend selection while measuring. ADR 0024 adopts the measured routing as
`bounded PySAT -> Ganak -> PySDD backup`; future corpus cases should continue
to be added to this benchmark before introducing a more granular selector.
