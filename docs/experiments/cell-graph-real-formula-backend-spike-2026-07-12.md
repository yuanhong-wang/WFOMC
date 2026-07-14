# Real-formula cell-graph backend spike

Date: 2026-07-12

## Purpose

The first synthetic spike showed that projected SAT enumeration is appropriate
for cells and that pair models should be compiled and summed instead of
materialized. This follow-up tests that conclusion on the repository's actual
typed FOL grounding and Tseitin CNF path, including exact rational weights and
the counting relation table required by incremental3.

The original one-off benchmark was consolidated into
`benchmarks/cell_graph_backends.py` after the production backend was selected.
The experiment did not change production code or runtime dependencies at the
time it was run.

## Formula and data flow

The quantifier-free FO2 formula contains:

- four unary predicates, including one free unary extension;
- four binary predicates, including one free binary extension;
- implications between unary types and both binary orientations;
- an equivalence defining one binary relation;
- a disjunction coupling both orientations of another relation.

It is grounded with the same `ground_on_tuple`, explicit local atom universe,
and `to_cnf_clauses` Tseitin encoder used by the cell-graph path.

With `PYTHONHASHSEED=0` this produces:

- cell CNF: 8 original atoms, 13 Tseitin auxiliaries, 43 clauses;
- pair CNF: 16 original atoms, 25 Tseitin auxiliaries, 84 clauses;
- 28 valid cells and 784 cell-pair queries;
- 2,544 satisfying original pair models in the native reference;
- six incremental3-compatible projection bits: `R(ba), R(ab), T(ba), T(ab),
  V(ba), V(ab)`.

All auxiliary literal weights are one. Cell-condition literal weights are
zero/one, so unary and diagonal weights are not counted twice. Off-diagonal
relation weights use `ArithmeticContext(ArithmeticBackend.FMPQ)`.

## Strategies

- Native: enumerate original pair models, retain them, then filter for every
  cell pair as the current `TwoTable.models` path does.
- PySDD per query: compile once, then perform one exact custom circuit traversal
  per cell pair.
- PySDD batch: project both the cell-condition atoms and counting atoms and
  construct the complete factor table in one exact traversal.
- GANAK d-DNNF per query/batch: compile the same unweighted Boolean CNF, run
  `ddnnf-cleanup`, then evaluate the cleaned circuit with the same exact factor
  algebra.

The repository-pinned GANAK commit
`82a1d1fb6f0d6fb4a46b825f84b29567728ae483` does not recognize `--compile`.
The GANAK rows therefore use a temporary build of current official master at
commit `0b4881b40126fb5ae8a1b7c9fdd4878afa25b051`; this does not change the
repository pin.

## Correctness checks

Before timings are accepted, the benchmark verifies:

- native and PySAT produce exactly the same 28 cell codes;
- all five pair strategies produce exactly the same 784 factor rows;
- every row agrees on every nonzero counting mask and exact `fmpq` weight;
- all strategies produce 4,672 nonzero entries after expanding the compact
  factors to the current cell-pair matrix shape.

Thus this experiment checks the complete counting factor, not only its scalar
sum.

## Results

Apple M4, Python 3.11.7, PySAT 1.9.dev5, PySDD 1.0.0. Times are the median of
three runs in milliseconds.

| strategy | build | query/materialize | total | retained models | circuit nodes |
|---|---:|---:|---:|---:|---:|
| native cell | 1.079 | 0 | 1.079 | 28 | - |
| PySAT projected cell | 0.143 | 0 | 0.143 | 28 | - |
| native pair materialization | 396.089 | 1094.184 | 1490.273 | 2,544 | - |
| PySDD exact, per query | 47.099 | 2671.734 | 2718.833 | 0 | 2,219 |
| PySDD exact, batch | 42.181 | 36.311 | 78.491 | 0 | 2,219 |
| GANAK d-DNNF exact, per query | 119.122 | 140.895 | 260.017 | 0 | 30 |
| GANAK d-DNNF exact, batch | 116.960 | 12.875 | 129.834 | 0 | 30 |

## Interpretation

PySAT remains the correct cell backend: projected enumeration is about 7.5
times faster here while returning the actual required cells. Tseitin variables
are existentially hidden by blocking only original atom variables.

The API shape matters as much as the compiler. Exact PySDD traversal once per
cell pair is slower than native materialization because it walks a 2,219-node
circuit 784 times. The same circuit evaluated once as a batched factor is about
19 times faster than native overall. Production should therefore not expose
only `weight(assumptions)` and call it in a nested cell-pair loop.

GANAK produces a much smaller cleaned circuit on this CNF: 30 nodes versus
2,219 SDD nodes. Its batch circuit evaluation is about three times faster than
PySDD's batch evaluation, but external compilation and cleanup cost more. For
one build, in-process PySDD batch has the best total time in this experiment;
for a circuit reused across weights or arithmetic contexts, GANAK's smaller
circuit has the better steady-state shape.

The batch result should be keyed by the atoms that actually condition a pair.
The 28 cells produce 784 matrix positions but only 144 distinct pair-condition
signatures because diagonal binary cell bits do not occur in this pair formula.
Materializing one `TwoTable` per cell pair repeats identical factors. A compact
`cell -> pair-signature` mapping can remove that duplication without changing
algorithm inputs.

## Grounding determinism issue found by the spike

`ground_on_tuple` currently forms `variables = tuple(free_vars(formula))`.
`free_vars` is a set-like result, so the mapping from `(X, Y)` to `(a, b)` is not
stable across hash seeds. In this logically symmetric pair construction the
result remains equivalent, but its syntax and Tseitin size differ:

| `PYTHONHASHSEED` | pair auxiliaries | pair clauses |
|---:|---:|---:|
| 0 | 25 | 84 |
| 1 | 29 | 96 |
| 2 | 25 | 84 |
| 3 | 29 | 96 |

This damages reproducible benchmarks, structural cache keys, and canonical
compilation. Calls that do not immediately conjoin both orientations may also
observe an unintended variable swap. The grounding function should use a
stable variable order rather than iterating the set directly.

## Recommended production direction

1. Enumerate cells with PySAT assumptions and blocking clauses over original
   cell atoms only.
2. Lower the pair formula once and preserve explicit original/auxiliary variable
   metadata.
3. Compile one Boolean circuit and evaluate one batched exact factor over cell
   condition bits plus incremental3 counting bits.
4. Store factors by distinct pair-condition signature; derive the ordinary
   scalar pair matrix by summing each factor.
5. Keep the unary direct-evaluation path for formulas with no off-diagonal
   binary choices.
6. Fix deterministic grounding before introducing structural circuit caching.
7. Benchmark several real formula families and vtree/variable orders before
   choosing PySDD or upgrading the pinned GANAK compiler. GANAK currently has
   the better reusable circuit, while PySDD has simpler and faster in-process
   construction on this single case.

## Limits

This remains a microbenchmark with one real formula. It does not measure peak
resident memory, symbolic polynomial factors, linear-order predecessor tables,
or an end-to-end incremental3 solve. The current GANAK master compiler is newer
than the repository pin and its d-DNNF path must not become a production
dependency without pinning, corpus-level differential tests, and malformed or
oversized circuit handling.
