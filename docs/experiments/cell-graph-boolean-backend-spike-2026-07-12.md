# Cell-graph Boolean backend spike

Date: 2026-07-12

## Question

The current cell-graph builder scans all Boolean assignments and materializes
all satisfying pair models. This spike tests whether a SAT solver is enough,
or whether pair weights need a non-enumerating WMC/knowledge-compilation path.

The benchmark is intentionally separate from production code. It compares:

- native scanning of every assignment;
- PySAT/CaDiCaL enumeration with blocking clauses;
- exact GANAK WMC invoked once per complete pair condition;
- PySDD compilation once followed by repeated conditioned WMC.

Every backend must return the same cell set or complete pair-weight table. A
mismatch aborts the benchmark.

## Workloads

The synthetic CNFs isolate the two shapes used by cell-graph construction:

- `cell-exactly-one-18`: 18 cell bits but only 18 valid cells;
- `cell-free-14`: 14 unconstrained cell bits and all 16,384 valid cells;
- `pair-u2-b8`: four conditioned cell bits, eight hidden binary bits, and 16
  pair queries;
- `pair-u3-b10`: six conditioned cell bits, ten hidden binary bits, and 64
  pair queries.

The hidden pair bits have non-unit integer literal weights. Native and PySAT
materialize every satisfying pair model and then filter the list for every
query, matching the current `TwoTable.models` shape. PySDD sums hidden bits in
the compiled circuit instead.

## Environment

- Apple M4, macOS 15.7.4
- Python 3.11.7
- PySAT 1.9.dev5 with CaDiCaL 1.9.5
- PySDD 1.0.0
- repository-pinned GANAK commit `82a1d1fb6f0d6fb4a46b825f84b29567728ae483`
- five repetitions; table reports the median

Historical command (the one-off spike was removed after the production backend
was selected; use `benchmarks/cell_graph_backends.py` for current measurements):

```bash
uv run --with python-sat --with pysdd \
  python benchmarks/cell_graph_backend_spike.py \
  --repetitions 5 \
  --output benchmarks/results/cell_graph_backend_spike.csv
```

## Results

Times are milliseconds for the complete workload, including construction and
all queries.

| workload | strategy | build | query | total | materialized models | compiled nodes |
|---|---:|---:|---:|---:|---:|---:|
| cell-exactly-one-18 | native scan | 538.219 | 0 | 538.219 | 18 | - |
| cell-exactly-one-18 | PySAT projected enum | 0.156 | 0 | 0.156 | 18 | - |
| cell-free-14 | native scan | 29.521 | 0 | 29.521 | 16,384 | - |
| cell-free-14 | PySAT projected enum | 51.651 | 0 | 51.651 | 16,384 | - |
| pair-u2-b8 | native materialize | 6.689 | 0.159 | 6.848 | 1,120 | - |
| pair-u2-b8 | PySAT materialize | 4.072 | 0.162 | 4.233 | 1,120 | - |
| pair-u2-b8 | PySDD compile once | 5.982 | 0.019 | 6.001 | 0 | 67 |
| pair-u2-b8 | GANAK per query | 0 | 81.265 | 81.265 | 0 | - |
| pair-u3-b10 | native materialize | 136.643 | 10.437 | 147.080 | 20,864 | - |
| pair-u3-b10 | PySAT materialize | 85.683 | 10.562 | 96.246 | 20,864 | - |
| pair-u3-b10 | PySDD compile once | 5.792 | 0.070 | 5.863 | 0 | 43 |
| pair-u3-b10 | GANAK per query | 0 | 349.833 | 349.833 | 0 | - |

## Interpretation

PySAT is a strong fit for cell enumeration when constraints eliminate most
assignments. In the sparse case it is roughly 3,450 times faster than native
scanning. It cannot remove the cost of producing the cells themselves: when
all 16,384 assignments are valid, projected SAT enumeration is about 1.75
times slower than the simple native loop.

Replacing native pair enumeration with SAT enumeration does not fix the pair
architecture. Both strategies retain exactly 20,864 models in the larger pair
case. PySAT reduces construction time, but the same model list remains and the
per-query filtering cost is unchanged.

Compile-once WMC changes the scaling shape. On the larger pair case PySDD is
about 25 times faster than native materialization and 16 times faster than
PySAT materialization. Its 64 conditioned queries take 0.070 ms in total,
versus about 10.5 ms for filtering either materialized model list. On the small
case, compilation overhead consumes the advantage, which suggests keeping the
unary/direct-evaluation fast path and considering a small-formula threshold.

GANAK produces exact results, but starting one process per pair query is the
wrong integration boundary. Fixed process and serialization cost makes it
slower than native enumeration in these tiny workloads. GANAK remains a good
candidate if its d-DNNF is compiled once and evaluated repeatedly in process.

## Decision supported by this spike

1. Use projected PySAT enumeration for cell construction, because the output
   really is the set of valid cells.
2. Do not use SAT model enumeration for pair tables and do not retain
   `TwoTable.models` in the production hot path.
3. Compile the pair Boolean structure once, condition it for each cell pair,
   and evaluate weights without enumerating hidden binary assignments.
4. Do not invoke the current GANAK subprocess adapter once per cell pair.
5. Before selecting PySDD as the production compiler, compare it with a
   repository-pinned GANAK d-DNNF compiler and evaluate both through the same
   exact `ArithmeticContext` circuit traversal.

## Limits and next experiment

This is a microbenchmark, not an end-to-end WFOMC benchmark. The CNFs are
synthetic, contain no Tseitin auxiliary variables, and do not measure resident
memory. PySDD's built-in WMC uses floating-point values in this spike; it was
used only after exact native, PySAT, and GANAK results agreed, with a numeric
tolerance check.

The next spike should use real `gnd_formula_cc` and `gnd_formula_ab` formulas
from the repository, add neutral Tseitin auxiliaries, and compare exact custom
evaluation of PySDD and GANAK-generated d-DNNF circuits. It should also produce
the compact counting-predicate mask table required by incremental3.

That follow-up is now recorded in
`cell-graph-real-formula-backend-spike-2026-07-12.md`. Its main refinement is
that exact circuit evaluation must be batched across cell-condition bits;
walking even a compiled circuit once per cell pair can still be slower than
native materialization.
