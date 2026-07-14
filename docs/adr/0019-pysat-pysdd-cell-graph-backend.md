# ADR 0019: Use projected SAT and batch exact SDD factors for cell graphs

## Status

Accepted — 2026-07-12; amended by ADR 0023 and ADR 0024 — 2026-07-13

## Context

Cell-graph construction used Python truth-table enumeration twice. Diagonal
models became cells, while every satisfying two-element model was retained in
`TwoTable.models` and repeatedly filtered for each cell pair and incremental3
evidence mask. Replacing the Python loop with SAT model enumeration improved
some cases but preserved the exponential pair-model materialization.

Microbenchmarks showed different requirements for the two stages. Cells are a
required output, so projected SAT enumeration is output-sensitive and useful
when valid cells are sparse. Pair algorithms need weighted aggregates rather
than models, so the Boolean structure should be compiled and evaluated once.
The real-formula experiment also showed that walking a compiled circuit once
per cell pair can be slower than native enumeration; condition and counting
bits must be evaluated as one batch factor.

## Decision

- Represent Tseitin CNF as a typed `TseitinCNF` containing original atom
  mappings, clauses, variable count, and auxiliary variables.
- Order free variables deterministically before grounding.
- Use PySAT with CaDiCaL to enumerate cell assignments projected to original
  diagonal atoms. Blocking clauses never contain Tseitin auxiliaries.
- For a bounded number of valid condition signatures, first try exact PySAT
  enumeration with one global model budget. If the budget is exhausted,
  discard the partial result and compile the pair CNF with PySDD.
- Use PySDD to compile pair CNF once per grounded pair formula on the fallback
  path.
- Traverse the SDD with the branch's `ArithmeticContext`; do not use PySDD's
  floating-point WMC for solver results.
- Project cell-condition atoms and requested incremental3 binary atoms in one
  traversal. Materialize pure `PairFactor` values and release the SDD before
  returning `CellGraphData`.
- Store scalar `total_weight` and optional sparse `counting_weights`; do not
  retain satisfying pair models or solver objects.
- Keep the unary-only direct evaluation path.
- Pass projected binary predicates explicitly from incremental3. The base cell
  graph does not import or inspect algorithm types.
- Express overlapping profile membership as a disjunction of profile formulas.
  Do not introduce hidden selector assignments whose multiplicity would change
  WMC.

## Dependency boundary

- `fol/cnf.py` owns Boolean encoding and imports no solver.
- `cell_graph/enumerate_cells.py` and
  `cell_graph/compute_pair_factors.py` are the only production modules
  importing PySAT.
- `cell_graph/compute_pair_factors.py` is the only production module importing
  PySDD.
- `cell_graph/build.py` orchestrates typed values only.
- Algorithms consume `CellGraphData` and `PairFactor`, never CNF or SDD nodes.

## Consequences

Cell construction no longer scans every Boolean assignment when only a few
cells are valid. Pair construction no longer stores or filters complete models,
and incremental3 receives exact mask weights directly. Rational and symbolic
weights use the same arithmetic context as the rest of the branch.

PySAT enumeration remains linear in the number of cells that must be returned.
The first implementation also expands compact pair factors back to a cell-pair
matrix, so very large cell sets can still require quadratic storage. A later
stage may aggregate full cells into algorithm-visible signatures; this ADR does
not authorize merging cells before the downstream equivalence contract is
defined.

SDD size depends on variable order and formula structure. Production must fail
explicitly on compilation errors and keep differential tests against native
reference semantics for small formulas. Native model enumeration remains a
test oracle but is not a cell-graph production fallback.

ADR 0024 changes the non-enumerating routing from direct PySDD fallback to
Ganak first with PySDD retained as an automatic in-process backup. The typed
CNF, batch factor, arithmetic, and ownership decisions in this ADR remain in
force.
