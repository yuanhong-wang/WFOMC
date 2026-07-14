# ADR 0024: Prefer Ganak for non-enumerating pair factors

## Status

Accepted — 2026-07-13

## Context

Bounded PySAT enumeration is the cheapest exact path for small local pair
theories, but it must discard partial output once its global model budget is
exceeded. The existing PySDD batch evaluator avoids model enumeration and
preserves exact branch arithmetic, yet its fixed balanced vtree can have a
severe structural compilation blow-up.

The paired benchmark in
`docs/experiments/cell-graph-pysdd-ganak-polynomial-2026-07-13.md` found that
one Ganak polynomial WMC was 9.3x, 29.2x, and 1448.6x faster on the three
repository pair CNFs. PySDD was faster on some regular large-output shapes,
but timed out beyond 120 seconds on a 200-cell constrained shape that Ganak
completed in about 236 ms. Cell count, signature count, and model count alone
therefore do not provide a reliable selector for the current PySDD vtree.

## Decision

- Route pair factors through `bounded PySAT -> Ganak -> PySDD backup`.
- Preserve the existing PySAT condition-signature and global-model limits.
- When PySAT is skipped or exhausts its budget, invoke Ganak once with one
  private polynomial marker per condition or incremental3 projection bit.
- Interpret each multilinear marker monomial as a projected truth mask and its
  coefficient as the exact branch-arithmetic weight. Existing solver symbols
  remain coefficient variables rather than mask variables.
- Support exact `FMPQ`, `FMPQ_POLY`, and `FMPQ_MPOLY` branches. Rounded
  arithmetic declines Ganak and uses PySDD.
- Give each pair Ganak call a 30-second timeout. A missing binary, timeout,
  process failure, parse failure, or unsupported arithmetic is non-fatal for
  cell-graph construction and selects PySDD.
- Keep PySDD installed and keep its exact custom factor traversal. Do not use
  floating-point PySDD WMC for exact branches.
- Put binary discovery, weighted DIMACS serialization, deterministic process
  execution, exact parsing, and pinned metadata in `wfomc.ganak`. Both the
  propositional algorithm and cell graph reuse this boundary.
- Keep marker construction and coefficient recovery in
  `cell_graph/compute_pair_factors_ganak.py`; the generic Ganak adapter has no
  knowledge of cells or condition signatures.
- Preserve `CellGraphData`, `PairFactor`, and algorithm input contracts. One
  immutable factor is still materialized per distinct condition signature and
  shared by equivalent cell pairs.

## Consequences

The non-enumerating default is robust against the observed PySDD structural
blow-ups while remaining exact and batch-oriented. Ganak is an optional
external runtime for lifted algorithms because PySDD provides a transparent
backup; it remains a required scalable backend for the propositional
algorithm.

Ganak process startup and polynomial output parsing can be slower than PySDD
on regular large-output formulas. The chosen order optimizes robustness rather
than claiming Ganak is universally fastest. INFO logs expose
`backend=ganak` or `backend=pysdd-backup` and separate enumeration, Ganak,
compile, evaluation, and materialization timings so future corpus evidence can
support a more structural selector.
