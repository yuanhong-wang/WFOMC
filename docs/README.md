# WFOMC Documentation

This directory only keeps current framework documentation and active plans.
Historical compiler/backend/lowering plans were removed because they no longer
match the codebase.

## Current Architecture

- `architecture-review-2026-07-10.md`: current-worktree architecture,
  verified failure modes, prioritized problems, target boundaries, and ADR
  recommendations.
- `adr/0013-evidence-stage-boundaries.md`: raw, reduced, and materialized
  evidence ownership.
- `adr/0014-cell-graph-build-boundary.md`: immutable cell-graph output,
  branch-owned arithmetic, and fast composition.
- `adr/0018-cell-graph-predicate-universe.md`: predicate-vocabulary
  preservation and sound unary/binary pair-table construction.
- `adr/0019-pysat-pysdd-cell-graph-backend.md`: projected cell enumeration and
  exact hybrid pair-factor construction.
- `adr/0023-bounded-pysat-and-metadata-predicate-universe.md`: bounded PySAT
  pair fast path, analytical free atoms, and compact condition factors.
- `adr/0024-ganak-primary-pair-factor-backend.md`: one-shot exact Ganak
  pair-factor WMC with PySDD as an automatic in-process backup.
- `adr/0020-unified-cell-graph-package.md`: one shared cell-graph package for
  construction, components, and evidence materialization.
- `adr/0021-error-ownership-and-public-hierarchy.md`: stable public error
  categories, domain ownership, and removal of the ambiguous planning error.
- `adr/0022-standard-library-logging.md`: default-silent library logging,
  CLI verbosity, and bounded phase/timing diagnostics.
- `adr/0015-algorithm-package-convention.md`: consistent spec/input/solve
  ownership across algorithm packages.

The earlier framework architecture pages were removed during the active
framework migration. Use the dated review above as the current snapshot and
the plans below as proposed or in-progress work, not as descriptions of the
running system.

## Active Plans

- `plans/2026-07-10-readable-algorithm-cell-graph-simplification.md`:
  implemented readability and cell-graph boundary simplification.
- `plans/2026-07-10-problem-stage-types.md`: implemented problem-stage
  separation and incremental3 counting preparation.
- `plans/2026-07-10-algo-capabilities-and-typed-boundaries.md`: implemented
  algorithm maturity metadata, CLI filtering, and typed stage boundaries.
- `plans/2026-07-10-branch-arithmetic-context.md`: implemented branch-owned
  arithmetic context propagation and supported rounded backends.
- `plans/2026-07-10-root-numeric-module-interfaces.md`: implemented explicit
  root numeric interfaces and removal of legacy numeric utilities.
- `plans/2026-07-06-algo-spec-owned-reduction-materialization.md`: completed
  framework boundary decision.
- `plans/2026-07-06-fol-formula-migration.md`: active typed FOL migration plan.
- `plans/2026-07-12-incremental3-existential-strategies-design.md`: counting
  and weighted-Skolem existential preparation for incremental3.
- `plans/2026-07-13-cell-graph-small-model-and-free-atom-optimization.md`:
  bounded pair enumeration and predicate-universe formula cleanup.
- `plans/2026-07-13-ganak-primary-pair-factor-backend.md`: implemented shared
  Ganak boundary and bounded PySAT/Ganak/PySDD pair-factor routing.
- `plans/2026-07-15-direct-propositional-grounding.md`: implemented direct
  source-formula grounding for the Ganak-backed propositional correctness
  oracle.

## Experiments

- `experiments/cell-graph-pysdd-ganak-polynomial-2026-07-13.md`: exact batched
  PairFactor comparison between the current PySDD evaluator and one-shot Ganak
  polynomial WMC on real and signature-scaling workloads.
- `experiments/incremental3-existential-strategies-2026-07-12.md`: exact-result
  and runtime comparison of counting versus Skolemization.
- `experiments/pre-refactor-test-parity-2026-07-12.md`: restored test coverage,
  exact-result parity, timings, fixed regressions, and remaining slow paths.

## Current Core Flow

```text
Problem
  -> source feature analysis + option resolution
  -> lifted algorithms: ReducedProblem(C2NormalForm)
       -> algorithm reduction chain
       -> CompiledProblem(QF formula + compiled weights + branch arithmetic)
  -> propositional: direct finite-domain grounding of the source Problem
       -> model-preserving ground CNF + source-compiled branch arithmetic
  -> algorithm-owned AlgoInput sharing the branch arithmetic
  -> solve
  -> decode
  -> WFOMCResult
```
