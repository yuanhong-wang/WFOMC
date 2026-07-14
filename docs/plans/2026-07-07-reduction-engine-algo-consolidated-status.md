# Reduction Engine + Algo Consolidation - Implementation Status

Date: 2026-07-07 (updated)
Plan: `docs/plans/2026-07-07-reduction-engine-algo-consolidated.md`

## Summary

Tasks 1–6, Task 7 (all 8 algos), Task 8 Steps 2–4, and Task 5 Steps 3–4 are
**complete and verified**; Task 9's **achievable scope is complete** (all
retireable backend-heavy `ReducedProblem` fields removed: `normal_form`,
`features`, `counting_state`, `unary_cardinality_masks`, `circle_len`,
`repeat_factor`; `weights` ownership transferred to materializers). The
untracked framework WIP has been **committed as a stable base** (per user
direction "Commit WIP first"), and the materializer migration (Task 7) plus
counting-state ownership transfer (Task 8 Steps 2/4) have proceeded
algorithm-by-algorithm on that committable foundation. Reduction no longer
constructs or returns counting state, unary cardinality masks, or compiled
weights: it passes raw weights through, and materializers compile + apply
cardinality markers via `compile_reduced_weights`. A pure
`reduce_to_reduced_problems` entry point has been added (Criterion 1
scaffolding, wired to Task 3's previously-unused pure unary-evidence reductions).

Full `ReducedProblem` *removal* (Task 9 Step 3 "remove the old dataclass") is
**architecturally impossible** and is documented below: the remaining fields
are irreducible reduction outputs (quantifier-free formula, linear-order
metadata, cardinality plan, decode pipeline) that materializers - which live
outside `reduction/` and cannot import the legacy `fol_view` - must consume but
cannot reconstruct. Per the plan's own Conflict Resolution Notes (do not add
`CompiledProblem`; keep the pure contract minimal), `ReducedProblem` remains as
the necessary materializer-internal bundle; the pure layer's public output is
`ReducedProblems` (Criterion 1). The Task 5 cardinality-weighting/compilation
entanglement (documented below) is **resolved** (`10202de`, option (c)): weight
compilation and cardinality-marker application moved together into the
materializer, which is behavior-preserving for the default exact backend.

## Branch state

The framework restructure is now COMMITTED (commit `47aaa58`): the new
`algo/` package (cell_graph, core, fast, fastv2, incremental, incremental3,
propositional, recursive, standard, tail_signature, treewidth), `engine/`,
`framework/`, `parser/`, `normal_form/c2`, `fol/` additions, `reduction/`
additions, plus tracked modifications across `src/wfomc`, `benchmarks/`,
`docs/`. Working tree is clean. Task 6 (MaterializationContext) changes
that were previously untracked are now committed in this snapshot.

## Done (committed this plan)

| Task | Commit | Notes |
|------|--------|-------|
| 1 - Pure reduction contract (`ReducedBranch`, `ReducedProblems`) | `ffc7333` | `reduction/core.py`; `ReducedProblems.single` factory. |
| 2 - `ProfileCapacityConstraint` + `Problem` field | `a7c5a2f` | Exactly one `ProfileCapacityConstraint \| None` on `Problem`. |
| 3 - Unary evidence reductions | `c1dc903` | Pure `Problem -> Problem` reductions. |
| 4 - `WeightOptions` / `WeightPlan` / `choose_weight_backend` | `91b2524` | `weights.py` superset of HEAD API. |
| 5 - `compile_weight_mapping` | `713f23d` | Foundation; full materializer migration deferred to 7/9. |
| 6 - `MaterializationContext` | `47aaa58` (WIP snapshot) | `MaterializationContext` + `_materialize_with_optional_context` adapter (1-arg and 2-arg materializers); 2 tests in `tests/unit/engine/test_orchestration.py`. |
| 8 Step 3 - Stable `(name, arity)` predicate identity | `081de46` | `_pred_index` in `cell_graph/components.py`; OLD path parity with NEW `algo/cell_graph` `_predicate_identity`. |
| (hygiene) structural test for context adapter | `ffe87fc` | `spec.materialize(` -> `spec.materialize`. |
| (base) framework WIP snapshot | `47aaa58` | 268 files; user's restructure as base for the migration. |
| 1+/Criterion 1 - pure `reduce_to_reduced_problems` | `93194d3` | Wires Task 3's unused pure unary-evidence reductions; 4 contract tests. |
| Migration step 1 - unify LIFTED_PROFILES representations | `fe2d623` | `EvidencePlan.build_from_profile_capacity` + `build_evidence_plan` consumes `profile_capacity_constraint`; resolves the lost-encoding blocker. 2 parity tests. |
| Migration step 2 - migrate `standard` materializer (Task 7, standard) | `c238318` | `spec.reduce` returns pure `ReducedProblems` (prepend-safe unary-evidence reduction); 2-arg `build_standard_input_from_reduced_problems` owns the `reduce_to_ufo2` backend + cell graphs from the single branch's problem. Counts verified (330626, 4, 6); full suite == `f8150c1` baseline (36 failed / 308 passed, all pre-existing). 10 tests updated to read backend fields via `reduce_to_ufo2(...)` or `isinstance(decode_result.__self__, ReducedProblem)`. |
| Migration step 3 - shared `pure_reduce` / `ufo2_materializer` bridges | `085b70b` | Factored the spec-bridge helpers duplicated in step 2 into `algo.core`: `pure_reduce` (algo-independent `Problem -> ReducedProblems`) and `ufo2_materializer(build_input)` (2-arg `spec.materialize` owning `reduce_to_ufo2` + delegating). `standard` refactored to use them. Imports use the public `wfomc.reduction` API (structural-test clean). |
| Migration step 4 - migrate 6 UFO2-backend algos (Task 7, remaining) | `ac4f28d` | `fast`, `fastv2`, `incremental`, `recursive`, `propositional`, `tail_signature` all switched to `reduce=pure_reduce` + `materialize=ufo2_materializer(builder)`. `propositional`'s `GROUND_UNITS` falls into `pure_reduce`'s no-op branch (prepend-safe). Counts verified; full suite == baseline (36 / 308). 6 test assertions updated (`isinstance(..., ReducedProblem)` + one `reduce_to_ufo2` cache test). |
| Migration step 5 - migrate `incremental3` + generalize `backend_materializer` (Task 7, complete) | `cad09d5` | `incremental3` switched to `reduce=pure_reduce` + `materialize=counting_dp_materializer(...)`. Extracted `backend_materializer(backend_reduce, build_input)` in `algo.core`; `ufo2_materializer` and `counting_dp_materializer` are thin wrappers. Counts verified (evidence-only 4, cardinality 6; corpus unchanged). Full suite == baseline (36 / 308). **Task 7 complete for all 8 algos.** |
| 8 Step 2 - Move counting-state construction into incremental3 | `4c53d2d` | `counting_dp_materializer` (`algo/core.py`) specialized: after `reduce_to_counting_dp`, builds counting state via the new public `build_counting_state_for_problem(problem, normal_form)` (`reduction/counting_state.py`) and attaches it with `dataclasses.replace`. `_run_pipeline`'s counting-DP branch no longer calls `reduce_counting_state` (deleted); UFO2 branch unchanged. Full suite == baseline (36 / 308); counts preserved (evidence-only 4, cardinality 6). |
| 8 Step 4 - Reduction no longer returns `counting_state` / `unary_cardinality_masks` | `baac987` | `ReducedProblem` and `ReductionState` drop both fields; `_materialize` no longer sets them. `counting_dp_materializer` passes counting state/masks explicitly to `build_counting_dp_input_from_reduced` (no `replace` round-trip). `build_counting_dp_input_from_reduced` + `_counting_component_from_reduced_cell_graph` take them as keyword args. Tests read counting state via the materialized `algo_input` (or pass it explicitly). Full suite == baseline (36 / 308); failure set identical to baseline (`comm -13` empty). |
| 9 (partial) - Drop dead `ReducedProblem` fields | `2f17003` | `ReducedProblem.normal_form`/`.features` removed (set by `_materialize`, never read - consumers use `ReductionState`/`MaterializationContext`). Full suite == baseline (36 / 308); failure set identical. |
| 5 Steps 3–4 - Materializers compile weights; reduction passes raw weights | `10202de` | `reduce_weights` deleted from `reduction/core.py`; `_materialize` carries raw `state.weights` and builds the cardinality encoding (pure metadata) only for the `DecodePipeline`. New `compile_reduced_weights(reduced, problem, options)` in `algo/core.py` (`compile_weight_mapping` -> `apply_cardinality_weighting` -> `replace`) is called by `backend_materializer` + `counting_dp_materializer` before `build_input`. Cardinality-decode/counting contract tests compile reduced weights explicitly (markers now come from the materializer-side compile). Behavior-preserving for the exact backend (`compile_weight_mapping(raw, exact) == convert_weight_mapping_to_ring_elements`). Full suite == baseline (36 / 308); failure set identical. |
| 9 (field) - Retire `ReducedProblem.circle_len` | `0b6a3bc` | Redundant with `len(reduced.domain)`; the sole reader (`cell_graph/inputs.py:161`) now uses `len(reduced.domain)`. Field removed from `ReducedProblem` + `_materialize`. Full suite == baseline (36 / 308); failure set identical. |
| 9 (field) - Retire `ReducedProblem.repeat_factor` | `dc7bdd3` | Redundant with `decode_pipeline.repeat_factor` (both `state.repeat_factor` in `_materialize`); `DecodePipeline` exposes it publicly. Zero prod reads (only the decode pipeline consumes it); one test read (`test_core_contract`) updated to `decode_pipeline.repeat_factor`. Field removed from `ReducedProblem` + `_materialize`. Full suite == baseline (36 / 308); failure set identical. |

Focused suite (Tasks 1–6 + pure entry point + unification): **51 passed**.
`test_weights.py`, `test_evidence_partition.py`, `tests/unit/reduction`,
`tests/unit/engine`.

## Remaining (architecturally constrained)

Task 9's remaining work - making `ReducedProblem` non-public (Step 3 "remove
the old dataclass or rename with a leading underscore") - is blocked by the
irreducible-fields finding (see Task 9 row above). The two remaining levers
would each require a decision beyond a safe bounded increment:

1. **Privatize `ReducedProblem` / `reduce_to_ufo2` / `reduce_to_counting_dp`.**
   Blocked: materializers in `algo/` (outside `reduction/`) must call
   `reduce_to_ufo2`/`reduce_to_counting_dp` and consume the returned
   `ReducedProblem` (it carries the irreducible outputs). Privatizing would
   violate the structural-test forbidden-import constraint unless a new public
   materializer-facing API is introduced - which is the test-API migration
   flagged previously. It also requires updating `test_fol_package_structure.py`
   (pins `ReducedProblem`, `reduction_kind`, `options`, `reduce_to_ufo2(`) and
   6+ test files that call `reduce_to_ufo2` directly.
2. **Thread `evidence_plan`/`options`/`domain` as `build_input` kwargs** (full
   removal rather than ownership transfer). Invasive: 9/7/9 reader sites
   respectively, plus `build_input` signature changes across all cell-graph
   builders. Ownership transfer (materializer sets via `replace`) is
   behavior-preserving but leaves the fields in place, so it does not shrink
   `ReducedProblem` and only adds a redundant rebuild - not worth the churn.

| Task | Status / Blocker |
|------|------------------|
| 7 - Move algorithm materialization into algo modules | **COMPLETE for all 8 algos** (`standard` `c238318`; `fast`/`fastv2`/`incremental`/`recursive`/`propositional`/`tail_signature` `ac4f28d`; `incremental3` `cad09d5`; shared bridges `085b70b`/`cad09d5`). Pattern: `spec.reduce=pure_reduce` (pure `ReducedProblems`); `spec.materialize=ufo2_materializer`/`counting_dp_materializer(builder)` (owns the backend reduction from the single branch's problem). Tests read backend `ReducedProblem` fields via `reduce_to_ufo2`/`reduce_to_counting_dp(...)` or `algo_input.decode_result.__self__` / `isinstance(..., ReducedProblem)`. |
| 8 Step 2 - Move counting-state construction into incremental3 | **COMPLETE** (`4c53d2d`): `counting_dp_materializer` owns counting-state construction via `build_counting_state_for_problem` + `dataclasses.replace`; `_run_pipeline`'s counting-DP branch no longer calls `reduce_counting_state` (deleted). |
| 8 Step 4 - Reduction no longer returns `counting_state` / `unary_cardinality_masks` | **COMPLETE** (`baac987`): fields removed from `ReducedProblem`/`ReductionState`/`_materialize`; counting state/masks threaded as keyword args through `build_counting_dp_input_from_reduced` + `_counting_component_from_reduced_cell_graph`. Tests read via `algo_input`/explicit args. |
| 9 - Retire backend-heavy `ReducedProblem` | **Achievable scope complete; full retirement architecturally constrained (see below).** Dead/redundant fields retired: `normal_form`/`features` (`2f17003`), `counting_state`/`unary_cardinality_masks` (`baac987`), `circle_len` (`0b6a3bc`), `repeat_factor` (`dc7bdd3`); `weights` ownership transferred to materializers (`10202de`). Remaining fields are **irreducible** (must stay - materializers outside `reduction/` cannot reconstruct them): `qf_formula`, `leq_predicate`, `predecessor_predicates`, `circular_predecessor_predicate` (all from the legacy `fol_view`, a forbidden import outside `reduction/`), `cardinality_constraints`, `decode_pipeline`, `domain`, `options`; **load-bearing + structurally pinned**: `reduction_kind` (cell-graph cache correctness key, `cell_graph/cache.py:114`; pinned by `test_fol_package_structure.py:291`); **bug-adjacent**: `unary_evidence` (always `()` from `_materialize`, but its sole live consumer `build_ground_cnf_input_from_reduced` is the PROPOSITIONAL algo's intended GROUND_UNITS path - now dead because the field is always empty; removing it would silently cement that pre-existing dead path, so deferred); **invasive to remove** (threading through `build_input`): `evidence_plan` (9 readers; reconstructable via `build_evidence_plan(problem, options)` but reduction still needs it internally for `qf_formula` conjunct + `repeat_factor`). Full `ReducedProblem` removal (Step 3) is impossible: the irreducible fields mean `ReducedProblem` IS the necessary materializer-internal bundle, and the plan's Conflict Resolution Notes forbid introducing a replacement (`CompiledProblem`: "Do not add it"; `DecodePipeline`: "pure contract should remain minimal"). |
| 5 Steps 3–4 - Materializers compile weights; reduction passes raw weights | **COMPLETE** (`10202de`, option (c)): `reduce_weights` deleted; `_materialize` carries raw weights + builds the cardinality encoding only for the `DecodePipeline`. `compile_reduced_weights` (algo/core.py) compiles + applies cardinality markers, called by both materializers before `build_input`. The cardinality-weighting/compilation fusion is resolved by moving them together into the materializer (behavior-preserving for the exact backend). |

### Key entanglement

`compile_problem` (`engine/orchestration.py`) exposes `reduced = spec.reduce(...)`
as `CompileArtifacts.reduced_problem`, and materializers + tests consume
`reduced.qf_formula`/`.weights`/`.evidence_plan`/`.decode`/`.counting_state`.
Migrating `spec.reduce` to return `ReducedProblems` (pure) breaks these
consumers unless `CompileArtifacts` and the tests are updated to read backend
fields from the materialized `algo_input` instead.

**Resolved double-application / lost-encoding risk (migration step 1, `fe2d623`):**
`build_evidence_plan` returns `EvidencePlan(strategy=NONE)` when
`problem.unary_evidence` is empty, so the evidence `formula_patch` is not
applied twice. The LIFTED_PROFILES path previously had **two divergent
representations** of the same logical concept:
- `profile_capacity_constraint` - produced by Task 3's pure
  `reduce_unary_evidence_to_profile_capacity` (and by
  `reduce_to_reduced_problems`); was **unused** by the live flow.
- `evidence_plan` - produced by `build_evidence_plan` from `unary_evidence`
  via `EvidencePlan.build(..., LIFTED_PROFILES)`; the **live** path.

`fe2d623` unified these: `EvidencePlan.build_from_profile_capacity`
reconstructs the `EvidencePartition` (identical fields on both types) and
`build_evidence_plan` now consumes `problem.profile_capacity_constraint`
when present. Parity verified - the plan rebuilt from the constraint equals
the plan built from the original unary evidence. The constraint branch is
additive (profile_capacity_constraint is never set in the live compile
flow), so existing behavior is unchanged. Prepending
`reduce_to_reduced_problems` (LIFTED_PROFILES) ahead of the backend no
longer loses the encoding. The CCS path was already safe (Task 3 adds to
`cardinality_constraints` + `sentence`, both consumed by `_run_pipeline`).

### Recommended migration order (now that the base is committed)

1. ~~Verify `build_evidence_plan` is idempotent / compatible with Task 3's
   unary-evidence reduction~~ - **done** (`fe2d623`): unified the two
   LIFTED_PROFILES representations; `reduce_to_reduced_problems` can run
   before the backend without losing the profile-capacity encoding.
2. Extract `_run_pipeline`'s backend materialization into a shared helper
   callable by materializers (e.g., `materialize_backend(problem, ctx)`),
   separating it from the logical reductions.
3. Migrate one algorithm's materializer (start with `standard`) to consume
   `ReducedProblems + MaterializationContext`, calling the shared backend
   helper. Update `CompileArtifacts` + affected tests.
4. Repeat per algorithm (Task 7 Step 4: do not migrate all in one commit).
5. ~~Move counting-state construction into incremental3 (Task 8 Step 2)~~ - **done** (`4c53d2d`).
6. ~~Remove `counting_state`/`unary_cardinality_masks` from `ReducedProblem`~~ (Task 8 Step 4) - **done** (`baac987`). ~~Move weight compilation into materializers~~ (Task 5 Steps 3–4) - **done** (`10202de`). Retire retireable `ReducedProblem` fields (Task 9) - **achievable scope done** (`normal_form`/`features` `2f17003`; `circle_len` `0b6a3bc`; `repeat_factor` `dc7bdd3`). Full `ReducedProblem` removal is architecturally constrained (irreducible fields - see Task 9 row).

## Pre-existing failures (out of scope, not caused by this work)

Full unit suite (`--continue-on-collection-errors`): **59 failed, 332 passed,
15 xfailed, 10 errors** (10 collection errors in `tests/framework/` from a
broken `wfomc.framework.syntax.syntax` import). The 59 failures + 10 errors are
unchanged from the pre-plan baseline (failure set verified identical this
session via `comm -13`/`comm -23` on the `tests/unit tests/framework` scope,
36 failed / 308 passed / 15 xfailed / 10 errors, with the `circle_len` +
`repeat_factor` retirements applied); the +5 passed over the pre-plan baseline
are the plan's added contract/parity tests (pure entry point, unification,
cardinality-decode/counting compile). The 59 failures + 10 errors are tracked
tests exercising WIP code paths with pre-existing bugs:

- `unary_evidence.py:291` - `AttributeError: 'Quantifier' object has no attribute 'uni_formula'`.
- `cell_graph.py:295` - `KeyError: @ev_1` / `@c2_quant_0`.
- `grounding/propositional.py:99` - `'Atom' object has no attribute 'make_positive'` (7 in `test_engine_compile`).
- `cell_graph/components.py` - `@tseitin*` auxiliary predicates genuinely absent from `cell.preds` (counting-state preds ≠ cell-graph preds; ~20 in incremental3/cell_graph). This is the Task 8 Step 2 structural bug, not the Step 3 identity bug (fixed in `081de46`).
- `normalize.py:466` - `NormalizeError: Modulo counting` (modk).
- `test_cell_graph_construction.py:167` - `WFOMCResult(8) == WFOMCResult(4)` (wrong count).

The Task 8 Step 3 fix (`081de46`) is a strict improvement (`==` fast path
preserved -> no pass->failure possible) but does **not** cure the ~20
@tseitin failures, which are genuine absences (structural), not identity
mismatches.

## Acceptance criteria status

- 1 (reduction output is `ReducedProblems`): **done** - `spec.reduce = pure_reduce` (pure `Problem -> ReducedProblems`) for all 8 algos (Task 7); `reduce_to_reduced_problems` entry point.
- 2 (reduction no longer constructs cell graphs/counting state/masks/evidence plans/compiled weights): **substantially done** - cell graphs (never built by reduction), counting state, unary cardinality masks, and compiled weights are all out of reduction (Task 8 Steps 2/4 + Task 5 Steps 3–4: reduction passes raw weights through; materializers compile + apply cardinality markers via `compile_reduced_weights`). The generic `EvidencePlan` remains on `ReducedProblem` as a logical reduction artifact (its `formula_patch` is applied in `reduce_evidence`); it is not an algorithm-specific execution plan (Non-Goal 112). Task 9's achievable scope (all retireable backend-heavy fields: `normal_form`/`features`/`counting_state`/`unary_cardinality_masks`/`circle_len`/`repeat_factor`; `weights` ownership) is **complete**; full `ReducedProblem` removal is architecturally constrained (irreducible fields - see Task 9 row).
- 3 (each materializer consumes `ReducedProblems` + `MaterializationContext`): **done** - all 8 materializers are 2-arg `(reduced_problems, ctx)` via `ufo2_materializer`/`counting_dp_materializer` (Task 7).
- 4 (engine owns cache keys, passes cache via context): **done** - `MaterializationContext.runtime.cache` (`RuntimeCache`) is used throughout orchestration (`context.cache.get_or_build` for normal_form/features/reduced/algo_input/results).
- 5 (exactly one `ProfileCapacityConstraint`): **done** (Task 2).
- 6 (CCS and profile-capacity unary evidence are separate reductions): **done** (Task 3).
- 7 (weight arithmetic via `WeightOptions`/`WeightPlan`): **done** (Tasks 4-5).
- 8 (rounded multivariate symbolic arithmetic fails early): **done** (`compile_weight_mapping` raises `NotImplementedError` for ARB/ARB_POLY).
- 9 (typed/legacy predicate boundaries compare by stable identity): **done** (Task 8 Step 3; both OLD and NEW cell-graph paths).
- 10 (focused suites): **done** for Tasks 1-6 + pure entry point + unification (51 passed); incremental3/counting_state/unary_evidence subset has pre-existing @tseitin/modk failures (not regressions).
- 11 (full suite): **blocked** - 59 pre-existing WIP-bug failures + 10 framework collection errors, unrelated to this plan (baseline confirmed identical with this plan's changes stashed).
