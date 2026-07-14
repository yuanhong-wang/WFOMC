# Typed FOL Reduction Migration Plan

> Date: 2026-07-07 | Status: active, partial migration in progress

## Goal

Make the production path use the typed `wfomc.fol` model end-to-end:

```text
Problem -> normal_form -> reduction -> algo materialization
```

Reduction must stop converting typed formulas to legacy `SC2`, `QFFormula`, or
`AtomicFormula`. Legacy FOL modules may remain under `wfomc.compat` and
`wfomc.fol.legacy_*`, but production code should not depend on them.

## Current State

Already moved toward typed FOL:

- Parser output is typed `Formula`.
- C2 normal-form input is typed.
- Counting, skolem, and evidence formula patches are typed.
- `fol.qf.Literal` exists as the signed typed-literal shape for cell tables.
- `cell_graph.formula_ops` exists as a transition adapter for formula execution.
- `reduction.fol_view` exists, but production reduction still uses legacy data.

Still transitional:

- `reduction.core` builds `TypedReductionSections`, but still calls
  `build_legacy_fol_reduction_view` for baseline `qf_formula`, order metadata,
  weights, and counting-DP state.
- `reduction.counting_state` still consumes legacy `QuantifiedFormula` /
  `AtomicFormula`.
- `grounding.propositional` still imports `native_boolean` and
  `fol.legacy_syntax`.
- `unary_evidence.py` is still a legacy-oriented public module.
- `wfomc.fol.__init__` still exports legacy names for compatibility.

## Target Architecture

### Typed FOL Is The Source IR

Use these modules from production code:

- `wfomc.fol.syntax` for public constructors and symbols.
- `wfomc.fol.formulas` for node classes.
- `wfomc.fol.analysis` for predicates, atoms, variables, constants.
- `wfomc.fol.rewrite` for substitution and structural rewrites.
- `wfomc.fol.qf` for typed literals, satisfiability, CNF, and model
  enumeration.

Do not use package-level legacy names such as:

- `from wfomc.fol import QFFormula`
- `from wfomc.fol import AtomicFormula`
- `from wfomc.fol import Pred, Const, X, top`
- `wfomc.fol.native_boolean`

### Reduction Owns Typed Section Materialization

`reduction.fol_view.build_typed_reduction_view(normal_form, problem)` should be
the only reduction entry point for extracting:

- universal quantifier-free body;
- existential sections;
- counting sections;
- order metadata;
- domain;
- weights;
- unary evidence view.

`ReductionState` should contain typed fields only:

- `sections`
- `domain`
- `order_metadata`
- `qf_formula`
- `weights`
- accumulated plans/states

`reduction.core` should not import `reduction.legacy_adapter`.

### CellGraph Depends On Formula Ops, Not Legacy FOL

`CellGraph` should treat formula execution as a service from
`cell_graph.formula_ops`:

- `qf_preds`
- `qf_atoms`
- `qf_ground_on_tuple`
- `qf_sub_nullary`
- `qf_satisfiable`
- `qf_model_literals`
- `qf_literal_universe`

`Cell`, `TwoTable`, and materialized algorithm table shapes should remain
stable. Internally they should carry `fol.qf.Literal`, not legacy
`AtomicFormula`.

### Compatibility Is Explicit

Legacy code should live only in:

- `wfomc.compat.*`
- `wfomc.fol.legacy_*`
- temporary compatibility tests

The root package `wfomc.fol` may lazily export legacy names during transition,
but production modules should import typed submodules directly.

## Implementation Steps

1. Finish typed QF behavior.
   - Keep `models(formula) -> Iterator[dict[Atom, bool]]` for existing tests.
   - Make `model_literals(formula)` produce `frozenset[Literal]`.
   - Replace brute-force enumeration with PySAT blocking-clause enumeration
     when possible; block only original atom variables.
   - Extend simplification enough for nullary substitution and ground equality.

2. Make `cell_graph.formula_ops` typed-only.
   - Remove fallback paths that import `reduction.legacy_adapter` or project
     typed formulas back to legacy.
   - Replace legacy profile selector helpers with typed equivalents.
   - Stop importing `QFFormula`, `Pred`, `Const`, `X`, `top`, and
     `exactly_one_qf` from `wfomc.fol` in `cell_graph/cell_graph.py`.

3. Complete `reduction.fol_view`.
   - Extract order metadata from all typed normal-form sections, not only the
     universal body.
   - Provide typed domain and typed weights directly from `Problem`.
   - Provide any unary evidence needed by materializers without legacy atoms.
   - Ensure `circle_len` and decode metadata are derived from typed fields.

4. Remove legacy adapter from `reduction.core`.
   - Initialize `ReductionState.qf_formula` from
     `sections.universal_body or true()`.
   - Use typed `order_metadata`, typed `domain`, and typed `weights`.
   - Build `EvidencePlan` from `Problem` and typed metadata; do not pass a
     legacy FOL view.
   - Delete `_typed_formula_to_legacy_formula` calls from the reduction path.

5. Port counting-DP state extraction.
   - Change `build_counting_state` to consume `TypedReductionSections`.
   - For each count section, read `comparator`, `count`, `body`,
     `outer_var`, and `counted_var` directly.
   - Accept typed `Atom` bodies of arity 1 or 2.
   - Build `binary_evidence` from typed `Literal` values.
   - Preserve `CountingState` field shapes expected by incremental3.

6. Port propositional grounding.
   - Make `_collect_clauses(formula, domain)` consume typed `Formula`.
   - Ground with `fol.rewrite.substitute`.
   - Convert typed formula to CNF structurally or through `fol.qf`.
   - Use typed `Atom` keys in `atom_to_id` and typed predicates in
     `id_to_predicate`.
   - Remove `native_boolean`, `AtomicFormula`, `Const`, `Pred`, `X`, and `Y`
     imports from `grounding.propositional`.

7. Retire production legacy imports.
   - Keep `reduction.legacy_adapter` only for `wfomc.compat.parser` and
     compatibility tests.
   - Update package-structure tests so production modules fail on
     `legacy_syntax`, `legacy_sc2`, `legacy_utils`, `native_boolean`, and
     package-level legacy `wfomc.fol` imports.

## Test Plan

Run narrow tests after each subsystem:

```bash
uv run pytest tests/unit/test_qf_boolean.py tests/test_formula_models.py -q
uv run pytest tests/unit/test_cell_graph_construction.py tests/unit/test_cell_graph_inputs.py -q
uv run pytest tests/unit/test_reduction.py tests/unit/test_counting_reduction.py tests/unit/test_engine_compile.py -q
uv run pytest tests/unit/test_problem_parser.py tests/unit/test_normal_form.py -q
uv run pytest tests/unit/test_package_structure.py -q
```

Then run broader regressions:

```bash
uv run pytest tests/test_incremental3_regressions.py tests/unary_evidence -q
uv run pytest -q
uv run ruff check src/wfomc tests/unit
```

Acceptance criteria:

- No production module outside `wfomc.compat` and `wfomc.fol.legacy_*` imports
  legacy FOL modules.
- `reduce_to_ufo2` and `reduce_to_counting_dp` construct typed
  `ReducedProblem.qf_formula`.
- Standard, fast, incremental, incremental3, tail-signature, and propositional
  materializers consume typed formulas or typed literals.
- Modulo counting examples such as `models/modk/0mod2-regular-graph.wfomcs`
  compile for incremental3.

## Non-Goals

- Do not rewrite solver kernels.
- Do not change materialized numeric table shapes.
- Do not delete `wfomc.compat` in this migration.
- Do not recreate a fake typed `QFFormula` wrapper.
- Do not reintroduce global mutable predicate registries.
