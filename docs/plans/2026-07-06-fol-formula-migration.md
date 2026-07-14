# FOL Formula Migration Plan

Status: active (Task 3 partial — cell-graph analysis migrated)
Date: 2026-07-06

## Goal

Make typed `wfomc.fol.formulas.Formula` the production logical IR from parser
through reduction and algorithm input materialization. Remove the production
dependency on legacy `SC2` / `QFFormula` / `QuantifiedFormula`.

## Progress (2026-07-06)

- **Task 1 done.** `fol/qf.py` exposes `Literal(atom, positive)` with `.pred`,
  `.args`, `.positive`, `__invert__`, `make_positive()`, `substitute()`; plus
  `model_literals`, `qf_simplify`, `is_satisfiable`, `to_cnf_clauses`.
- **Task 2 done.** `cell_graph/formula_ops.py` is the sole formula-execution
  layer for CellGraph. It is polymorphic: typed `Formula` primary path, legacy
  `QFFormula` dispatch (converting to typed `Literal`/`Atom` at model/atom
  boundaries) so CellGraph is engine-agnostic while `qf_formula` is still
  legacy. Helpers: `qf_preds`, `qf_atoms`, `qf_vars`, `qf_nullary_atoms`,
  `qf_make_atom`, `qf_ground_on_tuple`, `qf_sub_nullary`, `qf_satisfiable`,
  `qf_model_literals`, `qf_literal_universe`, `qf_exactly_one`.
- **Task 3 partial.** `cell_graph/cell_graph.py` no longer calls legacy
  `.models()/.atoms()/.vars()/.substitute()/.sub_nullary_atoms()/.simplify()/
  .satisfiable()` directly — all routed through `formula_ops`.
  `cell_graph/components.py` (`Cell.get_evidences`, `TwoTable`) and
  `cell_graph/utils.py` (`conditional_on`) now use typed `Literal`.
  Construction (`pred(c,c)`, `top`, `&`, `exactly_one_qf`, `Pred(...)`) is still
  legacy because `qf_formula` is still legacy — migrating it requires Task 4.
  **Result: 240→249 passing (9 fixed, 0 regressions); E2E counts correct and
  consistent across standard/fast/fastv2 (330626, 1024, 9440640).**
- **Task 5 started (additive).** `wfomc/fol/__init__.py` exports `Literal` and
  `positive_atom` from `wfomc.fol.qf` alongside the legacy surface. Verified
  `from wfomc.fol import Literal, positive_atom` works; no regressions
  (baseline still 34 failed / 257 passed / 15 xfailed, excluding
  `tests/unary_evidence/` pre-existing FAST hangs).
- **Task 4 diagnosis complete (no code change).** Precisely identified why the
  `qf_model_literals` `yield from` fix (correct, prerequisite) cannot land
  alone: the cell-graph formula is a mixed typed+legacy `And` because
  `reduced.qf_formula` is legacy while `cell_formulas` are typed, and
  `CellGraph.__init__` re-mixes via legacy `leq_pred`/`predecessor_preds`/
  tautology conjuncts. See "Diagnosis (2026-07-06 session)" below.
- **Task 4 partial (production code landed).** `cell_graph/formula_ops.py`
  now makes the typed IR's model enumeration robust to mixed typed+legacy
  formulas — the documented prerequisite for the flip:
  - `qf_model_literals`: fixed the generator-delegation bug (`yield from
    model_literals(...)` instead of `return ...`) on the pure-typed branch,
    AND added a mixed-formula branch that projects to legacy via
    `_typed_formula_to_legacy_formula` and enumerates via `.models()` when the
    typed formula has legacy children (`_is_pure_typed` helper added).
  - `qf_ground_on_tuple`: added the same mixed→legacy routing, because the
    typed `_substitute` cannot ground legacy children (the tautology
    conjuncts `pred(X) | ~pred(X)` would keep free var `X`).
  - **Result: 2 tests fixed** (`test_incremental_solve_uses_reduced_ordered_
    input_without_wfomc_context[linear_order_perm_pred]`,
    `test_ordered_cell_graph_input_keeps_order_metadata`); **3 pre-existing
    latent failures exposed** (`test_incremental3_corpus_reduces_natively`
    for `books-arragement`/`BA_CC`/`linear_order_perm`) — these were
    false-passing because the `return`-yields-nothing bug gave 0 cells,
    skipping the counting-DP path; correct enumeration now hits the
    pre-existing `Cell.is_positive` Tseitin-mismatch bug (`@tseitin2` not in
    `cell.preds`, which has only `@tseitin0`/`@tseitin1`) — a solver-kernel
    issue, out of scope (Non-Goals). Suite: 34→35 (net +1: +3 exposed, −2
    fixed). `reduced.qf_formula` is still legacy (flip blocked, see below).

## Current Blocker

Task 4 (typed `qf_formula` end-to-end) is a coupled multi-file migration:
flipping `qf_formula` to typed requires, in one atomic change,

- `reduction/core.py` `_run_pipeline`/`_materialize` produce typed
  `qf_formula` (`sections.universal_body` + typed evidence/counting/skolem
  patches) and drop the three `_typed_formula_to_legacy_formula` bridges and
  `build_legacy_fol_reduction_view` from the production path;
- `cell_graph/cell_graph.py` construction migrated to typed
  (`qf_make_atom`, `qf_top`, `qf_exactly_one`, typed `a/b/c/X`) so it no longer
  imports legacy `Pred/top/exactly_one_qf`;
- `unary_evidence.py` migrated from legacy `AtomicFormula` to typed `Literal`
  so `EvidenceProfile.evidence` and `_cell_formulas_from_evidence_partition`
  produce typed formulas;
- `grounding/propositional.py` and `algo/cell_graph/inputs.py` adapted to typed
  `Atom`/`Literal` (`.predicate`/`.terms`/`.positive`) for the propositional
  and treewidth paths.

Open risk: the legacy `qf_formula` carries Tseitin `extra_equiv` for ext/cnt
sections (`legacy_adapter._tseitin_transform_sectioned`) that the typed
`universal_body` lacks. The typed `reduce_skolem`/`reduce_counting` patches may
already make this redundant (direct Skolemisation vs Tseitin+Skolem); this must
be verified empirically by flipping `qf_formula` to typed and confirming the
cell-graph corpus counts are unchanged.

### Diagnosis (2026-07-06 session) — precise blocker for `yield from`

`qf_model_literals` (formula_ops.py:222) has a latent generator-delegation bug
on its typed branch:

```python
def qf_model_literals(formula, ...):
    if _legacy(formula):
        for model in formula.models():
            yield frozenset(_legacy_lit_to_typed(lit) for lit in model)
        return
    return model_literals(formula, atom_universe, max_models=max_models)  # BUG
```

`qf_model_literals` is a generator (it contains `yield`), so `return <gen>`
does NOT delegate iteration — the typed branch yields nothing. The correct fix
is `yield from model_literals(...)`. This is a prerequisite for Task 4: without
it, any typed `qf_formula` produces zero models → empty cell graph.

**Why `yield from` cannot land alone (bisected: +7 net failures, 41 vs 34).**

`yield from` fixes 1 test (`linear_order_perm_pred` — was getting 0 models) but
breaks 8 (4 `test_engine_compile`/`test_cell_graph_*` compile tests + 4
`test_incremental3_counting_native` corpus tests). Root cause, confirmed by
instrumentation: in the current baseline `reduced.qf_formula` is **legacy**
(`fol_view.qf_formula` from `_qf_universal_body`), so `gnd_formula_cc` is
legacy. But the `cell_formulas` passed to `CellGraph` are **typed**. In
`_build_cells` (cell_graph.py:287-294):

```python
gnd_formula = self.gnd_formula_cc                       # legacy
if cell_formula is not top:
    gnd_formula = gnd_formula & self._ground_on_tuple(cell_formula, c)  # typed
for model in qf_model_literals(gnd_formula):             # mixed!
```

`legacy & typed` dispatches to the typed `Formula.__and__`, producing a mixed
typed `And` whose children are `[Atom, QFFormula, Not]` — a raw legacy
`QFFormula` sits as a child of the typed `And`. The typed `model_literals`
enumerator (`fol/qf.py:_collect`) only recurses into
`Atom/Not/And/Or/Implies/Iff`; it does **not** recurse into legacy
`QFFormula`/`LegacyFormula`/`Quantifier` children (unlike `fol/analysis.py
:_formula_children`, which does). So `_atoms_list(formula)` returns an
incomplete atom set (e.g. 2 of 8 preds), `model_literals` yields partial
models, and `_build_cells` raises `KeyError: p_table_PRED` at
`Cell(tuple(code[p] for p in self.preds), self.preds)`.

With the buggy `return` (yield nothing), these 8 tests pass only because the
empty cell graph happens to satisfy their structural assertions.

**Implication for Task 4.** The full flip (typed `reduced.qf_formula` +
`yield from`) is the correct end state, but it is not sufficient on its own:
`CellGraph.__init__` re-mixes the formula by conjoining many legacy terms —
`pred(X) | ~pred(X)` tautologies for `required_unary_preds` (lines 86-87),
`self.leq_pred(c, c)` / `~self.leq_pred(a, b)` (lines 110-114), and
`~predecessor_preds(...)` reductions (lines 116-144). All of `leq_pred`,
`predecessor_preds`, `required_unary_preds`, `X`, `c`, `a`, `b`, `top` are
legacy in this file. A typed `reduced.qf_formula` `&`-ed with these legacy
conjuncts becomes mixed again → same `KeyError`. This is the source of the 31
regressions seen in the prior full-flip attempt.

**Therefore Task 4 must land as one atomic change:** (1) `yield from` fix;
(2) typed `reduced.qf_formula` (remove the 3 bridges +
`build_legacy_fol_reduction_view`); (3) typed `CellGraph` construction
(typed `leq_pred`/`predecessor_preds`/`required_unary_preds`/`X`/`c`/`a`/`b`/
`top`, via `qf_make_atom`/`qf_top`/`qf_exactly_one`); (4) typed `cell_formulas`
from `unary_evidence.py`. Only then is the cell-graph formula purely typed and
`model_literals` complete. Attempting any subset re-introduces the mixed
formula.

A lower-risk unblock (NOT yet implemented): extend `_collect`/`_evaluate` in
`fol/qf.py` to recurse into legacy `QFFormula`/`LegacyFormula` children (using
`_formula_children` and `qf_atoms` for collection, and a legacy-aware
evaluator for truth-table checks). This would make `model_literals` robust to
mixed formulas and let `yield from` land independently of the full flip.

### Diagnosis (2026-07-06 session) — flip attempt: `universal_body` is not QF

With the mixed-formula robustness above landed, the full flip (typed
`reduced.qf_formula` + remove 3 bridges) was attempted: `core.py` sets
`qf_formula=sections.universal_body` (typed) and `reduce_evidence`/
`reduce_counting`/`reduce_skolem` use typed patches directly. This was
**reverted** — it regressed 19 tests in the narrow set with `KeyError: P`/
`H`/`Q` at `_build_cells`. Root cause, confirmed by instrumentation:

`build_typed_reduction_view` (fol_view.py:42-45) strips only a *leading*
`∀∀` chain:

```python
qf = normal_form.universal
while isinstance(qf, Quantifier) and qf.kind == QuantifierKind.FORALL:
    qf = qf.body
```

For a problem whose universal section is a **conjunction of separately
quantified formulas** — `And(∀X.P(X), ∀Y.Q(Y))` (typed) — the top-level node
is `And`, not `Quantifier`, so the loop strips nothing and
`sections.universal_body` is returned **still containing `Quantifier`
children**. The legacy `_qf_universal_body` (legacy_adapter.py:523-527) does
not have this problem because it walks `.quantified_formula` to a `QFFormula`,
and legacy `.atoms()`/`.models()` ignore quantifier structure (treating
`P(X)`/`Q(Y)` as plain atoms with bound vars).

The typed enumerator (`fol/qf.py:_collect`/`_evaluate`) does **not** recurse
into `Quantifier`/`CountingQuantifier` nodes — so `_atoms_list` returns `[]`,
`model_literals` yields one empty model (`frozenset()`), `_build_cells`
builds `code = {}`, and `Cell(tuple(code[p] for p in self.preds), ...)`
raises `KeyError: P` (for every pred in `self.preds`).

**Two coupled fixes required to unblock the flip** (not yet implemented):

1. **Make `universal_body` actually QF.** Either (a) fix
   `build_typed_reduction_view` to strip/distribute quantifiers from a
   conjunction-of-quantifiers (e.g. recurse into `And`/`Or` args and strip
   leading quantifiers from each), or (b) make `_collect`/`_evaluate` in
   `fol/qf.py` recurse into `Quantifier`/`CountingQuantifier` bodies (mirroring
   legacy `.atoms()` which treats `P(X)` inside `∀X.P(X)` as a collectable
   atom). Option (b) is lower-risk but requires matching legacy `.models()`
   semantics for quantified sub-formulas exactly (does `∀X.P(X)` evaluate as
   `P(X)`'s truth value, or as opaque-True?) — verify against legacy cell
   counts before landing.
2. **Migrate `CellGraph.__init__` construction to typed** (tautologies,
   `leq_pred`, `predecessor_preds`, `X`/`c`/`a`/`b`/`top`) so the formula
   stays purely typed and doesn't rely on the mixed→legacy routing. (The
   routing added in `formula_ops.py` this session makes this less urgent —
   mixed formulas now enumerate correctly — but a purely-typed formula is the
   end goal.)

The flip is therefore still **not started** in production; `core.py` is at
baseline (legacy `qf_formula` + 3 bridges). The mixed-formula routing in
`formula_ops.py` is the concrete progress landed this session.

## Implementation Plan

### Task 1: Finish Typed QF Literal and Model API  — done

### Task 2: Add Cell-Graph Formula Adapter  — done

### Task 3: Migrate CellGraph Without Rewriting Its Algorithm  — partial

Analysis layer migrated (see Progress). Remaining: construction migration,
blocked on Task 4.

### Task 4: Remove Reduction Bridge  — not started

After typed `CellGraph` construction passes tests:

- make `ReducedProblem.qf_formula` typed end to end;
- stop calling `_typed_formula_to_legacy_formula` in `reduction/core.py`;
- remove `build_legacy_fol_reduction_view` from the production path;
- migrate `unary_evidence.py`, `grounding/propositional.py`,
  `algo/cell_graph/inputs.py` to typed;
- delete `reduction.legacy_adapter` if only compatibility tests remain.

### Task 5: Clean Public FOL Surface  — started (additive exports only)

Added `Literal`/`positive_atom` to `wfomc.fol` (no legacy removal yet). Once
production no longer imports legacy FOL modules:

- remove legacy exports from `wfomc.fol.__all__`;
- move any unavoidable old names behind `wfomc.compat`;
- delete legacy-only tests or mark them compatibility-only.
- Split `cell_graph/cell_graph.py` (≈890 lines) into core/optimized/factory
  submodules to satisfy the 200–400 line rule.

## Verification

Run the narrow checks first:

```bash
uv run pytest tests/unit/test_fol_syntax.py tests/unit/test_formula_models.py -q
uv run pytest tests/unit/test_cell_graph_construction.py tests/unit/test_cell_graph_inputs.py -q
uv run pytest tests/unit/test_reduction_backbone.py tests/unit/test_engine_compile.py -q
```

Then run:

```bash
uv run ruff check src/wfomc tests/unit
uv run pytest -q
```

Baseline: 41 pre-existing failures. After Task 3 partial: 32 failures
(9 fixed, 0 regressions).

## Non-Goals

- Do not rewrite standard/fast/incremental solver kernels.
- Do not change materialized numeric table shapes.
- Do not recreate a fake `QFFormula` wrapper around typed `Formula`.
- Do not reintroduce global mutable predicate registries.
