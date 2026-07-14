# Reduction Compat Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move all legacy-specific reduction code under `src/wfomc/reduction/compat/`, so the main `wfomc.reduction` package becomes typed-first and its legacy boundary is explicit.

**Architecture:** Keep `wfomc.reduction` as the public typed reduction pipeline. Create `wfomc.reduction.compat` for legacy view construction, legacy evidence/cardinality adapters, and the legacy sentence-to-counting-state extractor. The main pipeline may call compat only through narrow adapter functions, and structure tests should enforce that no legacy FOL imports appear in typed reduction modules.

**Tech Stack:** Python 3.11, dataclasses, pytest, typed `wfomc.fol` API, existing `wfomc.fol.compat` legacy bridge, existing `wfomc.reduction` pipeline.

---

## Current State

The `fol` package is now typed-first:

- External callers use `from wfomc.fol import ...`.
- Legacy FOL code lives under `src/wfomc/fol/compat/`.
- `wfomc.fol.__init__` exports the public typed DSL.

`reduction` is not yet as clean. The main pipeline still builds a legacy FOL view in `core.py`:

```python
from wfomc.fol.compat.legacy_adapter import build_legacy_fol_reduction_view

sections = build_typed_reduction_view(normal_form, problem)
fol_view = build_legacy_fol_reduction_view(problem)

state = ReductionState(
    ...
    order_metadata=fol_view.order_metadata,
    qf_formula=fol_view.qf_formula,
    weights=dict(fol_view.weights),
    evidence_plan=build_evidence_plan(problem, fol_view, options),
)
```

Legacy-specific code is currently mixed into these files:

- `src/wfomc/reduction/legacy_adapter.py`
  - Deprecated shim that re-exports `wfomc.fol.compat.legacy_adapter`.
- `src/wfomc/reduction/core.py`
  - Imports `build_legacy_fol_reduction_view`.
  - Uses legacy `fol_view.qf_formula`, `fol_view.weights`, and `fol_view.order_metadata`.
  - Builds legacy transformed sentence for incremental3 counting state.
- `src/wfomc/reduction/evidence_planning.py`
  - Imports `unary_evidence_from_legacy`.
  - Imports `cardinality_constraints_from_legacy`.
  - Accepts `fol_view`.
- `src/wfomc/reduction/counting_state.py`
  - Public algorithm state and mask classes are useful.
  - But the extractor imports `AtomicFormula`, `QuantifiedFormula`, `Pred`, `a`, and `b` from legacy FOL.
- `src/wfomc/reduction/_accessors.py`
  - Contains mixed typed/legacy helpers such as `count_body_pred()`.

The migration should not immediately delete legacy behavior. It should isolate it in `reduction.compat`, then make the typed path consume typed data by default.

## Target Layout

```text
src/wfomc/reduction/
  __init__.py
  _accessors.py              # typed generic accessors only
  core.py                    # typed pipeline; no direct legacy FOL imports
  counting.py                # typed counting-to-UFO2 reduction
  counting_state.py          # public CountingState/UnaryCardinalityMasks data only
  decode.py
  evidence_planning.py       # typed evidence/cardinality planning only
  fol_view.py                # typed view from C2NormalForm + Problem
  normal_form.py
  plan.py
  skolem.py
  state.py
  compat/
    __init__.py
    legacy_view.py           # narrow wrapper around wfomc.fol.compat.legacy_adapter
    evidence.py              # legacy unary evidence/cardinality conversions
    counting_state.py        # legacy sentence -> CountingState extractor
```

After the migration:

- `src/wfomc/reduction/legacy_adapter.py` should not exist.
- `src/wfomc/reduction/core.py` should not import `wfomc.fol.compat.*`.
- `src/wfomc/reduction/evidence_planning.py` should not import legacy conversion helpers.
- `src/wfomc/reduction/counting_state.py` should not import legacy FOL types.
- Only `src/wfomc/reduction/compat/*.py` should import `wfomc.fol.compat.*` or legacy FOL names.

## Compatibility Rule

The allowed import graph should be:

```text
wfomc.reduction.core
  -> wfomc.reduction.fol_view
  -> wfomc.reduction.evidence_planning
  -> wfomc.reduction.counting_state
  -> wfomc.reduction.compat.* only at explicit fallback points

wfomc.reduction.compat.*
  -> wfomc.fol.compat.*
  -> legacy FOL implementation
```

Do not allow this:

```text
wfomc.reduction.core -> wfomc.fol.compat.legacy_adapter
wfomc.reduction.evidence_planning -> unary_evidence_from_legacy
wfomc.reduction.counting_state -> AtomicFormula / QuantifiedFormula / Pred
```

---

## Task 1: Add Reduction Compat Boundary Tests

**Files:**
- Modify: `tests/unit/test_fol_package_structure.py`

**Step 1: Add a test that declares the target compat package**

Add this near the existing FOL/reduction package-structure tests:

```python
def test_reduction_legacy_code_lives_under_reduction_compat():
    assert Path("src/wfomc/reduction/compat").is_dir()
    assert Path("src/wfomc/reduction/compat/__init__.py").exists()
    assert Path("src/wfomc/reduction/compat/legacy_view.py").exists()
    assert Path("src/wfomc/reduction/compat/evidence.py").exists()
    assert Path("src/wfomc/reduction/compat/counting_state.py").exists()
    assert not Path("src/wfomc/reduction/legacy_adapter.py").exists()
```

**Step 2: Add a test that forbids direct legacy imports in typed reduction modules**

```python
def test_typed_reduction_modules_do_not_import_legacy_fol():
    typed_modules = [
        Path("src/wfomc/reduction/_accessors.py"),
        Path("src/wfomc/reduction/core.py"),
        Path("src/wfomc/reduction/counting.py"),
        Path("src/wfomc/reduction/counting_state.py"),
        Path("src/wfomc/reduction/evidence_planning.py"),
        Path("src/wfomc/reduction/fol_view.py"),
        Path("src/wfomc/reduction/normal_form.py"),
        Path("src/wfomc/reduction/skolem.py"),
        Path("src/wfomc/reduction/state.py"),
    ]
    forbidden = [
        "wfomc.fol.compat",
        "wfomc.fol.legacy_",
        "legacy_adapter",
        "AtomicFormula",
        "QuantifiedFormula",
        "unary_evidence_from_legacy",
        "cardinality_constraints_from_legacy",
    ]
    offenders = []
    for path in typed_modules:
        source = path.read_text()
        for needle in forbidden:
            if needle in source:
                offenders.append(f"{path}: {needle}")
    assert offenders == []
```

**Step 3: Add a test that only `reduction.compat` imports FOL compat**

```python
def test_only_reduction_compat_imports_fol_compat_inside_reduction():
    offenders = []
    allowed_prefix = Path("src/wfomc/reduction/compat")
    for path in sorted(Path("src/wfomc/reduction").rglob("*.py")):
        if allowed_prefix in path.parents:
            continue
        source = path.read_text()
        if "wfomc.fol.compat" in source:
            offenders.append(str(path))
    assert offenders == []
```

**Step 4: Update public module list**

In `PUBLIC_MODULES`, replace:

```python
"wfomc.reduction.legacy_adapter",
```

with:

```python
"wfomc.reduction.compat",
"wfomc.reduction.compat.legacy_view",
"wfomc.reduction.compat.evidence",
"wfomc.reduction.compat.counting_state",
```

**Step 5: Replace the old eager-load test**

Delete or rewrite `test_reduction_legacy_adapter_import_does_not_eager_load_sc2`.

Replacement:

```python
def test_reduction_compat_import_does_not_eager_load_legacy_sc2():
    saved_modules = {
        name: sys.modules[name]
        for name in tuple(sys.modules)
        if name == "wfomc.reduction.compat"
        or name.startswith("wfomc.reduction.compat.")
        or name.startswith("wfomc.fol.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    try:
        importlib.import_module("wfomc.reduction.compat")
        assert "wfomc.fol.compat.legacy_sc2" not in sys.modules
    finally:
        for name in tuple(sys.modules):
            if name.startswith("wfomc.reduction.compat.") or name.startswith("wfomc.fol."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
```

**Step 6: Run the new tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/test_fol_package_structure.py -q
```

Expected: FAIL because `reduction/compat` does not exist yet, `reduction/legacy_adapter.py` still exists, and typed modules still import legacy names.

---

## Task 2: Create `reduction.compat` Package

**Files:**
- Create: `src/wfomc/reduction/compat/__init__.py`
- Create: `src/wfomc/reduction/compat/legacy_view.py`

**Step 1: Create package initializer**

Create `src/wfomc/reduction/compat/__init__.py`:

```python
"""Compatibility adapters for legacy reduction inputs.

Typed reduction code should not import legacy FOL modules directly. Any
remaining bridge to legacy FOL, legacy evidence, or legacy counting-state
extraction lives under this package.
"""

from __future__ import annotations

__all__ = []
```

Keep it empty and non-eager. Do not import submodules here.

**Step 2: Create legacy view adapter**

Create `src/wfomc/reduction/compat/legacy_view.py`:

```python
"""Legacy FOL view adapter used by transitional reduction paths."""

from __future__ import annotations

from wfomc.fol.compat.legacy_adapter import (
    LegacyFolReductionView,
    build_legacy_fol_reduction_view,
)

__all__ = [
    "LegacyFolReductionView",
    "build_legacy_fol_reduction_view",
]
```

This is intentionally tiny. It gives reduction one local place to import legacy FOL view construction without reaching into `wfomc.fol.compat` from typed modules.

**Step 3: Run package tests**

Run:

```bash
uv run pytest tests/unit/test_fol_package_structure.py::test_reduction_compat_import_does_not_eager_load_legacy_sc2 -q
```

Expected: PASS. Importing `wfomc.reduction.compat` should not import `legacy_view`.

---

## Task 3: Move Legacy Evidence/Cardinality Conversion Behind Compat

**Files:**
- Create: `src/wfomc/reduction/compat/evidence.py`
- Modify: `src/wfomc/reduction/evidence_planning.py`
- Test: `tests/unit/test_fol_package_structure.py`

**Step 1: Create compat evidence module**

Create `src/wfomc/reduction/compat/evidence.py`:

```python
"""Legacy evidence/cardinality adapters for reduction."""

from __future__ import annotations

from wfomc.cardinality import CardinalityConstraints, cardinality_constraints_from_legacy
from wfomc.evidence import UnaryEvidence, unary_evidence_from_legacy


def unary_evidence_from_legacy_input(value: object) -> UnaryEvidence:
    return unary_evidence_from_legacy(value)


def cardinality_constraints_from_legacy_constraint(value: object) -> CardinalityConstraints:
    return cardinality_constraints_from_legacy(value)


__all__ = [
    "cardinality_constraints_from_legacy_constraint",
    "unary_evidence_from_legacy_input",
]
```

**Step 2: Split typed and compat evidence planning**

In `src/wfomc/reduction/evidence_planning.py`, remove these direct imports:

```python
cardinality_constraints_from_legacy,
unary_evidence_from_legacy,
```

Keep typed imports:

```python
from wfomc.cardinality import (
    CardinalityConstraints,
    cardinality_constraints_from_simple_constraints,
    combine_cardinality_constraints,
)
from wfomc.evidence import EvidencePlan, EvidenceStrategy, UnaryEvidence
```

Change `build_evidence_plan()` signature from:

```python
def build_evidence_plan(problem: "Problem", fol_view: object, options: "AlgoOptions") -> EvidencePlan:
```

to:

```python
def build_evidence_plan(problem: "Problem", options: "AlgoOptions") -> EvidencePlan:
```

Use typed evidence directly:

```python
evidence = getattr(problem, "unary_evidence", None)
is_empty = getattr(evidence, "is_empty", None)
if evidence is None or (is_empty is not None and is_empty) or not evidence:
    return EvidencePlan(strategy=EvidenceStrategy.NONE)

strategy = _options_evidence_strategy(options)
if strategy is None or strategy is EvidenceStrategy.NONE:
    return EvidencePlan(strategy=EvidenceStrategy.NONE)

if not isinstance(evidence, UnaryEvidence):
    from wfomc.reduction.compat.evidence import unary_evidence_from_legacy_input

    evidence = unary_evidence_from_legacy_input(evidence)

return EvidencePlan.build(
    evidence,
    frozenset(getattr(problem, "domain", ()) or ()),
    strategy,
)
```

This keeps the fallback explicit and local to one call site.

**Step 3: Update cardinality fallback**

Change:

```python
declared = cardinality_constraints_from_legacy(
    getattr(problem, "cardinality_constraint", None)
)
```

to:

```python
from wfomc.reduction.compat.evidence import cardinality_constraints_from_legacy_constraint

declared = cardinality_constraints_from_legacy_constraint(
    getattr(problem, "cardinality_constraint", None)
)
```

The typed module still has a fallback, but it no longer imports legacy conversion helpers at module import time.

**Step 4: Update `core.py` caller**

Change:

```python
evidence_plan=build_evidence_plan(problem, fol_view, options),
```

to:

```python
evidence_plan=build_evidence_plan(problem, options),
```

**Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/unit/test_fol_package_structure.py tests/unit/test_problem_parser.py tests/unary_evidence -q
```

Expected: package-structure test should pass for evidence import checks. Some unary evidence tests may still fail if they rely on legacy `fol_view.unary_evidence`; if so, add a transitional compat-only function:

```python
def build_evidence_plan_from_legacy_view(problem, fol_view, options) -> EvidencePlan:
    ...
```

and call it only from `reduction.compat` adapter code, not from the typed `core.py` path.

---

## Task 4: Split Public Counting State from Legacy Extractor

**Files:**
- Modify: `src/wfomc/reduction/counting_state.py`
- Create: `src/wfomc/reduction/compat/counting_state.py`
- Modify: `src/wfomc/reduction/core.py`
- Test: `tests/unit/test_incremental3_counting_native.py`
- Test: `tests/test_incremental3_regressions.py`

**Step 1: Keep public data classes in `counting_state.py`**

`src/wfomc/reduction/counting_state.py` should keep:

- `CountingState`
- `UnaryCardinalityMasks`

Change the type annotations that currently mention legacy `Pred` to generic `object`:

```python
@dataclass(frozen=True)
class CountingState:
    ext_preds: tuple[object, ...]
    cnt_preds: tuple[object, ...]
    cnt_params: tuple[int, ...]
    cnt_remainder: tuple[object, ...]
    exist_mod: bool
    mod_pred_index: tuple[int, ...]
    exist_le: bool
    le_index: tuple[int, ...]
    binary_evidence: tuple[frozenset, ...]
    c_type_shape: tuple[int, ...]
```

Update mask type annotations:

```python
self.mod_constraints: list[tuple[object, int, int]] = []
self.eq_constraints: list[tuple[object, int]] = []
self.le_constraints: list[tuple[object, int]] = []
```

Remove these imports from `counting_state.py`:

```python
from itertools import product
from wfomc.fol.compat.legacy_adapter import AtomicFormula, Formula, Pred, QuantifiedFormula, a, b
```

**Step 2: Move legacy extractor into compat**

Create `src/wfomc/reduction/compat/counting_state.py` with the old extractor code:

```python
"""Legacy sentence to incremental3 CountingState extraction."""

from __future__ import annotations

from itertools import product

from wfomc.fol.compat.legacy_adapter import (
    AtomicFormula,
    Formula,
    Pred,
    QuantifiedFormula,
    a,
    b,
)
from wfomc.reduction.counting_state import CountingState, UnaryCardinalityMasks
```

Move these definitions from `reduction/counting_state.py` into this file:

- `build_binary_evidence`
- `_counting_predicate`
- `_CountingExtractor`
- `build_counting_state`

Rename the public extractor to make the legacy dependency explicit:

```python
def build_counting_state_from_legacy_sentence(
    sentence: object,
) -> tuple[CountingState, UnaryCardinalityMasks]:
    ...
```

Set:

```python
__all__ = [
    "build_binary_evidence",
    "build_counting_state_from_legacy_sentence",
]
```

**Step 3: Add a deprecated lazy shim if needed**

If existing tests or public callers import `build_counting_state` from `wfomc.reduction.counting_state`, keep this wrapper temporarily:

```python
def build_counting_state(sentence: object) -> tuple[CountingState, UnaryCardinalityMasks]:
    from wfomc.reduction.compat.counting_state import (
        build_counting_state_from_legacy_sentence,
    )

    return build_counting_state_from_legacy_sentence(sentence)
```

This wrapper is acceptable during transition because it does not import legacy FOL directly. Mark it for deletion once callers move.

**Step 4: Update `core.reduce_counting_state()`**

Change:

```python
from wfomc.reduction.counting_state import build_counting_state
from wfomc.fol.compat.legacy_adapter import build_legacy_fol_reduction_view
...
fol_view = build_legacy_fol_reduction_view(reduction_problem)
counting_state, unary_cardinality_masks = build_counting_state(fol_view.transformed_sentence)
```

to:

```python
from wfomc.reduction.compat.counting_state import (
    build_counting_state_from_legacy_sentence,
)
from wfomc.reduction.compat.legacy_view import build_legacy_fol_reduction_view
...
fol_view = build_legacy_fol_reduction_view(reduction_problem)
counting_state, unary_cardinality_masks = build_counting_state_from_legacy_sentence(
    fol_view.transformed_sentence
)
```

This is the first explicit legacy island in `core.py`. Task 7 will wrap it further so `core.py` no longer imports compat directly.

**Step 5: Update tests**

In `tests/unit/test_incremental3_counting_native.py`, prefer:

```python
from wfomc.reduction.compat.counting_state import (
    build_counting_state_from_legacy_sentence,
)
```

If a test is checking the public data class only, keep:

```python
from wfomc.reduction.counting_state import CountingState
```

**Step 6: Run focused tests**

Run:

```bash
uv run pytest tests/unit/test_incremental3_counting_native.py tests/test_incremental3_regressions.py -q
```

Expected: PASS.

---

## Task 5: Move `reduction.legacy_adapter` Into Compat and Delete Shim

**Files:**
- Delete: `src/wfomc/reduction/legacy_adapter.py`
- Modify: `tests/unit/test_fol_package_structure.py`
- Search/Modify: any import of `wfomc.reduction.legacy_adapter`

**Step 1: Search for imports**

Run:

```bash
rg "wfomc\\.reduction\\.legacy_adapter|reduction\\.legacy_adapter" -n src tests docs
```

Expected before this task: only package-structure tests and docs should mention it.

**Step 2: Delete the shim**

Delete:

```text
src/wfomc/reduction/legacy_adapter.py
```

No replacement import should be offered in `wfomc.reduction.__init__`.

**Step 3: Update package-structure tests**

Remove tests that import `wfomc.reduction.legacy_adapter`.

Add:

```python
def test_reduction_legacy_adapter_shim_removed():
    assert not Path("src/wfomc/reduction/legacy_adapter.py").exists()
```

**Step 4: Update docs references later**

Do not churn all historical docs in this task. Only update active docs if they claim `reduction.legacy_adapter` still exists.

**Step 5: Run package tests**

Run:

```bash
uv run pytest tests/unit/test_fol_package_structure.py -q
```

Expected: PASS.

---

## Task 6: Make Typed View Provide Main Reduction Inputs

**Files:**
- Modify: `src/wfomc/reduction/fol_view.py`
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/reduction/state.py`
- Test: `tests/unit/test_reduction_plan.py`
- Test: `tests/unit/test_normal_form.py`
- Test: `tests/unit/test_cell_graph_inputs.py`

**Step 1: Extend `TypedReductionSections`**

In `src/wfomc/reduction/fol_view.py`, add fields:

```python
@dataclass(frozen=True)
class TypedReductionSections:
    universal_body: Formula | None
    existential_formulas: tuple[Formula, ...] = ()
    count_sections: tuple[object, ...] = ()
    order_metadata: TypedOrderMetadata = field(default_factory=TypedOrderMetadata)
    weights: tuple[tuple[object, tuple[object, object]], ...] = ()
```

**Step 2: Build typed weights**

Add helper:

```python
def _problem_weight_items(problem: "Problem") -> tuple[tuple[object, tuple[object, object]], ...]:
    weights = getattr(problem, "weights", None)
    if weights is None:
        return ()
    if isinstance(weights, dict):
        return tuple(sorted(weights.items(), key=lambda item: str(item[0])))
    items = getattr(weights, "items", None)
    if callable(items):
        return tuple(sorted(items(), key=lambda item: str(item[0])))
    return tuple(sorted(tuple(weights), key=lambda item: str(item[0])))
```

Use it in `build_typed_reduction_view()`:

```python
return TypedReductionSections(
    universal_body=qf,
    existential_formulas=existential,
    count_sections=counts,
    order_metadata=_extract_order_metadata(predicates, problem),
    weights=_problem_weight_items(problem),
)
```

**Step 3: Change `_run_pipeline()` to use typed sections**

In `src/wfomc/reduction/core.py`, remove the eager legacy view construction:

```python
from wfomc.fol.compat.legacy_adapter import build_legacy_fol_reduction_view
```

Change state construction:

```python
sections = build_typed_reduction_view(normal_form, problem)
typed_qf = sections.universal_body
if typed_qf is None:
    typed_qf = _typed_true()

state = ReductionState(
    ...
    order_metadata=sections.order_metadata,
    qf_formula=typed_qf,
    weights=dict(sections.weights),
    evidence_plan=build_evidence_plan(problem, options),
)
```

**Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/unit/test_reduction_plan.py tests/unit/test_normal_form.py tests/unit/test_cell_graph_inputs.py -q
```

Expected: any failure here is likely due to typed weights or missing generated predicates. Fix by adjusting `_problem_weight_items()` first, not by reintroducing legacy view into `core.py`.

**Step 5: Run compile-path tests**

Run:

```bash
uv run pytest tests/unit/test_engine_compile.py tests/unit/test_cell_graph_construction.py -q
```

Expected: possible failures if cell graph still expects legacy `qf_formula`. If so, keep Task 6 as a feature flag or defer flipping `qf_formula` until Task 8. The target remains: `core.py` should not eagerly build a legacy view.

---

## Task 7: Wrap Legacy Counting-State Fallback Behind One Compat Function

**Files:**
- Create or modify: `src/wfomc/reduction/compat/counting_state.py`
- Modify: `src/wfomc/reduction/core.py`

**Step 1: Add a high-level compat function**

In `src/wfomc/reduction/compat/counting_state.py`, add:

```python
def build_counting_state_from_problem_normal_form(
    problem: object,
    normal_form: object,
) -> tuple[CountingState, UnaryCardinalityMasks]:
    from wfomc.reduction.compat.legacy_view import build_legacy_fol_reduction_view
    from wfomc.reduction.normal_form import problem_with_normal_form_sentence

    reduction_problem = problem_with_normal_form_sentence(problem, normal_form)
    fol_view = build_legacy_fol_reduction_view(reduction_problem)
    return build_counting_state_from_legacy_sentence(fol_view.transformed_sentence)
```

**Step 2: Make `core.reduce_counting_state()` call only this function**

Change:

```python
from wfomc.reduction.compat.counting_state import (
    build_counting_state_from_legacy_sentence,
)
from wfomc.reduction.compat.legacy_view import build_legacy_fol_reduction_view
from wfomc.reduction.normal_form import problem_with_normal_form_sentence
```

to:

```python
from wfomc.reduction.compat.counting_state import (
    build_counting_state_from_problem_normal_form,
)
```

Then:

```python
counting_state, unary_cardinality_masks = (
    build_counting_state_from_problem_normal_form(state.problem, state.normal_form)
)
```

Now `core.py` imports `reduction.compat`, but it does not import `wfomc.fol.compat`.

**Step 3: Decide whether direct `reduction.compat` import is acceptable in `core.py`**

For this migration, it is acceptable if:

- there is exactly one compat import in `core.py`;
- it exists only in `reduce_counting_state()`;
- it is documented as the incremental3 legacy island.

Add this comment:

```python
# Incremental3 still extracts counting state from the legacy transformed
# sentence. Keep the bridge in reduction.compat until counting-state extraction
# is ported to typed C2 sections.
```

**Step 4: Run tests**

Run:

```bash
uv run pytest tests/unit/test_fol_package_structure.py tests/unit/test_incremental3_counting_native.py tests/test_incremental3_regressions.py -q
```

Expected: PASS.

---

## Task 8: Tighten `_accessors.py` to Typed-First Helpers

**Files:**
- Modify: `src/wfomc/reduction/_accessors.py`
- Modify: `src/wfomc/reduction/counting.py`
- Test: `tests/unit/test_normal_form.py`

**Step 1: Rename mixed helper**

Current:

```python
def count_body_pred(body: object) -> object:
    return getattr(body, "predicate", None) or getattr(body, "pred", None)
```

Replace with typed-only helper:

```python
def typed_count_body_pred(body: object) -> object:
    from wfomc.fol import Atom

    if isinstance(body, Atom):
        return body.predicate
    raise TypeError(f"Expected typed Atom count body, got {body!r}")
```

Keep a compat helper only if needed, and place it in `src/wfomc/reduction/compat/counting.py` or inside `compat/counting_state.py`.

**Step 2: Tighten `is_count_body_atom()`**

Current helper accepts both typed and legacy:

```python
pred = getattr(body, "pred", None)
return pred is not None and getattr(pred, "arity", None) == arity
```

Change to typed-only:

```python
def is_count_body_atom(body: object, *, arity: int) -> bool:
    from wfomc.fol import Atom

    return isinstance(body, Atom) and len(body.terms) == arity + 1
```

If typed `Atom` currently stores terms in `args`, use the actual field from `src/wfomc/fol/syntax.py`. Do not rely on legacy `.pred`.

**Step 3: Update `counting.py` imports**

Change:

```python
count_body_pred as _count_body_pred,
```

to:

```python
typed_count_body_pred as _count_body_pred,
```

**Step 4: Run counting tests**

Run:

```bash
uv run pytest tests/unit/test_normal_form.py tests/unit/test_reduction_plan.py -q
```

Expected: PASS.

---

## Task 9: Enforce External Reduction API Boundary

**Files:**
- Modify: `tests/unit/test_fol_package_structure.py`
- Optionally modify: `src/wfomc/reduction/__init__.py`

**Step 1: Confirm public API**

`src/wfomc/reduction/__init__.py` should export only stable typed pipeline APIs:

```python
__all__ = [
    "CountingReduction",
    "DecodePipeline",
    "ReducedProblem",
    "ReductionPlan",
    "ReductionState",
    "can_reduce_counting_to_ufo2",
    "reduce_to_ufo2",
    "reduce_to_counting_dp",
]
```

Do not export `compat`.

**Step 2: Add external import test**

```python
def test_external_code_uses_root_reduction_public_api():
    forbidden = (
        "from wfomc.reduction.core import",
        "from wfomc.reduction.fol_view import",
        "from wfomc.reduction.legacy_adapter import",
        "from wfomc.reduction.compat import",
        "from wfomc.reduction.compat.",
    )
    allowed_prefix = Path("src/wfomc/reduction")
    offenders = []
    for root in (Path("src/wfomc"), Path("tests")):
        for path in sorted(root.rglob("*.py")):
            if path == Path("tests/unit/test_fol_package_structure.py"):
                continue
            if allowed_prefix in path.parents:
                continue
            source = path.read_text()
            for marker in forbidden:
                if marker in source:
                    offenders.append(f"{path}: {marker}")
    assert offenders == []
```

Keep exceptions if tests intentionally validate compat behavior. Prefer moving those tests under `tests/compat/` or naming them explicitly as allowed.

**Step 3: Run package tests**

Run:

```bash
uv run pytest tests/unit/test_fol_package_structure.py -q
```

Expected: PASS.

---

## Task 10: Full Verification and Cleanup

**Files:**
- Modify as needed based on failures.
- Remove generated caches.

**Step 1: Run focused reduction/FOL suite**

Run:

```bash
uv run pytest \
  tests/unit/test_fol_package_structure.py \
  tests/unit/test_reduction_plan.py \
  tests/unit/test_normal_form.py \
  tests/unit/test_problem_parser.py \
  tests/unit/test_cell_graph_inputs.py \
  tests/unit/test_cell_graph_construction.py \
  tests/unit/test_incremental3_counting_native.py \
  tests/test_incremental3_regressions.py \
  -q
```
Expected: PASS, except any already-known xfail.

**Step 2: Run broader smoke tests**

Run:

```bash
uv run pytest tests/unit/test_fol_syntax.py tests/unit/test_qf_boolean.py tests/test_formula_models.py -q
```

Expected: PASS.

**Step 3: Scan for forbidden imports**

Run:

```bash
rg "wfomc\\.fol\\.compat|wfomc\\.fol\\.legacy_|wfomc\\.reduction\\.legacy_adapter|AtomicFormula|QuantifiedFormula|unary_evidence_from_legacy|cardinality_constraints_from_legacy" -n src/wfomc/reduction tests
```

Expected:

- Matches under `src/wfomc/reduction/compat/` are allowed.
- Matches in `tests/unit/test_fol_package_structure.py` are allowed.
- Matches elsewhere should be fixed or explicitly justified.

**Step 4: Clear generated caches**

Run:

```bash
find src/wfomc/reduction src/wfomc/fol -name '__pycache__' -type d -prune -exec rm -rf {} +
```

**Step 5: Review final file tree**

Run:

```bash
find src/wfomc/reduction -maxdepth 2 -type f | sort
```

Expected tree includes:

```text
src/wfomc/reduction/compat/__init__.py
src/wfomc/reduction/compat/counting_state.py
src/wfomc/reduction/compat/evidence.py
src/wfomc/reduction/compat/legacy_view.py
```

and does not include:

```text
src/wfomc/reduction/legacy_adapter.py
```

---

## Migration Checkpoints

### Checkpoint A: Boundary Created

Done when:

- `src/wfomc/reduction/compat/` exists.
- `reduction.compat` imports are lazy.
- `reduction/legacy_adapter.py` is removed.
- Package-structure tests enforce the boundary.

### Checkpoint B: Legacy Extractor Isolated

Done when:

- `reduction/counting_state.py` has no `wfomc.fol.compat` import.
- `reduction/compat/counting_state.py` owns legacy sentence extraction.
- Incremental3 tests pass.

### Checkpoint C: Typed Main Pipeline

Done when:

- `_run_pipeline()` does not eagerly build `LegacyFolReductionView`.
- `qf_formula`, `weights`, and `order_metadata` come from `TypedReductionSections`.
- Evidence planning takes `problem` and `options`, not `fol_view`.

### Checkpoint D: Enforced Boundary

Done when:

- Only `src/wfomc/reduction/compat/*.py` imports legacy FOL modules.
- External callers use `from wfomc.reduction import ...`.
- Focused reduction/FOL/cell-graph tests pass.

## Risks and Handling

### Risk: Cell graph still expects legacy-shaped formula behavior

Symptoms:

- `tests/unit/test_cell_graph_construction.py` fails.
- Formula model enumeration changes.
- Mixed typed/legacy formula fallback disappears too early.

Handling:

- Do not reintroduce legacy view into `core.py`.
- Keep a compat adapter at the cell-graph boundary if needed.
- Prefer fixing `cell_graph/formula_ops.py` typed behavior.

### Risk: Problem weights are not simple dictionaries

Symptoms:

- `dict(sections.weights)` misses weights.
- Solver tests fail with missing predicate weights.

Handling:

- Add unit tests around `_problem_weight_items(problem)`.
- Support existing weight container shape via `.items()`, tuple pairs, or `weight_items()` helper.

### Risk: Unary evidence tests rely on legacy ground atom conversion

Symptoms:

- `tests/unary_evidence/*` fail after removing `fol_view` from `build_evidence_plan`.

Handling:

- Keep lazy fallback through `reduction.compat.evidence`.
- Do not import `unary_evidence_from_legacy` directly in `evidence_planning.py`.

### Risk: Incremental3 counting extraction is still legacy-only

Symptoms:

- `reduce_to_counting_dp` fails.

Handling:

- Keep `reduction.compat.counting_state.build_counting_state_from_problem_normal_form()`.
- Treat it as the one allowed legacy island until typed counting-state extraction is implemented from `C2NormalForm`.

## Not In Scope

- Rewriting `wfomc.cell_graph` to fully typed-only.
- Removing `wfomc.fol.compat`.
- Replacing legacy parser compatibility facades.
- Porting incremental3 counting-state extraction fully to typed C2 sections.
- Full solver algorithm redesign.

## Final Acceptance Criteria

- `src/wfomc/reduction/compat/` contains all reduction legacy adapters.
- `src/wfomc/reduction/legacy_adapter.py` is gone.
- `src/wfomc/reduction/core.py` has no direct `wfomc.fol.compat` import.
- `src/wfomc/reduction/evidence_planning.py` has no direct legacy conversion imports.
- `src/wfomc/reduction/counting_state.py` has no direct legacy FOL imports.
- `tests/unit/test_fol_package_structure.py` enforces the new boundary.
- Focused tests pass:

```bash
uv run pytest \
  tests/unit/test_fol_package_structure.py \
  tests/unit/test_reduction_plan.py \
  tests/unit/test_normal_form.py \
  tests/unit/test_problem_parser.py \
  tests/unit/test_cell_graph_inputs.py \
  tests/unit/test_cell_graph_construction.py \
  tests/unit/test_incremental3_counting_native.py \
  tests/test_incremental3_regressions.py \
  -q
```
