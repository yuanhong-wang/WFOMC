# FOL Typed Convergence And Compat Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Converge `wfomc.fol` into a clean typed Formula package, and move all legacy/compatibility FOL code under `wfomc.fol.compat`.

**Architecture:** Typed FOL keeps one AST (`Formula`) and separates construction, structural analysis, rewriting, semantics, and presentation into explicit modules. Legacy FOL, native boolean, old parser helpers, and typed-to-legacy adapters become compatibility modules under `wfomc.fol.compat`, with short-lived import shims for old paths while production code is migrated.

**Tech Stack:** Python 3.11, dataclasses, pytest, existing `wfomc.fol` typed modules, existing legacy FOL modules, existing parser/reduction/cell-graph tests.

---

## Current State

The current `fol` package already has a typed core:

- `src/wfomc/fol/formulas.py`: typed Formula AST nodes.
- `src/wfomc/fol/symbols.py`: typed `Predicate`, `Variable`, `Constant`, `Sort`, `Function`.
- `src/wfomc/fol/terms.py`: typed term nodes.
- `src/wfomc/fol/context.py`: construction, interning, context-local helpers.
- `src/wfomc/fol/syntax.py`: public typed facade, but currently imports constructor helpers from `builders.py`.
- `src/wfomc/fol/analysis.py`: structure queries such as predicates, free vars, atoms.
- `src/wfomc/fol/rewrite.py`: substitution and syntactic rewrites.
- `src/wfomc/fol/qf.py`: typed quantifier-free Boolean semantics plus `Literal`.

The compatibility/legacy surface is still mixed into the same package:

- `src/wfomc/fol/legacy_syntax.py`
- `src/wfomc/fol/legacy_sc2.py`
- `src/wfomc/fol/legacy_utils.py`
- `src/wfomc/fol/native_boolean.py`
- `src/wfomc/fol/ast.py`
- `src/wfomc/fol/signature.py`
- `src/wfomc/reduction/legacy_adapter.py`
- `src/wfomc/compat/fol_parser.py`
- `src/wfomc/compat/wfomcs_parser.py`
- legacy exports in `src/wfomc/fol/__init__.py`

The main production dependants are:

- `src/wfomc/parser/transformers/fol.py`: typed parser.
- `src/wfomc/normal_form/c2/normalize.py`: typed normal form.
- `src/wfomc/reduction/*.py`: mixed typed and legacy-adapter reduction.
- `src/wfomc/cell_graph/*.py`: typed formula ops, but still has legacy fallback.
- `src/wfomc/grounding/propositional.py`: still uses `native_boolean` and legacy syntax.
- `src/wfomc/unary_evidence.py`: still imports legacy names from root `wfomc.fol`.
- `src/wfomc/compat/*.py`: deprecated public compatibility APIs.

Target package layout:

```text
src/wfomc/fol/
  __init__.py              # typed-first facade, re-exports dsl.py public API
  analysis.py              # structure queries only
  context.py               # construction engine / interning
  dsl.py                   # public typed construction facade and traversal helpers
  literals.py              # signed Atom helper
  pretty.py                # formatting only
  rewrite.py               # syntactic transformation only
  semantics.py             # truth/model/CNF/SAT semantics over typed Formula
  syntax.py                # typed symbols, terms, and Formula AST only
  compat/
    __init__.py
    legacy_adapter.py      # typed <-> legacy bridge, moved from reduction
    legacy_sc2.py
    legacy_syntax.py
    legacy_utils.py
    native_boolean.py
```

Short-lived shims may remain at old paths during migration:

```text
src/wfomc/fol/qf.py
src/wfomc/fol/ast.py
src/wfomc/fol/signature.py
src/wfomc/fol/legacy_sc2.py
src/wfomc/fol/legacy_syntax.py
src/wfomc/fol/legacy_utils.py
src/wfomc/fol/native_boolean.py
```

Each shim should contain only imports from `wfomc.fol.compat.*` or new typed modules plus a deprecation comment.

---

## Design Rules

1. `Formula` is the only formula representation for the typed package.
2. `Formula` nodes must not grow execution methods such as `models()`, `satisfiable()`, or `to_cnf()`.
3. `syntax.py` is the only ergonomic typed public facade.
4. `context.py` is the only construction engine.
5. `analysis.py` is structural and must not evaluate truth.
6. `rewrite.py` is syntactic and must not enumerate models.
7. `semantics.py` owns truth, model, SAT, and CNF behavior.
8. `literals.py` owns signed atoms.
9. Any import of `legacy_*`, `native_boolean`, or legacy `Pred/Const/AtomicFormula/SC2/QFFormula` must be either inside `wfomc.fol.compat`, `wfomc.compat`, or an explicitly named adapter.
10. New production code must not import from root `wfomc.fol` when it needs a precise typed object; use `wfomc.fol.syntax`, `wfomc.fol.formulas`, `wfomc.fol.symbols`, `wfomc.fol.literals`, or `wfomc.fol.semantics`.

---

## Task 1: Add Package Boundary Tests

**Files:**

- Modify: `tests/unit/test_package_structure.py`
- Test: `tests/unit/test_package_structure.py`

**Step 1: Write failing tests for typed module boundaries**

Add tests that encode the target invariants before moving code:

```python
def test_typed_fol_modules_do_not_import_compat_modules():
    typed_modules = [
        Path("src/wfomc/fol/analysis.py"),
        Path("src/wfomc/fol/context.py"),
        Path("src/wfomc/fol/dsl.py"),
        Path("src/wfomc/fol/literals.py"),
        Path("src/wfomc/fol/pretty.py"),
        Path("src/wfomc/fol/rewrite.py"),
        Path("src/wfomc/fol/semantics.py"),
        Path("src/wfomc/fol/syntax.py"),
    ]
    forbidden = [
        "wfomc.fol.legacy_",
        "wfomc.fol.native_boolean",
        "wfomc.fol.compat",
        "wfomc.reduction.legacy_adapter",
    ]
    for path in typed_modules:
        if not path.exists():
            continue
        source = path.read_text()
        for needle in forbidden:
            assert needle not in source, f"{path} imports {needle}"


def test_legacy_fol_modules_live_under_fol_compat():
    compat_modules = [
        Path("src/wfomc/fol/compat/legacy_syntax.py"),
        Path("src/wfomc/fol/compat/legacy_sc2.py"),
        Path("src/wfomc/fol/compat/legacy_utils.py"),
        Path("src/wfomc/fol/compat/native_boolean.py"),
    ]
    for path in compat_modules:
        assert path.exists()
```

Do not add `qf.py` to the typed module list because it will become a shim in Task 5.

**Step 2: Run the package tests and verify failure**

Run:

```bash
uv run pytest tests/unit/test_package_structure.py -q
```

Expected: FAIL because `fol/literals.py`, `fol/semantics.py`, and `fol/compat/*` do not exist yet.

**Step 3: Commit the failing boundary tests**

```bash
git add tests/unit/test_package_structure.py
git commit -m "test: define typed fol and compat boundaries"
```

---

## Task 2: Merge `builders.py` Into `syntax.py`

**Files:**

- Modify: `src/wfomc/fol/syntax.py`
- Delete or Shim: `src/wfomc/fol/builders.py`
- Modify: `tests/unit/test_package_structure.py`
- Test: `tests/unit/test_fol_syntax.py`

**Step 1: Move constructor helpers into `syntax.py`**

Remove:

```python
from .builders import (
    atom,
    conjunction,
    count,
    disjunction,
    eq,
    exists,
    false,
    forall,
    iff,
    implies,
    legacy,
    neg,
    true,
)
```

Add the helper implementations directly in `syntax.py` after the default constants:

```python
def atom(predicate: object, *terms: object) -> Formula:
    return context_for(predicate, *terms).atom(predicate, *terms)


def eq(left: object, right: object) -> Formula:
    return context_for(left, right).eq(left, right)


def neg(formula: object) -> Formula:
    return context_for(formula).neg(formula)


def conjunction(*formulas: object) -> Formula:
    return context_for(*formulas).conjunction(*formulas)


def disjunction(*formulas: object) -> Formula:
    return context_for(*formulas).disjunction(*formulas)


def implies(left: object, right: object) -> Formula:
    return context_for(left, right).implies(left, right)


def iff(left: object, right: object) -> Formula:
    return context_for(left, right).iff(left, right)


def forall(variables: object | Iterable[object], body: object) -> Formula:
    return context_for(body).forall(variables, body)


def exists(variables: object | Iterable[object], body: object) -> Formula:
    return context_for(body).exists(variables, body)


def count(variable: object, comparator: str, count_value: object, body: object) -> Formula:
    return context_for(body).count(variable, comparator, count_value, body)


def true() -> Formula:
    return _DEFAULT_CONTEXT.true()


def false() -> Formula:
    return _DEFAULT_CONTEXT.false()
```

Do not keep `legacy()` in the typed public surface after Task 4. If tests still need it temporarily, keep it marked deprecated:

```python
def legacy(value: object) -> Formula:
    from wfomc.fol.compat.legacy_adapter import wrap_legacy_formula

    return wrap_legacy_formula(value)
```

**Step 2: Remove `builders.py` from package-structure expectations**

Update any list in `tests/unit/test_package_structure.py` that currently requires `wfomc.fol.builders`.

If external import compatibility is required for one release, keep a shim:

```python
"""Deprecated constructor shim; import from wfomc.fol.syntax."""

from .syntax import (
    atom,
    conjunction,
    count,
    disjunction,
    eq,
    exists,
    false,
    forall,
    iff,
    implies,
    neg,
    true,
)

__all__ = [
    "atom",
    "conjunction",
    "count",
    "disjunction",
    "eq",
    "exists",
    "false",
    "forall",
    "iff",
    "implies",
    "neg",
    "true",
]
```

Preferred final state: delete `src/wfomc/fol/builders.py` after all internal imports are gone.

**Step 3: Run tests**

```bash
uv run pytest tests/unit/test_fol_syntax.py tests/unit/test_package_structure.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add src/wfomc/fol/syntax.py src/wfomc/fol/builders.py tests/unit/test_package_structure.py
git commit -m "refactor: merge fol builders into typed syntax facade"
```

---

## Task 3: Split `Literal` Out Of `qf.py`

**Files:**

- Create: `src/wfomc/fol/literals.py`
- Modify: `src/wfomc/fol/qf.py`
- Modify: `src/wfomc/fol/__init__.py`
- Modify: `src/wfomc/cell_graph/components.py`
- Modify: `src/wfomc/cell_graph/utils.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Modify: `src/wfomc/cell_graph/formula_ops.py`
- Test: `tests/unit/test_qf_boolean.py`
- Test: `tests/test_formula_models.py`

**Step 1: Create `literals.py`**

```python
"""Signed atoms for typed FOL formulas."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.fol.formulas import Atom


@dataclass(frozen=True, slots=True)
class Literal:
    """A signed atom: positive means the atom holds, negative means it does not."""

    atom: Atom
    positive: bool = True

    @property
    def pred(self) -> object:
        return self.atom.predicate

    @property
    def predicate(self) -> object:
        return self.atom.predicate

    @property
    def args(self) -> tuple[object, ...]:
        return self.atom.terms

    @property
    def terms(self) -> tuple[object, ...]:
        return self.atom.terms

    def __invert__(self) -> "Literal":
        return Literal(self.atom, not self.positive)

    def make_positive(self) -> "Literal":
        return Literal(self.atom, True)

    def substitute(self, mapping: dict[object, object]) -> "Literal":
        from wfomc.fol.rewrite import substitute

        new_atom = substitute(self.atom, mapping)
        if isinstance(new_atom, Atom):
            return Literal(new_atom, self.positive)
        return self

    def __str__(self) -> str:
        sign = "" if self.positive else "~"
        return f"{sign}{self.atom}"

    def __hash__(self) -> int:
        return hash((self.atom, self.positive))


def positive_atom(lit_or_atom: Atom | Literal) -> Atom:
    if isinstance(lit_or_atom, Literal):
        return lit_or_atom.atom
    return lit_or_atom


__all__ = ["Literal", "positive_atom"]
```

**Step 2: Update `qf.py` imports**

Remove the local `Literal` class and import it:

```python
from wfomc.fol.literals import Literal, positive_atom
```

Keep re-exporting `Literal` and `positive_atom` from `qf.py` for compatibility until Task 5.

**Step 3: Update direct production imports**

Change:

```python
from wfomc.fol.qf import Literal
```

to:

```python
from wfomc.fol.literals import Literal
```

in:

- `src/wfomc/cell_graph/components.py`
- `src/wfomc/cell_graph/utils.py`
- `src/wfomc/cell_graph/cell_graph.py`

In `src/wfomc/cell_graph/formula_ops.py`, change:

```python
from wfomc.fol.qf import Literal, is_satisfiable, model_literals, positive_atom, qf_simplify
```

to:

```python
from wfomc.fol.literals import Literal, positive_atom
from wfomc.fol.semantics import is_satisfiable, model_literals
from wfomc.fol.rewrite import simplify_boolean as qf_simplify
```

The `semantics` and `simplify_boolean` imports are introduced in later tasks; if this task is implemented before Task 4, keep the old qf imports for `is_satisfiable`, `model_literals`, and `qf_simplify`.

**Step 4: Update root lazy exports**

Change:

```python
"Literal": ("wfomc.fol.qf", "Literal"),
"positive_atom": ("wfomc.fol.qf", "positive_atom"),
```

to:

```python
"Literal": ("wfomc.fol.literals", "Literal"),
"positive_atom": ("wfomc.fol.literals", "positive_atom"),
```

**Step 5: Run tests**

```bash
uv run pytest tests/unit/test_qf_boolean.py tests/test_formula_models.py tests/unit/test_package_structure.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/fol/literals.py src/wfomc/fol/qf.py src/wfomc/fol/__init__.py src/wfomc/cell_graph tests/unit/test_package_structure.py
git commit -m "refactor: move typed fol literals to dedicated module"
```

---

## Task 4: Move Boolean Semantics From `qf.py` To `semantics.py`

**Files:**

- Create: `src/wfomc/fol/semantics.py`
- Modify: `src/wfomc/fol/qf.py`
- Modify: `src/wfomc/cell_graph/formula_ops.py`
- Modify: `tests/unit/test_qf_boolean.py`
- Modify: `tests/test_formula_models.py`
- Test: `tests/unit/test_qf_boolean.py`
- Test: `tests/test_formula_models.py`

**Step 1: Create `semantics.py`**

Move these functions from `qf.py`:

- `_atoms_list`
- `_collect`
- `_evaluate`
- `_evaluate_eq`
- `to_cnf_clauses`
- `_tseitin`
- `is_satisfiable`
- `models`
- `model_literals`
- `_brute_model_literals`

The public surface should be:

```python
__all__ = [
    "evaluate",
    "is_satisfiable",
    "model_literals",
    "models",
    "to_cnf_clauses",
]
```

Make `_evaluate` public as `evaluate`:

```python
def evaluate(formula: Formula, assignment: dict[Atom, bool]) -> bool:
    ...
```

Then update internal calls from `_evaluate(...)` to `evaluate(...)`.

**Step 2: Make unsupported non-QF formulas explicit**

At the top of public semantics functions, reject quantified formulas:

```python
from wfomc.fol.analysis import is_quantifier_free


def _require_quantifier_free(formula: Formula) -> None:
    if not is_quantifier_free(formula):
        raise ValueError("Formula semantics are only defined for quantifier-free formulas")
```

Call `_require_quantifier_free(formula)` in:

- `evaluate`
- `to_cnf_clauses`
- `models`
- `model_literals`
- `is_satisfiable`

If existing tests rely on silently accepting quantified formulas, add targeted tests documenting the new error and update those call sites to pass a quantifier-free body.

**Step 3: Update tests to import from `semantics.py`**

Change:

```python
from wfomc.fol.qf import is_satisfiable, models
```

to:

```python
from wfomc.fol.semantics import is_satisfiable, models
```

in:

- `tests/unit/test_qf_boolean.py`
- `tests/test_formula_models.py`

**Step 4: Keep `qf.py` as a shim**

Replace `qf.py` contents with:

```python
"""Deprecated quantifier-free facade.

Use wfomc.fol.literals, wfomc.fol.semantics, and wfomc.fol.rewrite instead.
"""

from __future__ import annotations

from wfomc.fol.literals import Literal, positive_atom
from wfomc.fol.rewrite import simplify_boolean as qf_simplify
from wfomc.fol.semantics import (
    is_satisfiable,
    model_literals,
    models,
    to_cnf_clauses,
)

__all__ = [
    "Literal",
    "is_satisfiable",
    "model_literals",
    "models",
    "positive_atom",
    "qf_simplify",
    "to_cnf_clauses",
]
```

This requires Task 5's `simplify_boolean` first, or keep `qf_simplify` in `qf.py` until Task 5 is done.

**Step 5: Run tests**

```bash
uv run pytest tests/unit/test_qf_boolean.py tests/test_formula_models.py tests/unit/test_package_structure.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/fol/semantics.py src/wfomc/fol/qf.py tests/unit/test_qf_boolean.py tests/test_formula_models.py tests/unit/test_package_structure.py
git commit -m "refactor: move typed boolean semantics out of qf module"
```

---

## Task 5: Move Boolean Simplification Into `rewrite.py`

**Files:**

- Modify: `src/wfomc/fol/rewrite.py`
- Modify: `src/wfomc/fol/qf.py`
- Modify: `src/wfomc/cell_graph/formula_ops.py`
- Test: `tests/unit/test_qf_boolean.py`

**Step 1: Add `simplify_boolean` to `rewrite.py`**

Move `qf_simplify` from `qf.py` to `rewrite.py` and rename it:

```python
def simplify_boolean(formula: Formula) -> Formula:
    """Simplify quantifier-free Boolean structure without changing FOL meaning."""
    ...
```

Keep the same behavior:

- Fold `BoolConst`.
- Simplify `Not(BoolConst)`.
- Remove `true` from conjunction.
- Short-circuit conjunction on `false`.
- Remove `false` from disjunction.
- Short-circuit disjunction on `true`.
- Simplify `Implies` and `Iff` when either side is a `BoolConst`.
- Simplify `Eq(x, x)` to `true`.

**Step 2: Add compatibility alias in `qf.py`**

```python
from wfomc.fol.rewrite import simplify_boolean as qf_simplify
```

**Step 3: Update production imports**

In `src/wfomc/cell_graph/formula_ops.py`, import:

```python
from wfomc.fol.rewrite import simplify_boolean as qf_simplify
```

Do not import `qf_simplify` from `wfomc.fol.qf` in production code after this task.

**Step 4: Run tests**

```bash
uv run pytest tests/unit/test_qf_boolean.py tests/unit/test_reduction.py tests/unit/test_reduction_backbone.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/wfomc/fol/rewrite.py src/wfomc/fol/qf.py src/wfomc/cell_graph/formula_ops.py
git commit -m "refactor: move boolean simplification into fol rewrite"
```

---

## Task 6: Retire `qf.py` From Internal Imports

**Files:**

- Modify: `src/wfomc/cell_graph/formula_ops.py`
- Modify: `src/wfomc/cell_graph/components.py`
- Modify: `src/wfomc/cell_graph/utils.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Modify: `src/wfomc/fol/__init__.py`
- Modify: `tests/unit/test_qf_boolean.py`
- Modify: `tests/test_formula_models.py`
- Test: `tests/unit/test_qf_boolean.py`
- Test: `tests/test_formula_models.py`

**Step 1: Replace all internal imports from `wfomc.fol.qf`**

Run:

```bash
rg "wfomc\\.fol\\.qf|from \\.qf" src tests
```

Expected before edit: only `qf.py` shim, old tests, and cell graph code.

Change all internal code to:

```python
from wfomc.fol.literals import Literal, positive_atom
from wfomc.fol.semantics import is_satisfiable, model_literals, models, to_cnf_clauses
from wfomc.fol.rewrite import simplify_boolean
```

**Step 2: Keep external compatibility**

Leave `src/wfomc/fol/qf.py` as a shim for external callers. Mark it deprecated in the docstring only; do not emit runtime warnings yet because this is a library import path and warnings may break strict tests.

**Step 3: Add a boundary test**

Add:

```python
def test_production_code_does_not_import_fol_qf():
    for path in Path("src/wfomc").rglob("*.py"):
        if path == Path("src/wfomc/fol/qf.py"):
            continue
        source = path.read_text()
        assert "wfomc.fol.qf" not in source
        assert "from .qf" not in source
```

If `src/wfomc/fol/__init__.py` intentionally points `Literal` to `literals.py`, it should pass.

**Step 4: Run tests**

```bash
uv run pytest tests/unit/test_qf_boolean.py tests/test_formula_models.py tests/unit/test_package_structure.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/wfomc tests/unit/test_qf_boolean.py tests/test_formula_models.py tests/unit/test_package_structure.py
git commit -m "refactor: remove internal fol qf imports"
```

---

## Task 7: Move Legacy FOL Modules Under `fol/compat`

**Files:**

- Create: `src/wfomc/fol/compat/__init__.py`
- Move: `src/wfomc/fol/legacy_syntax.py` -> `src/wfomc/fol/compat/legacy_syntax.py`
- Move: `src/wfomc/fol/legacy_sc2.py` -> `src/wfomc/fol/compat/legacy_sc2.py`
- Move: `src/wfomc/fol/legacy_utils.py` -> `src/wfomc/fol/compat/legacy_utils.py`
- Move: `src/wfomc/fol/native_boolean.py` -> `src/wfomc/fol/compat/native_boolean.py`
- Modify: old files as shims
- Test: `tests/unit/test_package_structure.py`

**Step 1: Create the compat package**

```python
"""Compatibility-only FOL modules.

Production typed FOL code should not import from this package except through
explicit migration adapters.
"""
```

**Step 2: Move modules preserving content**

Use regular file moves for the four modules:

```bash
mkdir -p src/wfomc/fol/compat
git mv src/wfomc/fol/legacy_syntax.py src/wfomc/fol/compat/legacy_syntax.py
git mv src/wfomc/fol/legacy_sc2.py src/wfomc/fol/compat/legacy_sc2.py
git mv src/wfomc/fol/legacy_utils.py src/wfomc/fol/compat/legacy_utils.py
git mv src/wfomc/fol/native_boolean.py src/wfomc/fol/compat/native_boolean.py
```

**Step 3: Update intra-compat imports**

Inside moved modules, replace:

```python
from wfomc.fol.legacy_syntax import ...
from wfomc.fol.legacy_sc2 import ...
from wfomc.fol.legacy_utils import ...
from wfomc.fol import native_boolean
```

with:

```python
from wfomc.fol.compat.legacy_syntax import ...
from wfomc.fol.compat.legacy_sc2 import ...
from wfomc.fol.compat.legacy_utils import ...
from wfomc.fol.compat import native_boolean
```

Prefer relative imports inside `fol/compat`:

```python
from .legacy_syntax import ...
from .legacy_sc2 import ...
from .legacy_utils import ...
```

**Step 4: Create old-path shims**

For each old module, create a shim:

```python
"""Deprecated compatibility shim; import from wfomc.fol.compat.legacy_syntax."""

from wfomc.fol.compat.legacy_syntax import *  # noqa: F401,F403
```

Use matching module names for each shim.

**Step 5: Run import tests**

```bash
uv run pytest tests/unit/test_package_structure.py -q
python - <<'PY'
import wfomc.fol.compat.legacy_syntax
import wfomc.fol.compat.legacy_sc2
import wfomc.fol.compat.legacy_utils
import wfomc.fol.compat.native_boolean
import wfomc.fol.legacy_syntax
import wfomc.fol.legacy_sc2
import wfomc.fol.legacy_utils
import wfomc.fol.native_boolean
print("ok")
PY
```

Expected: PASS and `ok`.

**Step 6: Commit**

```bash
git add src/wfomc/fol/compat src/wfomc/fol/legacy_syntax.py src/wfomc/fol/legacy_sc2.py src/wfomc/fol/legacy_utils.py src/wfomc/fol/native_boolean.py tests/unit/test_package_structure.py
git commit -m "refactor: move legacy fol modules under fol compat"
```

---

## Task 8: Move `ast.py` And `signature.py` To Compat Or Remove Them

**Files:**

- Modify or Delete: `src/wfomc/fol/ast.py`
- Modify or Delete: `src/wfomc/fol/signature.py`
- Create Optional: `src/wfomc/fol/compat/ast.py`
- Create Optional: `src/wfomc/fol/compat/signature.py`
- Modify: `src/wfomc/fol/__init__.py`
- Modify: `tests/unit/test_package_structure.py`

**Step 1: Decide treatment**

Current state:

- `ast.py` is only a compatibility export for classes already in `formulas.py`.
- `signature.py` is only a compatibility export for classes already in `symbols.py` and `terms.py`.

Preferred final state:

- Delete both from the typed module set.
- Keep old paths as shims for external callers only if package tests require backwards compatibility.

**Step 2: Update root exports**

Change `src/wfomc/fol/__init__.py` typed symbol exports from:

```python
"Constant": ("wfomc.fol.signature", "Constant"),
"Predicate": ("wfomc.fol.signature", "Predicate"),
"Variable": ("wfomc.fol.signature", "Variable"),
```

to:

```python
"Constant": ("wfomc.fol.symbols", "Constant"),
"Predicate": ("wfomc.fol.symbols", "Predicate"),
"Variable": ("wfomc.fol.symbols", "Variable"),
```

Do the same for `Function`, `Sort`, `Term`, and `FunctionApplication` if they are exported.

**Step 3: If keeping shims, make them explicitly deprecated**

`src/wfomc/fol/ast.py`:

```python
"""Deprecated typed AST shim; import from wfomc.fol.formulas."""

from wfomc.fol.formulas import *  # noqa: F401,F403
```

`src/wfomc/fol/signature.py`:

```python
"""Deprecated typed symbol shim; import from wfomc.fol.symbols and wfomc.fol.terms."""

from wfomc.fol.symbols import *  # noqa: F401,F403
from wfomc.fol.terms import FunctionApplication, Term
```

**Step 4: Run tests**

```bash
uv run pytest tests/unit/test_package_structure.py tests/unit/test_fol_syntax.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/wfomc/fol/ast.py src/wfomc/fol/signature.py src/wfomc/fol/__init__.py tests/unit/test_package_structure.py
git commit -m "refactor: retire fol ast and signature aliases"
```

---

## Task 9: Move `reduction/legacy_adapter.py` To `fol/compat`

**Files:**

- Move: `src/wfomc/reduction/legacy_adapter.py` -> `src/wfomc/fol/compat/legacy_adapter.py`
- Create Shim: `src/wfomc/reduction/legacy_adapter.py`
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/reduction/counting_state.py`
- Modify: `src/wfomc/cell_graph/formula_ops.py`
- Modify: `tests/unit/test_package_structure.py`
- Test: `tests/unit/test_reduction.py`
- Test: `tests/unit/test_reduction_backbone.py`

**Step 1: Move the adapter**

```bash
git mv src/wfomc/reduction/legacy_adapter.py src/wfomc/fol/compat/legacy_adapter.py
```

This adapter conceptually belongs to compatibility because its purpose is typed-to-legacy projection and old SC2/QF object construction.

**Step 2: Update imports inside the moved file**

Replace old imports:

```python
from wfomc.fol.legacy_syntax import ...
from wfomc.fol.legacy_utils import ...
from wfomc.fol.legacy_sc2 import ...
```

with:

```python
from wfomc.fol.compat.legacy_syntax import ...
from wfomc.fol.compat.legacy_utils import ...
from wfomc.fol.compat.legacy_sc2 import ...
```

**Step 3: Update callers**

Replace:

```python
from wfomc.reduction.legacy_adapter import ...
```

with:

```python
from wfomc.fol.compat.legacy_adapter import ...
```

Known callers:

- `src/wfomc/reduction/core.py`
- `src/wfomc/reduction/counting_state.py`
- `src/wfomc/cell_graph/formula_ops.py`

**Step 4: Add old-path shim**

`src/wfomc/reduction/legacy_adapter.py`:

```python
"""Deprecated shim; import legacy FOL adapters from wfomc.fol.compat.legacy_adapter."""

from wfomc.fol.compat.legacy_adapter import *  # noqa: F401,F403
```

**Step 5: Run tests**

```bash
uv run pytest tests/unit/test_reduction.py tests/unit/test_reduction_backbone.py tests/unit/test_package_structure.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/reduction/legacy_adapter.py src/wfomc/fol/compat/legacy_adapter.py src/wfomc/reduction src/wfomc/cell_graph tests/unit/test_package_structure.py
git commit -m "refactor: move reduction legacy adapter into fol compat"
```

---

## Task 10: Update `src/wfomc/compat/*` To Import From `fol.compat`

**Files:**

- Modify: `src/wfomc/compat/cardinality.py`
- Modify: `src/wfomc/compat/problem.py`
- Modify: `src/wfomc/compat/fol_parser.py`
- Modify: `src/wfomc/compat/wfomcs_parser.py`
- Modify: `src/wfomc/compat/parser.py`
- Test: `tests/unit/test_package_structure.py`

**Step 1: Replace legacy root imports**

In `src/wfomc/compat/cardinality.py`, change:

```python
from wfomc.fol import Pred
```

to:

```python
from wfomc.fol.compat.legacy_syntax import Pred
```

In `src/wfomc/compat/problem.py`, change:

```python
from wfomc.fol import AtomicFormula, Const, Pred, SC2, formula_to_str
```

to:

```python
from wfomc.fol.compat.legacy_syntax import AtomicFormula, Const, Pred
from wfomc.fol.compat.legacy_sc2 import SC2
from wfomc.fol.compat.legacy_utils import formula_to_str
```

In `src/wfomc/compat/fol_parser.py`, change legacy imports to `wfomc.fol.compat.*`.

In `src/wfomc/compat/wfomcs_parser.py`, change:

```python
from wfomc.fol.legacy_sc2 import SC2, to_sc2
from wfomc.fol.legacy_syntax import Const, Pred
```

to:

```python
from wfomc.fol.compat.legacy_sc2 import SC2, to_sc2
from wfomc.fol.compat.legacy_syntax import Const, Pred
```

In `src/wfomc/compat/parser.py`, replace function-local imports of `wfomc.fol.legacy_syntax`.

**Step 2: Run compatibility tests**

```bash
uv run pytest tests/unit/test_package_structure.py tests/unit/test_runtime_cache.py tests/unit/test_evidence_planning.py -q
```

Expected: PASS.

**Step 3: Commit**

```bash
git add src/wfomc/compat tests/unit/test_package_structure.py
git commit -m "refactor: point deprecated compat APIs at fol compat"
```

---

## Task 11: Make Root `wfomc.fol` Typed-First

**Files:**

- Modify: `src/wfomc/fol/__init__.py`
- Modify: tests that import legacy names from root `wfomc.fol`
- Test: `tests/unit/test_package_structure.py`

**Step 1: Split root exports into typed and deprecated sections**

Keep typed exports:

```python
_TYPED_EXPORTS = {
    "FormulaKind": ("wfomc.fol.syntax", "FormulaKind"),
    "TypedFormula": ("wfomc.fol.syntax", "Formula"),
    "Formula": ("wfomc.fol.syntax", "Formula"),
    "Constant": ("wfomc.fol.symbols", "Constant"),
    "Predicate": ("wfomc.fol.symbols", "Predicate"),
    "Variable": ("wfomc.fol.symbols", "Variable"),
    "Literal": ("wfomc.fol.literals", "Literal"),
    "positive_atom": ("wfomc.fol.literals", "positive_atom"),
    "syntax": ("wfomc.fol.syntax", None),
}
```

Move legacy exports to:

```python
_DEPRECATED_COMPAT_EXPORTS = {
    "AtomicFormula": ("wfomc.fol.compat.legacy_syntax", "AtomicFormula"),
    "Const": ("wfomc.fol.compat.legacy_syntax", "Const"),
    "Pred": ("wfomc.fol.compat.legacy_syntax", "Pred"),
    "QFFormula": ("wfomc.fol.compat.legacy_syntax", "QFFormula"),
    "SC2": ("wfomc.fol.compat.legacy_sc2", "SC2"),
    "to_sc2": ("wfomc.fol.compat.legacy_sc2", "to_sc2"),
    "native_boolean": ("wfomc.fol.compat.native_boolean", None),
    ...
}
```

Then:

```python
_EXPORTS = {**_TYPED_EXPORTS, **_DEPRECATED_COMPAT_EXPORTS}
```

**Step 2: Avoid typed/legacy name collision**

Current root `Formula` points to legacy `Formula`. Change it to typed `Formula`.

If external compatibility requires legacy `Formula`, expose it as:

```python
"LegacyFormulaBase": ("wfomc.fol.compat.legacy_syntax", "Formula")
```

Do not keep root `Formula` as legacy after this task.

**Step 3: Update tests and production code that still import legacy from root**

Known tests/callers:

- `tests/unary_evidence/test_required_unary_preds.py`
- `tests/unary_evidence/test_cell_evidence_allocation.py`
- `tests/unary_evidence/test_unary_evidence_partition.py`
- `tests/unit/test_runtime_cache.py`
- `tests/unit/test_evidence_planning.py`
- `src/wfomc/unary_evidence.py`
- `src/wfomc/cell_graph/cell_graph.py`

For legacy call sites, replace:

```python
from wfomc.fol import Pred, Const, X
```

with:

```python
from wfomc.fol.compat.legacy_syntax import Pred, Const, X
```

For typed call sites, use:

```python
from wfomc.fol.syntax import Predicate, Constant, X
```

**Step 4: Add no-root-import production test**

Add or update:

```python
def test_production_code_does_not_import_legacy_from_root_fol():
    allowed = {
        Path("src/wfomc/fol/__init__.py"),
    }
    for path in Path("src/wfomc").rglob("*.py"):
        if path in allowed:
            continue
        source = path.read_text()
        assert "from wfomc.fol import" not in source
```

If this is too broad during migration, allow only explicitly deprecated `src/wfomc/compat/*` until Task 10 lands.

**Step 5: Run tests**

```bash
uv run pytest tests/unit/test_package_structure.py tests/unary_evidence tests/unit/test_runtime_cache.py tests/unit/test_evidence_planning.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/fol/__init__.py src/wfomc/unary_evidence.py src/wfomc/cell_graph tests tests/unit/test_package_structure.py
git commit -m "refactor: make fol root facade typed first"
```

---

## Task 12: Migrate `grounding/propositional.py` Off `native_boolean`

**Files:**

- Modify: `src/wfomc/grounding/propositional.py`
- Add or Modify: tests covering propositional grounding
- Test: related grounding tests

**Step 1: Inspect current behavior**

Run:

```bash
sed -n '1,260p' src/wfomc/grounding/propositional.py
rg "grounding.propositional|ground.*propositional|get_models|get_atom|get_symbol" -n tests src
```

Identify which functions require:

- Boolean expression construction.
- Atom collection.
- Model enumeration.
- Substitution.
- Formatting.

**Step 2: Replace `native_boolean` semantics with typed FOL semantics**

Change imports from:

```python
from wfomc.fol import native_boolean as boolean
from wfomc.fol.legacy_syntax import AtomicFormula, Const, Pred, X, Y
```

to typed equivalents:

```python
from wfomc.fol.formulas import Atom, Formula
from wfomc.fol.literals import Literal
from wfomc.fol.semantics import model_literals, models
from wfomc.fol.syntax import Constant, Predicate, X, Y, atom, conjunction, disjunction, false, neg, true
```

If a legacy public API still needs `AtomicFormula`, put that conversion at the edge:

```python
from wfomc.fol.compat.legacy_syntax import AtomicFormula as LegacyAtomicFormula
```

**Step 3: Add adapter helper only if necessary**

If callers still pass legacy `AtomicFormula`, add a local function:

```python
def _legacy_atom_to_typed(atom_: object) -> Atom:
    from wfomc.fol.compat.legacy_syntax import AtomicFormula

    if isinstance(atom_, Atom):
        return atom_
    if isinstance(atom_, AtomicFormula):
        return atom(atom_.pred, *atom_.args)
    raise TypeError(f"Expected typed or legacy atom, got {type(atom_).__name__}")
```

Keep this helper private and document that it will be removed after legacy parser retirement.

**Step 4: Run tests**

```bash
uv run pytest tests -q -k "ground or propositional"
```

If there are no focused tests, run:

```bash
uv run pytest tests/unit/test_package_structure.py tests/unit/test_problem_parser.py tests/unit/test_engine_compile.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/wfomc/grounding/propositional.py tests
git commit -m "refactor: migrate propositional grounding to typed fol semantics"
```

---

## Task 13: Tighten `cell_graph/formula_ops.py` Typed Boundary

**Files:**

- Modify: `src/wfomc/cell_graph/formula_ops.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Modify: `src/wfomc/cell_graph/components.py`
- Modify: `src/wfomc/cell_graph/utils.py`
- Test: cell graph and reduction tests

**Step 1: Rename formula ops docstring**

Current docstring says this module accepts legacy `QFFormula` during migration. Keep the adapter behavior, but make its responsibility explicit:

```python
"""Formula operations consumed by CellGraph.

The public functions return typed Formula, Atom, and Literal objects. Legacy
formula handling is compatibility-only and must route through
wfomc.fol.compat.legacy_adapter.
"""
```

**Step 2: Update legacy imports**

Replace:

```python
from wfomc.reduction.legacy_adapter import _typed_formula_to_legacy_formula
from wfomc.fol.legacy_syntax import AtomicFormula
```

with:

```python
from wfomc.fol.compat.legacy_adapter import _typed_formula_to_legacy_formula
from wfomc.fol.compat.legacy_syntax import AtomicFormula
```

**Step 3: Use new typed modules**

Ensure imports come from:

```python
from wfomc.fol.literals import Literal, positive_atom
from wfomc.fol.semantics import is_satisfiable, model_literals
from wfomc.fol.rewrite import simplify_boolean
```

**Step 4: Add migration assertion tests**

Add a package structure test:

```python
def test_cell_graph_legacy_access_goes_through_fol_compat():
    source = Path("src/wfomc/cell_graph/formula_ops.py").read_text()
    assert "wfomc.reduction.legacy_adapter" not in source
    assert "wfomc.fol.legacy_syntax" not in source
    assert "wfomc.fol.compat" in source
```

**Step 5: Run tests**

```bash
uv run pytest tests/unit/test_reduction.py tests/unit/test_reduction_backbone.py tests/unit/test_package_structure.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/cell_graph tests/unit/test_package_structure.py
git commit -m "refactor: route cell graph legacy formula handling through fol compat"
```

---

## Task 14: Remove `LegacyFormula` From Typed Public Construction

**Files:**

- Modify: `src/wfomc/fol/formulas.py`
- Modify: `src/wfomc/fol/context.py`
- Modify: `src/wfomc/fol/syntax.py`
- Modify: `src/wfomc/fol/compat/legacy_adapter.py`
- Modify: tests that mention `LegacyFormula`
- Test: `tests/unit/test_fol_syntax.py`
- Test: `tests/unit/test_package_structure.py`

**Step 1: Locate legacy typed wrapper usage**

Run:

```bash
rg "LegacyFormula|FormulaKind\\.LEGACY|\\.legacy\\(|legacy\\(" src tests
```

Classify usages:

- Compatibility adapter usage: move to `fol.compat.legacy_adapter`.
- Typed syntax tests: update to typed-only tests.
- Production typed modules: remove.

**Step 2: Move wrapper, if still needed, into compat**

If conversion needs to keep a wrapper during transition, move `LegacyFormula` to `src/wfomc/fol/compat/legacy_adapter.py` or `src/wfomc/fol/compat/wrappers.py`.

Do not expose it from `wfomc.fol.syntax`.

**Step 3: Remove `legacy()` from `syntax.py.__all__`**

Delete:

```python
"legacy",
```

from `__all__`.

Delete the public function from `syntax.py`.

**Step 4: Update `FormulaKind`**

Only remove `FormulaKind.LEGACY` after no production or test code references it. If removal is too invasive, keep the enum value but mark it deprecated in a comment:

```python
LEGACY = "legacy"  # compat-only; typed modules must not construct this
```

Preferred final state: remove `LEGACY`.

**Step 5: Add boundary test**

```python
def test_typed_syntax_does_not_export_legacy_constructor():
    import wfomc.fol.syntax as syntax

    assert "legacy" not in syntax.__all__
    assert not hasattr(syntax, "legacy")
```

Only add this after all old callers are updated.

**Step 6: Run tests**

```bash
uv run pytest tests/unit/test_fol_syntax.py tests/unit/test_package_structure.py tests/unit/test_reduction.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add src/wfomc/fol tests/unit/test_fol_syntax.py tests/unit/test_package_structure.py tests/unit/test_reduction.py
git commit -m "refactor: remove legacy formula construction from typed syntax"
```

---

## Task 15: Final Internal Import Audit

**Files:**

- Modify as needed after audit.
- Test: full suite or focused suites below.

**Step 1: Run import search**

```bash
rg "wfomc\\.fol\\.legacy_|wfomc\\.fol\\.native_boolean|from wfomc\\.fol import|wfomc\\.reduction\\.legacy_adapter|wfomc\\.fol\\.qf|FormulaKind\\.LEGACY|LegacyFormula" src tests
```

Expected allowed matches:

- `src/wfomc/fol/compat/*`
- old-path shim modules under `src/wfomc/fol/*.py`
- `src/wfomc/reduction/legacy_adapter.py` shim
- compatibility tests that intentionally exercise deprecated import paths

No production typed module should match.

**Step 2: Run package structure tests**

```bash
uv run pytest tests/unit/test_package_structure.py -q
```

Expected: PASS.

**Step 3: Run typed FOL and parser tests**

```bash
uv run pytest tests/unit/test_fol_syntax.py tests/unit/test_qf_boolean.py tests/test_formula_models.py tests/unit/test_problem_parser.py tests/unit/test_normal_form.py -q
```

Expected: PASS.

**Step 4: Run reduction and cell graph tests**

```bash
uv run pytest tests/unit/test_reduction.py tests/unit/test_reduction_backbone.py tests/unit/test_counting_reduction.py tests/unit/test_engine_compile.py -q
```

Expected: PASS.

**Step 5: Run broader smoke test**

```bash
uv run pytest tests -q
```

Expected: PASS or only known unrelated failures. Any failure involving import paths, formula model enumeration, parser output, reduction materialization, or cell graph behavior blocks the migration.

**Step 6: Commit**

```bash
git add src tests
git commit -m "chore: enforce typed fol and compat import boundaries"
```

---

## Final Acceptance Criteria

Typed formula convergence is complete when:

- `src/wfomc/fol/formulas.py` contains only typed AST definitions.
- `src/wfomc/fol/context.py` is the only typed construction engine.
- `src/wfomc/fol/syntax.py` exposes ergonomic typed constructors and no `legacy()` helper.
- `src/wfomc/fol/literals.py` owns `Literal`.
- `src/wfomc/fol/semantics.py` owns `models`, `model_literals`, `is_satisfiable`, `evaluate`, and `to_cnf_clauses`.
- `src/wfomc/fol/qf.py` is either deleted or only a deprecated shim.
- `src/wfomc/fol/builders.py` is deleted or only a deprecated shim.
- `src/wfomc/fol/ast.py` and `src/wfomc/fol/signature.py` are deleted or only deprecated shims.

Compat consolidation is complete when:

- All legacy FOL implementation modules live under `src/wfomc/fol/compat`.
- `src/wfomc/reduction/legacy_adapter.py` is deleted or only a shim to `wfomc.fol.compat.legacy_adapter`.
- `src/wfomc/compat/*` imports legacy FOL through `wfomc.fol.compat.*`, not root `wfomc.fol`.
- Production typed modules do not import `wfomc.fol.compat`.
- Any remaining use of compat in production is isolated to named adapter modules with a clear removal path.
- Root `wfomc.fol` is typed-first: `Formula`, `Predicate`, `Variable`, and `Constant` resolve to typed classes, while legacy names are deprecated compatibility exports or removed.

Recommended final verification:

```bash
uv run pytest tests/unit/test_package_structure.py \
  tests/unit/test_fol_syntax.py \
  tests/unit/test_qf_boolean.py \
  tests/test_formula_models.py \
  tests/unit/test_problem_parser.py \
  tests/unit/test_normal_form.py \
  tests/unit/test_reduction.py \
  tests/unit/test_reduction_backbone.py \
  tests/unit/test_counting_reduction.py \
  tests/unit/test_engine_compile.py -q
```

Then:

```bash
uv run pytest tests -q
```

---

## Suggested Commit Sequence

1. `test: define typed fol and compat boundaries`
2. `refactor: merge fol builders into typed syntax facade`
3. `refactor: move typed fol literals to dedicated module`
4. `refactor: move typed boolean semantics out of qf module`
5. `refactor: move boolean simplification into fol rewrite`
6. `refactor: remove internal fol qf imports`
7. `refactor: move legacy fol modules under fol compat`
8. `refactor: retire fol ast and signature aliases`
9. `refactor: move reduction legacy adapter into fol compat`
10. `refactor: point deprecated compat APIs at fol compat`
11. `refactor: make fol root facade typed first`
12. `refactor: migrate propositional grounding to typed fol semantics`
13. `refactor: route cell graph legacy formula handling through fol compat`
14. `refactor: remove legacy formula construction from typed syntax`
15. `chore: enforce typed fol and compat import boundaries`
