# Boolean WMC Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current mixed FOL syntax/SymPy Boolean layer with a clean immutable core FOL AST, rewrite pipeline, internal Boolean DAG, and projected-WMC backend, using bitsets as compact representation rather than as a brute-force solver strategy.

**Architecture:** First introduce a new core syntax under `wfomc.fol.core` and make the existing `syntax.py` a compatibility facade/adapter. Rewrite passes and the Boolean backend consume the core AST only. The first production backend will use hash-consed Boolean DAG nodes, bitset assumptions/projections, simplification, and cached DPLL/projected WMC; the old SymPy behavior remains available only as a reference oracle during migration.

**Tech Stack:** Python 3.11, dataclasses with slots/frozen nodes, python-flint ring elements, pytest, existing perf harness under `src/wfomc/perf`.

---

## Relationship To Existing Plan

This plan supersedes the performance-sensitive parts of `docs/plans/2026-06-22-wfomc-dsl-performance-architecture.md`, especially Task 5 onward. The previous parser cache, logging cleanup, cardinality validation, perf harness, and pynauty fallback remain useful. The partial immutable traversal work can be folded into the new core syntax task.

## Non-Functional Requirements

### Correctness
- Existing `uv run pytest tests/wfomc_test.py` must pass after every migration phase.
- Compatibility adapter tests must prove old `syntax.py` formulas lower to the new core AST without changing meaning.
- SymPy reference tests must agree with the new Boolean backend for atoms, substitution, satisfiability, models for small formulas, WMC, and projected WMC.
- `QFFormula.models()` remains available during migration, but it is not a hot-path target.

### Performance
- Production `wfomc()` path should not call SymPy Boolean functions after Task 5.
- `CellGraph._build_cells()` and `_build_two_tables()` should move from full model materialization to projected aggregation.
- Perf harness should compare baseline against parser, context, cell graph, solve, and total timings.

### Maintainability
- New rewrite passes must consume the new core AST, not old `syntax.py` objects.
- `syntax.py` remains a compatibility facade only after the core AST exists.
- New core code lives under `src/wfomc/fol/core/` and `src/wfomc/fol/boolean/`.
- The WMC interface must allow future ROBDD/d-DNNF replacement without changing `CellGraph` again.

## Target Module Layout

```text
src/wfomc/fol/
  syntax.py                    # compatibility facade/adapters during migration
  core/
    __init__.py
    terms.py                   # Pred/Var/Const or core equivalents
    formula.py                 # immutable FOL/CoreBool/CoreQuantifier AST
    builders.py                # explicit surface builders, no hidden rewrites
    lower.py                   # old syntax -> core adapter
    visitors.py                # traversal/rewrite primitives
  boolean_algebra.py           # compatibility facade
  sympy_boolean_algebra.py     # reference oracle
  boolean/
    __init__.py
    dag.py                     # BoolConst/Atom/Not/And/Or/manager
    bitset.py                  # assignment/projection masks
    simplify.py                # condition/simplify operations
    dpll.py                    # cached DPLL WMC
    projection.py              # projected WMC aggregation
    facade.py                  # old boolean_algebra-compatible API
tests/fol/
  test_boolean_dag.py
  test_boolean_backend_equivalence.py
  test_core_syntax.py
  test_core_lowering.py
  test_projected_wmc.py
```

## Task 1: Introduce Core FOL Syntax

**Files:**
- Create: `src/wfomc/fol/core/__init__.py`
- Create: `src/wfomc/fol/core/terms.py`
- Create: `src/wfomc/fol/core/formula.py`
- Create: `src/wfomc/fol/core/builders.py`
- Test: `tests/fol/test_core_syntax.py`

**Step 1: Define immutable terms**

Create frozen, slotted core terms:

```python
CorePred(name: str, arity: int)
CoreVar(name: str)
CoreConst(name: str)
```

**Step 2: Define immutable formula nodes**

Create frozen, slotted formula nodes:

```python
CoreAtom(pred: CorePred, args: tuple[CoreTerm, ...])
CoreNot(child: CoreFormula)
CoreAnd(children: tuple[CoreFormula, ...])
CoreOr(children: tuple[CoreFormula, ...])
CoreImplies(left: CoreFormula, right: CoreFormula)
CoreEquivalent(left: CoreFormula, right: CoreFormula)
CoreForall(var: CoreVar, body: CoreFormula)
CoreExists(var: CoreVar, body: CoreFormula)
CoreCounting(var: CoreVar, comparator: str, count: int, body: CoreFormula)
CoreTop()
CoreBottom()
```

**Step 3: Make boolean nodes n-ary**

`CoreAnd` and `CoreOr` should store n-ary tuples. Builders can flatten nested nodes, but they must not perform semantic rewrites such as pushing formulas through quantifiers.

**Step 4: Test immutability and shape**

Run:

```bash
uv run pytest tests/fol/test_core_syntax.py -q
```

Expected: PASS.

## Task 2: Add Core Traversal And Rewrite Primitives

**Files:**
- Create: `src/wfomc/fol/core/visitors.py`
- Test: `tests/fol/test_core_syntax.py`

**Step 1: Implement traversal**

Add:

```python
pre_order(node, fn)
post_order(node, fn)
```

These must return new nodes and never mutate an input node.

**Step 2: Implement generic children replacement**

Add a helper that reconstructs a node from new children.

**Step 3: Test no mutation**

Verify a rewrite pass changes the returned formula but leaves the source formula unchanged.

## Task 3: Add Old Syntax To Core Lowering

**Files:**
- Create: `src/wfomc/fol/core/lower.py`
- Test: `tests/fol/test_core_lowering.py`

**Step 1: Lower old terms and predicates**

Convert old `Pred`, `Var`, and `Const` to core terms.

**Step 2: Lower old formulas**

Convert old `AtomicFormula`, `QFFormula`, `QuantifiedFormula`, `Negation`, and `BinaryFormula` into core AST. For old `QFFormula` that wraps backend expressions, use its atoms and backend structure until `boolean_algebra` is replaced.

**Step 3: Test representative parser outputs**

Parse formulas with the old parser and lower them to core syntax.

**Step 4: Verify**

Run:

```bash
uv run pytest tests/fol/test_core_lowering.py tests/wfomc_test.py -q
```

Expected: PASS.

## Task 4: Port Rewrite Passes To Core AST

**Files:**
- Create: `src/wfomc/fol/core/rewrites.py`
- Modify: `src/wfomc/fol/sc2.py`
- Test: `tests/fol/test_core_lowering.py`

**Step 1: Implement explicit passes**

Add:

```python
eliminate_implication_equivalence(core_formula)
push_negation(core_formula)
distribute_quantifiers(core_formula)
rename_variables(core_formula)
collect_sc2(core_formula)
```

**Step 2: Keep old `to_sc2` as adapter**

Old `to_sc2(formula)` should:

```text
old syntax -> core AST -> core rewrite pipeline -> old/current SC2 output
```

**Step 3: Test no hidden constructor rewrites**

Verify builders do not push QF formulas through quantifiers automatically.

**Step 4: Run verification**

Run:

```bash
uv run pytest tests/fol tests/wfomc_test.py -q
```

Expected: PASS.

## Task 5: Preserve SymPy As Reference Oracle

**Files:**
- Create: `src/wfomc/fol/sympy_boolean_algebra.py`
- Modify: `src/wfomc/fol/boolean_algebra.py`
- Test: `tests/fol/test_boolean_backend_equivalence.py`

**Step 1: Move current implementation**

Move the current SymPy implementation from `boolean_algebra.py` into `sympy_boolean_algebra.py` unchanged.

**Step 2: Keep compatibility facade**

Make `boolean_algebra.py` re-export the SymPy implementation for now:

```python
from .sympy_boolean_algebra import *
```

**Step 3: Add oracle smoke tests**

Add tests that import both modules and verify current atoms/models behavior on small formulas.

**Step 4: Run verification**

Run:

```bash
uv run pytest tests/fol/test_boolean_backend_equivalence.py tests/wfomc_test.py -q
```

Expected: PASS.

## Task 6: Add Boolean DAG Core

**Files:**
- Create: `src/wfomc/fol/boolean/__init__.py`
- Create: `src/wfomc/fol/boolean/dag.py`
- Test: `tests/fol/test_boolean_dag.py`

**Step 1: Define node types**

Create frozen, slotted nodes:

```python
BoolConst(value: bool)
AtomNode(atom_id: int)
NotNode(child: int)
AndNode(children: tuple[int, ...])
OrNode(children: tuple[int, ...])
```

Use integer node ids managed by `BoolDag`.

**Step 2: Add hash-consing**

`BoolDag` should return the same node id for structurally identical expressions.

**Step 3: Normalize n-ary nodes**

Flatten nested `And`/`Or`, remove duplicate children, and short-circuit with constants.

**Step 4: Test**

Run:

```bash
uv run pytest tests/fol/test_boolean_dag.py -q
```

Expected: PASS.

## Task 7: Add Bitset Assignment And Evaluation

**Files:**
- Create: `src/wfomc/fol/boolean/bitset.py`
- Modify: `src/wfomc/fol/boolean/dag.py`
- Test: `tests/fol/test_boolean_dag.py`

**Step 1: Define masks**

Represent assumptions as:

```python
true_mask: int
false_mask: int
```

**Step 2: Add evaluation**

Add:

```python
evaluate(node_id, true_mask) -> bool
```

for complete assignments, and conflict checks for partial assumptions.

**Step 3: Add atom support queries**

Add:

```python
support(node_id) -> frozenset[int]
```

with memoization.

**Step 4: Test**

Run:

```bash
uv run pytest tests/fol/test_boolean_dag.py -q
```

Expected: PASS.

## Task 8: Add Condition And Simplify

**Files:**
- Create: `src/wfomc/fol/boolean/simplify.py`
- Test: `tests/fol/test_boolean_dag.py`

**Step 1: Implement condition**

Add:

```python
condition(node_id, atom_id, value) -> node_id
```

that substitutes one atom and simplifies.

**Step 2: Implement assumptions simplify**

Add:

```python
simplify_under(node_id, true_mask, false_mask) -> node_id
```

**Step 3: Test short-circuit behavior**

Verify `And(False, X) -> False`, `Or(True, X) -> True`, double negation, and duplicate removal.

## Task 9: Add Cached DPLL WMC

**Files:**
- Create: `src/wfomc/fol/boolean/dpll.py`
- Test: `tests/fol/test_projected_wmc.py`

**Step 1: Implement variable selection**

Start with a deterministic heuristic:

```python
choose smallest atom id from support(node_id)
```

**Step 2: Implement WMC**

Add:

```python
wmc(node_id, weights, true_mask=0, false_mask=0) -> RingElement
```

Use recurrence:

```text
w_pos(a) * WMC(F[a=True]) + w_neg(a) * WMC(F[a=False])
```

**Step 3: Cache residuals**

Cache by simplified node id and relevant assumptions.

**Step 4: Test against brute force for small formulas**

Only tests may brute force all assignments.

## Task 10: Add Projected WMC

**Files:**
- Create: `src/wfomc/fol/boolean/projection.py`
- Test: `tests/fol/test_projected_wmc.py`

**Step 1: Define output**

Return:

```python
dict[int, RingElement]
```

where key is a projection mask over selected atom ids.

**Step 2: Implement projected recursion**

At branch time, accumulate projection assignment only when the branched atom is in the projection set.

**Step 3: Test**

Verify projected sums equal ordinary WMC when summed over all projection keys.

## Task 11: Implement New Compatibility Facade

**Files:**
- Create: `src/wfomc/fol/boolean/facade.py`
- Modify: `src/wfomc/fol/boolean_algebra.py`
- Test: `tests/fol/test_boolean_backend_equivalence.py`

**Step 1: Reproduce old function names**

Implement:

```python
get_symbol(atom)
get_atom(symbol)
Equivalent(*args)
And(*args)
Or(*args)
Not(*args)
Implies(*args)
get_atoms(expr)
get_models(expr)
substitute(expr, mapping)
simplify(expr)
satisfiable(expr)
```

using the Boolean DAG backend.

**Step 2: Add compatibility model adapter**

`get_models(expr)` may use DPLL enumeration or a small brute-force adapter for compatibility only. It must not be used by new `CellGraph` hot paths.

**Step 3: Switch default facade**

Make `boolean_algebra.py` export from `boolean.facade`.

**Step 4: Verify**

Run:

```bash
uv run pytest tests/fol/test_boolean_backend_equivalence.py tests/wfomc_test.py -q
```

Expected: PASS.

## Task 8: Add CellGraph Projected-WMC Integration

**Files:**
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Modify: `src/wfomc/cell_graph/components.py`
- Test: `tests/cell_graph/test_projected_cell_graph.py`

**Step 1: Add builder path**

Add a new internal path that computes:

```text
cell_weights
two_table_weights
```

from projected WMC rather than full model materialization.

**Step 2: Keep old path behind tests**

Use old `get_models()` path as reference in tests only.

**Step 3: Compare tables**

For representative models, assert old and new cell/two-table weights match exactly.

**Step 4: Perf check**

Run:

```bash
uv run pytest tests/perf -m perf -q
```

Expected: PASS, with cell graph timing recorded.

## Task 9: Remove SymPy From Production Path

**Files:**
- Modify: `src/wfomc/fol/boolean_algebra.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Modify: `pyproject.toml`
- Test: full suite

**Step 1: Add guard test**

Add a test that monkeypatches the SymPy oracle module and proves `wfomc()` does not import/use it in production paths.

**Step 2: Move SymPy to dev dependency**

Move `sympy` from project dependencies to dev dependencies if reference tests still need it.

**Step 3: Verify**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

## Task 10: Document Backend Contract

**Files:**
- Create: `docs/boolean-backend.md`
- Modify: `README.md`

**Step 1: Document public compatibility**

Explain which old APIs remain compatibility-only.

**Step 2: Document production APIs**

Document `wmc` and `projected_wmc` as the intended hot-path interfaces.

**Step 3: Document future backend options**

Mention ROBDD/d-DNNF as possible implementations behind the same interface.

## Migration Notes

- Do not rewrite the parser first.
- Do not rewrite all of `syntax.py` first.
- Do not make bitset brute force the final strategy.
- Do introduce the Boolean backend behind an adapter, then move `CellGraph` onto projected WMC.
- Keep SymPy as an oracle only until the new backend has enough equivalence coverage.
