# Typing Convergence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace compatibility-era `object`/`Any` typing with explicit typed interfaces across the active WFOMC pipeline, while deleting typing tests that only pin removed legacy behavior.

**Architecture:** Keep public user APIs ergonomic but typed with `TypeAlias` boundaries. Make `Problem -> C2NormalForm -> ReducedProblems -> MaterializedProblem -> AlgoInput -> WFOMCResult` explicit in type signatures. Do not type around legacy `fol.compat`; either isolate it under `compat` or leave a documented blocker for the later legacy-removal plan.

**Tech Stack:** Python 3.11 dataclasses, `typing`/`collections.abc`, `pytest`, current `wfomc.fol` typed formula IR, `python-flint` numeric values.

---

## Current Typing Problems

1. `engine/orchestration.py` still treats core pipeline values as `object`.
2. `normal_form/c2/norm_form.py` stores formula sections as `object`, even though the active parser produces typed FOL nodes.
3. `algo/materialization.py`, `algo/materialization_view.py`, and `algo/incremental3/counting_state.py` use `object` for predicates, formulas, constants, and weights.
4. `weights.py` and `arithmetic.py` need clear aliases for raw weight values, compiled weight values, symbolic variables, and backend context outputs.
5. `evidence.py` mixes new evidence IR with legacy conversion helpers, causing weak typing in otherwise active code.
6. Structure tests still pin some compatibility modules and should be rewritten once those compatibility paths are no longer part of active typing.

## Non-Goals

- Do not delete `wfomc.fol.compat` in this plan. That requires the separate `cell_graph`/legacy-removal work.
- Do not rewrite algorithm kernels for performance.
- Do not introduce mypy/pyright config yet. This plan first makes local annotations truthful and testable.
- Do not make public DSL builders too narrow; `wfomc.fol.atom(...)`, `forall(...)`, etc. should remain ergonomic.

## Target Type Shape

Add aliases where they reduce noise:

```python
# src/wfomc/fol/types.py or colocated in fol/syntax.py if preferred
FormulaLike = Formula | bool
TermLike = Term | str | int
PredicateLike = Predicate | str
```

Use domain-specific aliases:

```python
# src/wfomc/weights.py
RawWeightValue = int | float | str | Rational | object
CompiledWeightValue = object
WeightMapping = Mapping[object, tuple[RawWeightValue, RawWeightValue]]
CompiledWeightMapping = dict[object, tuple[CompiledWeightValue, CompiledWeightValue]]
```

For core pipeline signatures:

```python
Problem
C2NormalForm
FeatureSet
ReducedProblems
MaterializedProblem
AlgoInput
WFOMCResult
```

---

### Task 1: Add Structure Tests for the New Typing Boundary

**Files:**
- Modify: `tests/unit/test_fol_package_structure.py`

**Step 1: Add tests that active pipeline modules do not use broad public `object` signatures**

Add a focused test that scans only active boundary modules:

```python
def test_active_pipeline_boundaries_are_explicitly_typed():
    files = [
        Path("src/wfomc/engine/orchestration.py"),
        Path("src/wfomc/algo/materialization.py"),
        Path("src/wfomc/algo/materialization_view.py"),
        Path("src/wfomc/algo/incremental3/counting_state.py"),
    ]
    forbidden = (
        "problem: object",
        "normal_form: object",
        "features: object",
        "reduced: object",
        "reduced_problems: object",
    )
    offenders = []
    for path in files:
        source = path.read_text()
        for marker in forbidden:
            if marker in source:
                offenders.append(f"{path}: {marker}")
    assert offenders == []
```

Do not scan `fol.dsl`, `fol.syntax`, `engine.runtime` cache internals, or `compat`; those still intentionally accept dynamic values.

**Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=src:. pytest tests/unit/test_fol_package_structure.py::test_active_pipeline_boundaries_are_explicitly_typed -q
```

Expected: FAIL with offenders in `engine/orchestration.py`, `algo/materialization.py`, and `algo/incremental3/counting_state.py`.

**Step 3: Commit**

```bash
git add tests/unit/test_fol_package_structure.py
git commit -m "test: pin explicit active pipeline typing"
```

---

### Task 2: Type `normal_form.c2` Section Dataclasses

**Files:**
- Modify: `src/wfomc/normal_form/c2/norm_form.py`
- Modify: `src/wfomc/normal_form/c2/normalize.py`
- Modify: `src/wfomc/normal_form/c2/validation.py`
- Test: `tests/unit/test_normal_form.py`
- Test: `tests/unit/test_engine_compile.py`

**Step 1: Introduce typed imports under `TYPE_CHECKING`**

In `norm_form.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wfomc.fol import Atom, Formula, Variable
```

**Step 2: Replace broad dataclass fields**

Change:

```python
count: object
body: object
marker: object | None = None
source: object | None = None
outer_var: object | None = None
counted_var: object | None = None
```

to:

```python
count: int | tuple[int, int]
body: "Formula"
marker: "Atom | None" = None
source: "Formula | None" = None
outer_var: "Variable | None" = None
counted_var: "Variable | None" = None
```

For `PredicateDefinition`:

```python
atom: "Atom"
body: "Formula"
variables: tuple["Variable", ...] = ()
source: "Formula | None" = None
```

For `C2NormalForm`:

```python
universal: "Formula | None" = None
forall_exists: tuple["Formula", ...] = ()
exists: tuple["Formula", ...] = ()
```

**Step 3: Update normalization helper annotations**

In `normalize.py`, replace public/internal signatures where straightforward:

```python
def normalize(sentence: Formula) -> C2NormalForm:
def _to_c2_nnf(formula: Formula) -> Formula:
def _negated_to_c2_nnf(formula: Formula) -> Formula:
def _combine_universal_parts(parts: tuple[Formula, ...]) -> Formula | None:
```

Keep only truly generic helpers as `object`, such as term sorting or cache keys.

**Step 4: Update validation signatures**

In `validation.py`, type section validators:

```python
def _validate_forall_count(forall_count: ForallCountSection, index: int) -> None:
def _validate_count(count: CountSection, index: int) -> None:
def _validate_count_definition(count_definition: CountDefinition, index: int) -> None:
```

**Step 5: Run normal form tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_normal_form.py tests/unit/test_engine_compile.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/normal_form/c2 tests/unit/test_normal_form.py tests/unit/test_engine_compile.py
git commit -m "refactor: type C2 normal form sections"
```

---

### Task 3: Type Engine Orchestration Around Real Pipeline Values

**Files:**
- Modify: `src/wfomc/engine/orchestration.py`
- Modify: `src/wfomc/algo/core.py`
- Test: `tests/unit/engine/test_orchestration.py`
- Test: `tests/unit/test_engine_compile.py`

**Step 1: Import concrete types**

In `engine/orchestration.py`:

```python
from wfomc.algo.core import AlgoInput, AlgoName, AlgoOptions, algo_spec
from wfomc.normal_form import C2NormalForm
from wfomc.problem import Problem
from wfomc.reduction import ReducedProblems
```

**Step 2: Type `CompileArtifacts`**

Change:

```python
parsed_problem: object
normal_form: C2NormalForm
features: FeatureSet
reduced_problem: object | None = None
algo_input: object | None = None
```

to:

```python
parsed_problem: Problem
normal_form: C2NormalForm
features: FeatureSet
reduced_problem: ReducedProblems | None = None
algo_input: AlgoInput | None = None
```

**Step 3: Type public orchestration functions**

Use:

```python
def compile_problem(
    problem: Problem,
    *,
    algo: AlgoName | str = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> CompileArtifacts:
```

Similarly for `solve`, `solve_uncached`, `analyze_problem`.

**Step 4: Type materialization helpers**

Use:

```python
def _materialize_with_optional_context(
    materialize: Callable[..., AlgoInput],
    reduced: ReducedProblems,
    ctx: MaterializationContext,
) -> AlgoInput:
```

`_materialize_accepts_context` can remain reflective:

```python
def _materialize_accepts_context(materialize: Callable[..., object]) -> bool:
```

**Step 5: Keep cache key helpers flexible**

Do not over-type `_object_key`, `_mapping_key`, `_problem_key`; they deliberately serialize arbitrary values. Add a comment:

```python
# Cache key helpers intentionally accept object because they serialize mixed
# dataclass/enums/FLINT values at runtime boundaries.
```

**Step 6: Run engine tests**

```bash
PYTHONPATH=src:. pytest tests/unit/engine/test_orchestration.py tests/unit/test_engine_compile.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add src/wfomc/engine/orchestration.py src/wfomc/algo/core.py tests/unit/engine/test_orchestration.py tests/unit/test_engine_compile.py
git commit -m "refactor: type engine orchestration pipeline"
```

---

### Task 4: Type Algorithm Specs and Materializer Function Signatures

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: all `src/wfomc/algo/*/spec.py`
- Test: `tests/unit/test_algo_planning.py`
- Test: `tests/unit/engine/test_orchestration.py`

**Step 1: Define precise callable aliases**

In `algo/core.py`:

```python
if TYPE_CHECKING:
    from wfomc.engine.orchestration import MaterializationContext
    from wfomc.normal_form import C2NormalForm
    from wfomc.problem import Problem
    from wfomc.reduction import ReducedProblems

ReduceFn = Callable[
    ["C2NormalForm", "Problem"],
    "ReducedProblems",
]

MaterializeFn = Callable[
    ["ReducedProblems", "MaterializationContext"],
    AlgoInput,
]

SolveFn = Callable[[AlgoInput, RuntimeContext | None], WFOMCResult]
```

Because current reduce/materialize functions accept keyword-only `features/options`, use a `Protocol` instead of plain `Callable` if needed:

```python
class ReduceFn(Protocol):
    def __call__(
        self,
        normal_form: "C2NormalForm",
        problem: "Problem",
        *,
        features: FeatureSet,
        options: AlgoOptions,
    ) -> "ReducedProblems": ...
```

**Step 2: Update `AlgoSpec`**

Change fields from broad callable/object types to the aliases/protocols above.

**Step 3: Update materializer builders**

Type:

```python
def backend_materializer(
    backend_reduce: Callable[..., MaterializedProblem],
    build_input: Callable[[MaterializedProblem], AlgoInput],
) -> MaterializeFn:
```

For counting:

```python
def counting_dp_materializer(
    build_input: Callable[..., AlgoInput],
) -> MaterializeFn:
```

**Step 4: Run planning tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_algo_planning.py tests/unit/engine/test_orchestration.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/wfomc/algo tests/unit/test_algo_planning.py tests/unit/engine/test_orchestration.py
git commit -m "refactor: type algorithm spec contracts"
```

---

### Task 5: Type MaterializedProblem and Materialization View

**Files:**
- Modify: `src/wfomc/algo/materialization.py`
- Modify: `src/wfomc/algo/materialization_view.py`
- Test: `tests/unit/test_cell_graph_inputs.py`
- Test: `tests/unit/test_cell_graph_construction.py`
- Test: `tests/unit/test_counting_reduction.py`

**Step 1: Import typed FOL symbols**

Use:

```python
from wfomc.fol import Atom, Formula, Predicate
```

Under `TYPE_CHECKING`, import:

```python
from wfomc.normal_form import C2NormalForm
from wfomc.problem import Problem
```

**Step 2: Type materialized fields**

Change:

```python
domain: frozenset[object]
qf_formula: object | None
weights: tuple[tuple[object, tuple[RingElement, RingElement]], ...]
unary_evidence: tuple[object, ...]
leq_predicate: object | None
predecessor_predicates: dict[int, object] | None
```

to:

```python
domain: frozenset[Constant]
qf_formula: Formula | None
weights: tuple[tuple[Predicate, tuple[RingElement, RingElement]], ...]
unary_evidence: tuple[Literal, ...]
leq_predicate: Predicate | None = None
predecessor_predicates: dict[int, Predicate] | None = None
circular_predecessor_predicate: Predicate | None = None
```

If `Constant` is too narrow because legacy constants still appear in tests, use:

```python
DomainElement = Constant | object
```

but prefer typed `Constant` in active code.

**Step 3: Type materialization state**

Use the same field types as `MaterializedProblem`.

**Step 4: Type materialization view**

In `materialization_view.py`:

```python
class MaterializationOrderMetadata:
    leq_predicate: Predicate | None = None
    predecessor_predicates: dict[int, Predicate] | None = None
    circular_predecessor_predicate: Predicate | None = None

class MaterializationSections:
    universal_body: Formula | None
    existential_formulas: tuple[Formula, ...] = ()
    forall_counts: tuple[ForallCountSection, ...] = ()
    counts: tuple[CountSection, ...] = ()
```

**Step 5: Run cell graph/materialization tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_cell_graph_inputs.py tests/unit/test_cell_graph_construction.py tests/unit/test_counting_reduction.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/algo/materialization.py src/wfomc/algo/materialization_view.py tests/unit/test_cell_graph_inputs.py tests/unit/test_cell_graph_construction.py tests/unit/test_counting_reduction.py
git commit -m "refactor: type algorithm materialization artifacts"
```

---

### Task 6: Type Incremental3 Counting State

**Files:**
- Modify: `src/wfomc/algo/incremental3/counting_state.py`
- Modify: `src/wfomc/algo/incremental3/input.py`
- Modify: `tests/test_incremental3_regressions.py`
- Test: `tests/unit/test_incremental3_counting_native.py`

**Step 1: Import typed FOL and normal-form types**

```python
from wfomc.fol import Literal, Predicate
from wfomc.normal_form import C2NormalForm, CountSection, ForallCountSection
```

**Step 2: Type `CountingState`**

Change:

```python
ext_preds: tuple[object, ...]
cnt_preds: tuple[object, ...]
cnt_remainder: tuple[object, ...]
binary_evidence: tuple[frozenset, ...]
```

to:

```python
ext_preds: tuple[Predicate, ...]
cnt_preds: tuple[Predicate, ...]
cnt_remainder: tuple[int | None, ...]
binary_evidence: tuple[frozenset[Literal], ...]
```

**Step 3: Type `UnaryCardinalityMasks`**

Change list fields to:

```python
self.mod_constraints: list[tuple[Predicate, int, int]]
self.eq_constraints: list[tuple[Predicate, int]]
self.le_constraints: list[tuple[Predicate, int]]
```

**Step 4: Type constructors**

```python
def build_counting_state_for_normal_form(
    normal_form: C2NormalForm,
) -> tuple[CountingState, UnaryCardinalityMasks]:

def _count_sections(normal_form: C2NormalForm) -> tuple[CountSection, ...]:
def _forall_count_sections(normal_form: C2NormalForm) -> tuple[ForallCountSection, ...]:
```

**Step 5: Run incremental3 tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_incremental3_counting_native.py tests/test_incremental3_regressions.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/wfomc/algo/incremental3 tests/unit/test_incremental3_counting_native.py tests/test_incremental3_regressions.py
git commit -m "refactor: type incremental3 counting state"
```

---

### Task 7: Add Weight and Arithmetic Type Aliases

**Files:**
- Modify: `src/wfomc/weights.py`
- Modify: `src/wfomc/arithmetic.py`
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/algo/materialization.py`
- Test: `tests/unit/test_weights.py`
- Test: `tests/unit/test_arithmetic_context.py`

**Step 1: Add aliases in `weights.py`**

```python
from typing import TypeAlias

RawWeightValue: TypeAlias = int | float | str | Rational | object
CompiledWeightValue: TypeAlias = object
RawWeightMapping: TypeAlias = Mapping[object, tuple[RawWeightValue, RawWeightValue]]
CompiledWeightMapping: TypeAlias = dict[object, tuple[CompiledWeightValue, CompiledWeightValue]]
```

This still permits FLINT values while removing repeated broad mapping spellings.

**Step 2: Update public function signatures**

```python
def compile_weight_mapping(
    weights: RawWeightMapping,
    plan: WeightPlan,
) -> CompiledWeightMapping:
```

```python
def convert_weight_mapping_to_ring_elements(
    weights: RawWeightMapping,
) -> CompiledWeightMapping:
```

**Step 3: Type `ArithmeticContext` return helpers where possible**

Use concrete FLINT return types for `zero`, `one`, `coerce`, `symbol`, etc. If a method returns backend-dependent values, use `CompiledWeightValue`.

**Step 4: Run weight tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_weights.py tests/unit/test_arithmetic_context.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/wfomc/weights.py src/wfomc/arithmetic.py src/wfomc/algo/core.py src/wfomc/algo/materialization.py tests/unit/test_weights.py tests/unit/test_arithmetic_context.py
git commit -m "refactor: add explicit weight typing aliases"
```

---

### Task 8: Split New Evidence Types From Legacy Evidence Adapters

**Files:**
- Modify: `src/wfomc/evidence.py`
- Create: `src/wfomc/compat/evidence.py`
- Modify: `src/wfomc/compat/__init__.py`
- Modify: callers found by `rg "unary_evidence_from_legacy|evidence_plan_from_legacy" src tests`
- Test: `tests/unit/test_evidence_planning.py`
- Test: `tests/unit/test_evidence_execution.py`
- Test: `tests/unit/test_fol_package_structure.py`

**Step 1: Create compat evidence module**

Move these functions from `evidence.py` to `compat/evidence.py`:

```python
unary_evidence_from_legacy
evidence_plan_from_legacy
_strategy_from_legacy_plan
```

Keep `evidence.py` focused on canonical evidence IR:

```python
EvidenceStrategy
UnaryLiteral
GroundUnaryLiteral
UnaryEvidence
EvidenceProfile
EvidencePartition
ProfileCapacityConstraint
EvidencePlan
CellEvidenceAllocation
```

**Step 2: Update imports**

Run:

```bash
rg "unary_evidence_from_legacy|evidence_plan_from_legacy" src tests
```

Change any legacy caller to:

```python
from wfomc.compat.evidence import unary_evidence_from_legacy
```

Do not import compat from active reduction/materialization modules.

**Step 3: Update structure tests**

Change `test_typed_reduction_modules_do_not_import_legacy_fol` to also assert:

```python
"unary_evidence_from_legacy",
"evidence_plan_from_legacy",
```

are absent from active modules.

**Step 4: Run evidence tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_evidence_planning.py tests/unit/test_evidence_execution.py tests/unit/test_fol_package_structure.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/wfomc/evidence.py src/wfomc/compat/evidence.py src/wfomc/compat/__init__.py tests/unit/test_evidence_planning.py tests/unit/test_evidence_execution.py tests/unit/test_fol_package_structure.py
git commit -m "refactor: move legacy evidence adapters to compat"
```

---

### Task 9: Remove Obsolete Typing/Legacy Structure Tests

**Files:**
- Modify: `tests/unit/test_fol_package_structure.py`
- Modify: tests found by `rg "fol.compat|legacy_syntax|legacy_sc2|legacy_utils" tests`

**Step 1: Delete tests that require legacy to exist**

Remove or invert these tests when their corresponding modules are no longer active requirements:

```python
test_legacy_fol_modules_live_under_fol_compat
test_fol_compat_import_does_not_eager_load_legacy_sc2
test_legacy_fol_parser_lives_in_compat
test_legacy_solver_facade_lives_only_in_compat
test_cell_graph_legacy_access_goes_through_fol_compat
```

For this typing plan, only delete tests whose modules were actually moved/deleted in previous tasks. Do not delete tests for `fol.compat` until the cell graph legacy-removal plan lands.

**Step 2: Keep useful typing tests**

Keep:

```python
test_public_api_uses_explicit_facade_types
test_active_pipeline_boundaries_are_explicitly_typed
test_typed_fol_modules_do_not_import_compat_modules
test_typed_reduction_modules_do_not_import_legacy_fol
```

**Step 3: Run structure tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_fol_package_structure.py -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tests/unit/test_fol_package_structure.py
git commit -m "test: remove obsolete legacy typing structure checks"
```

---

### Task 10: Final Type-Driven Cleanup Pass

**Files:**
- Modify: any active file found by the commands below
- Test: full relevant suite

**Step 1: Scan remaining weak typing in active modules**

Run:

```bash
rg -n "\\bobject\\b|Any|type: ignore|cast\\(" src/wfomc \
  --glob '!src/wfomc/compat/**' \
  --glob '!src/wfomc/fol/compat/**'
```

Classify each hit:

- keep if it is a DSL ergonomic boundary;
- keep if it is a runtime cache/key serialization boundary;
- replace if it is pipeline data (`Problem`, `C2NormalForm`, `Formula`, `Predicate`, `MaterializedProblem`, `AlgoInput`, etc.);
- move to compat if it only exists for legacy conversion.

**Step 2: Add comments only for intentional dynamic boundaries**

Example:

```python
# Intentionally dynamic: runtime cache keys may contain dataclasses, enums,
# FLINT values, formulas, and domain constants.
def _object_key(value: object) -> object:
    ...
```

Do not add comments for every `object`; only for boundary helpers where a future reader might otherwise tighten it incorrectly.

**Step 3: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit tests/test_incremental3_regressions.py -q
PYTHONPATH=src:. pytest tests/unary_evidence/test_algorithm_matrix.py -q
```

Expected:

- unit/regression suite passes;
- unary evidence matrix keeps the intentionally skipped tests until evidence strategy behavior is revisited.

**Step 4: Commit**

```bash
git add src tests
git commit -m "refactor: finish active pipeline typing cleanup"
```

---

## Acceptance Criteria

1. `wfomc.api` and `wfomc.__init__` expose explicit types only; no `object` facade signatures.
2. `engine/orchestration.py` signatures use `Problem`, `C2NormalForm`, `FeatureSet`, `ReducedProblems`, `AlgoInput`, and `WFOMCResult`.
3. `normal_form.c2` dataclasses use typed FOL nodes instead of raw `object`.
4. `algo.materialization` exposes a typed `MaterializedProblem`.
5. `incremental3.counting_state` uses typed predicates/literals/normal-form sections.
6. Legacy evidence conversion helpers are not in `wfomc.evidence`.
7. Remaining `object` annotations are documented dynamic boundaries or public DSL ergonomics, not accidental compatibility residue.
8. Structure tests no longer require deleted legacy modules.
9. These pass:

```bash
PYTHONPATH=src:. pytest tests/unit tests/test_incremental3_regressions.py -q
PYTHONPATH=src:. pytest tests/unary_evidence/test_algorithm_matrix.py -q
```

## Known Blocker Outside This Plan

`wfomc.fol.compat` cannot be deleted until `wfomc.cell_graph` and `wfomc.grounding.propositional` no longer use legacy formula/Boolean objects. That should be a separate plan:

- typed `CellGraph`;
- remove legacy branch in `cell_graph.formula_ops`;
- remove legacy branch in `grounding.propositional`;
- delete `wfomc.compat`;
- delete `wfomc.fol.compat`;
- rewrite/delete legacy tests.
