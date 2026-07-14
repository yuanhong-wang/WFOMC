# Accessor Boundary Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete the active-pipeline duck-typing/accessor layer and make core WFOMC classes export the behavior currently hidden behind `getattr` helpers.

**Architecture:** The active path should be explicit: `Problem -> C2NormalForm -> FeatureSet -> ReducedProblems -> MaterializedProblem -> AlgoInput`. Business objects own their own cache keys, predicates, counts, weights, and emptiness checks. Compatibility-era shape probing is deleted from active packages instead of moved around.

**Tech Stack:** Python 3.11 dataclasses, typed `wfomc.fol` formulas, `wfomc.problem.Problem`, `wfomc.normal_form.C2NormalForm`, `wfomc.reduction.ReducedProblems`, pytest.

---

## Current Problem

`src/wfomc/reduction/_accessors.py` is only the visible symptom. The same pattern exists in engine, reduction, materialization, cell graph cache, evidence, weights, and C2 normalization:

```text
getattr(obj, "field", fallback)
object-typed helper
legacy field probe
private key builder for another class
```

This makes the real package boundaries unclear. It also lets legacy formula/normal-form shapes leak back into the new path.

## Scope

Delete or replace active-pipeline business `getattr` usage in:

- `src/wfomc/reduction/_accessors.py`
- `src/wfomc/reduction/normal_form.py`
- `src/wfomc/reduction/core.py`
- `src/wfomc/reduction/evidence_planning.py`
- `src/wfomc/reduction/counting.py`
- `src/wfomc/engine/features.py`
- `src/wfomc/engine/orchestration.py`
- `src/wfomc/algo/core.py`
- `src/wfomc/algo/materialization.py`
- `src/wfomc/algo/materialization_view.py`
- `src/wfomc/algo/incremental3/counting_state.py`
- `src/wfomc/algo/cell_graph/cache.py`
- `src/wfomc/algo/cell_graph/components.py`
- `src/wfomc/algo/cell_graph/inputs.py`
- `src/wfomc/evidence.py`
- `src/wfomc/weights.py`
- `src/wfomc/normal_form/c2/normalize.py`
- `src/wfomc/normal_form/c2/validation.py`

Allowed remaining dynamic access:

- package lazy import `__getattr__` in `__init__.py`;
- runtime/plugin probing, for example tail-signature engine loading;
- external numeric-library probing around FLINT/SymPy values;
- generic cache storage APIs where the value is intentionally opaque.

## Non-Goals

- Do not preserve legacy formula support in active paths.
- Do not move deleted behavior into another `compat` package unless a test still explicitly exercises a public compatibility API.
- Do not redesign algorithms or arithmetic semantics.
- Do not add mypy/pyright yet; this is a code boundary cleanup first.

---

## Target Class Responsibilities

### `Problem`

Modify `src/wfomc/problem.py`.

Add typed imports under `TYPE_CHECKING` and make the class expose its own boundary behavior:

```python
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wfomc.fol import Formula


@dataclass(frozen=True)
class Problem:
    sentence: "Formula"
    domain: frozenset[object] = frozenset()
    weights: Mapping[object, tuple[object, object]] = field(default_factory=dict)
    cardinality_constraints: CardinalityConstraints = field(default_factory=CardinalityConstraints)
    unary_evidence: UnaryEvidence = field(default_factory=UnaryEvidence)
    options: Mapping[str, object] = field(default_factory=dict)
    profile_capacity_constraint: ProfileCapacityConstraint | None = None

    @property
    def has_unary_evidence(self) -> bool:
        return not self.unary_evidence.is_empty

    @property
    def has_profile_capacity_constraint(self) -> bool:
        return (
            self.profile_capacity_constraint is not None
            and not self.profile_capacity_constraint.is_empty
        )

    @property
    def has_cardinality_constraints(self) -> bool:
        return not self.cardinality_constraints.is_empty

    def weight_items(self) -> tuple[tuple[object, object, object], ...]:
        return tuple(
            (predicate, positive, negative)
            for predicate, (positive, negative) in self.weights.items()
        )

    def cache_key_parts(self, *, include_domain_size: bool) -> tuple[object, ...]:
        domain_part = len(self.domain) if include_domain_size else tuple(sorted(map(str, self.domain)))
        return (
            repr(self.sentence),
            domain_part,
            tuple(sorted((repr(k), repr(v)) for k, v in self.weights.items())),
            self.unary_evidence.cache_key_parts(),
            self.cardinality_constraints.cache_key_parts(),
            None
            if self.profile_capacity_constraint is None
            else self.profile_capacity_constraint.cache_key_parts(),
            tuple(sorted((str(k), repr(v)) for k, v in self.options.items())),
        )
```

If `CardinalityConstraints.cache_key_parts()` does not exist, add it in `src/wfomc/cardinality.py` rather than keeping `_object_key` fallbacks in engine.

### `C2NormalForm`

Modify `src/wfomc/normal_form/c2/norm_form.py`.

Add methods that replace `reduction._accessors` and `reduction.normal_form`:

```python
@dataclass(frozen=True)
class C2NormalForm:
    ...

    @property
    def has_counting(self) -> bool:
        return bool(self.counts or self.forall_counts or self.count_definitions)

    @property
    def has_forall_counting(self) -> bool:
        return bool(self.forall_counts)

    def all_count_sections(self) -> tuple[CountSection | ForallCountSection, ...]:
        return (*self.counts, *self.forall_counts)

    @property
    def has_rewritten_count_body(self) -> bool:
        return any(section.source != section.body for section in self.all_count_sections())

    def to_sentence(self) -> "Formula | None":
        ...
```

`to_sentence()` should replace `src/wfomc/reduction/normal_form.py`. It should use typed `fol.conjunction`, `fol.count`, and `fol.forall`, not sectioned legacy formulas.

### `ReducedProblems`

Modify `src/wfomc/reduction/core.py`.

Add:

```python
@dataclass(frozen=True)
class ReducedProblems:
    ...

    @classmethod
    def single(...):
        ...

    def expect_single_branch(self) -> ReducedBranch:
        if len(self.branches) != 1:
            raise ValueError("Expected exactly one reduced branch")
        return self.branches[0]
```

Then remove all `getattr(reduced_problems, "branches", ())` calls.

### `AlgoOptions`

Modify `src/wfomc/algo/core.py`.

Add:

```python
@dataclass(frozen=True)
class AlgoOptions:
    ...

    def resolved_evidence_strategy(self) -> EvidenceStrategy | None:
        return _evidence_strategy(self.evidence_strategy)
```

Then delete `options_evidence_strategy()` from `_accessors.py`.

### Evidence Types

Modify `src/wfomc/evidence.py`.

Add cache-key ownership:

```python
@dataclass(frozen=True)
class UnaryLiteral:
    ...

    def cache_key_parts(self) -> tuple[object, bool]:
        return (_predicate_key(self.predicate), self.positive)


@dataclass(frozen=True)
class GroundUnaryLiteral:
    ...

    def cache_key_parts(self) -> tuple[object, str, bool]:
        return (_predicate_key(self.predicate), str(self.constant), self.positive)


@dataclass(frozen=True)
class UnaryEvidence:
    ...

    def cache_key_parts(self) -> tuple[object, ...]:
        return tuple(sorted(literal.cache_key_parts() for literal in self.literals))


@dataclass(frozen=True)
class EvidenceProfile:
    ...

    def cache_key_parts(self) -> tuple[int, tuple[object, ...]]:
        return (
            self.size,
            tuple(sorted(literal.cache_key_parts() for literal in self.literals)),
        )


@dataclass(frozen=True)
class EvidencePartition:
    ...

    def cache_key_parts(self) -> tuple[int, object, tuple[object, ...]]:
        return (
            self.domain_size,
            repr(self.assignment_count),
            tuple(profile.cache_key_parts() for profile in self.profiles),
        )


@dataclass(frozen=True)
class ProfileCapacityConstraint:
    ...

    def cache_key_parts(self) -> tuple[int, object, tuple[object, ...]]:
        return (
            self.domain_size,
            repr(self.assignment_count),
            tuple(profile.cache_key_parts() for profile in self.profiles),
        )
```

Also delete the legacy adapter functions from this file:

- `CellEvidenceAllocation.from_legacy`
- `unary_evidence_from_legacy`
- `evidence_plan_from_legacy`
- `_strategy_from_legacy_plan`
- legacy-only `_ground_atom_sort_key`

If any active test still needs them, rewrite the test to construct `UnaryEvidence`, `EvidencePartition`, or `ProfileCapacityConstraint` directly.

---

## Task 1: Add Structure Tests that Pin the Cleanup Boundary

**Files:**

- Modify: `tests/unit/test_fol_package_structure.py`

**Step 1: Add a test that forbids `_accessors.py`**

```python
def test_reduction_accessors_module_is_removed():
    assert not Path("src/wfomc/reduction/_accessors.py").exists()
```

**Step 2: Add a test that active modules do not import `_accessors`**

```python
def test_active_modules_do_not_import_reduction_accessors():
    offenders = []
    for path in Path("src/wfomc").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text()
        if "wfomc.reduction._accessors" in source or "reduction._accessors" in source:
            offenders.append(str(path))
    assert offenders == []
```

**Step 3: Add a focused business-field `getattr` test**

Do not ban all `getattr`; ban active legacy/business fields.

```python
def test_active_pipeline_does_not_probe_legacy_business_fields():
    scanned = [
        Path("src/wfomc/reduction"),
        Path("src/wfomc/engine"),
        Path("src/wfomc/algo/materialization.py"),
        Path("src/wfomc/algo/materialization_view.py"),
        Path("src/wfomc/algo/incremental3/counting_state.py"),
        Path("src/wfomc/algo/cell_graph/cache.py"),
        Path("src/wfomc/algo/cell_graph/components.py"),
        Path("src/wfomc/algo/cell_graph/inputs.py"),
        Path("src/wfomc/normal_form/c2"),
    ]
    forbidden_fields = (
        '"cnt_formulas"',
        '"uni_formula"',
        '"ext_formulas"',
        '"quantifier_scope"',
        '"quantified_formula"',
        '"preds"',
        '"consts"',
        '"qf_formula"',
        '"evidence_profiles"',
        '"required_unary_preds"',
        '"cardinality_constraint"',
        '"cardinality_constraints"',
    )
    allowed_files = {
        "src/wfomc/engine/runtime.py",
    }
    offenders = []
    files = []
    for item in scanned:
        files.extend(item.rglob("*.py") if item.is_dir() else [item])
    for path in files:
        if str(path) in allowed_files:
            continue
        source = path.read_text()
        for field in forbidden_fields:
            if f"getattr(" in source and field in source:
                offenders.append(f"{path}: {field}")
    assert offenders == []
```

**Step 4: Run tests and verify they fail before implementation**

Run:

```bash
PYTHONPATH=src:. pytest tests/unit/test_fol_package_structure.py -q
```

Expected: FAIL with `_accessors.py` imports and legacy probes.

---

## Task 2: Move Accessor Behavior onto Core Dataclasses

**Files:**

- Modify: `src/wfomc/problem.py`
- Modify: `src/wfomc/cardinality.py`
- Modify: `src/wfomc/normal_form/c2/norm_form.py`
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/evidence.py`
- Test: `tests/unit/test_problem.py` or create `tests/unit/test_problem_model.py`
- Test: `tests/unit/test_normal_form.py`
- Test: `tests/unit/test_evidence.py`

**Step 1: Add `Problem` methods**

Add:

- `has_unary_evidence`
- `has_profile_capacity_constraint`
- `has_cardinality_constraints`
- `weight_items()`
- `cache_key_parts(include_domain_size: bool)`

**Step 2: Add `C2NormalForm` methods**

Add:

- `has_counting`
- `has_forall_counting`
- `all_count_sections()`
- `has_rewritten_count_body`
- `to_sentence()`

`to_sentence()` should be the only replacement for `src/wfomc/reduction/normal_form.py`.

**Step 3: Add `ReducedProblems.expect_single_branch()`**

Use this from materializers and old single-branch algorithm bridges.

**Step 4: Add `AlgoOptions.resolved_evidence_strategy()`**

This replaces `options_evidence_strategy()`.

**Step 5: Add evidence cache keys**

Add `cache_key_parts()` on `UnaryLiteral`, `GroundUnaryLiteral`, `UnaryEvidence`, `EvidenceProfile`, `EvidencePartition`, and `ProfileCapacityConstraint`.

**Step 6: Add tests for these methods**

Use direct object construction. Do not import `fol.compat`.

Run:

```bash
PYTHONPATH=src:. pytest tests/unit/test_problem_model.py tests/unit/test_normal_form.py tests/unit/test_evidence.py -q
```

Expected: PASS after implementation.

---

## Task 3: Delete `reduction._accessors` and Inline Callers to Owner Methods

**Files:**

- Delete: `src/wfomc/reduction/_accessors.py`
- Modify: `src/wfomc/reduction/counting.py`
- Modify: `src/wfomc/reduction/evidence_planning.py`
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/algo/materialization.py`

**Step 1: Replace normal-form helpers**

Replace:

```python
normal_form_has_counting(normal_form)
forall_counts(normal_form)
counts(normal_form)
count_definitions(normal_form)
predicate_definitions(normal_form)
has_rewritten_count_body(normal_form)
```

with:

```python
normal_form.has_counting
normal_form.forall_counts
normal_form.counts
normal_form.count_definitions
normal_form.predicate_definitions
normal_form.has_rewritten_count_body
```

**Step 2: Replace problem helpers**

Replace:

```python
has_unary_evidence(problem)
has_cardinality_constraint(problem)
weight_items(problem)
```

with:

```python
problem.has_unary_evidence
problem.has_cardinality_constraints
problem.weight_items()
```

**Step 3: Replace option helper**

Replace:

```python
options_evidence_strategy(options)
```

with:

```python
options.resolved_evidence_strategy()
```

**Step 4: Type counting reduction section inputs**

In `src/wfomc/reduction/counting.py`, change section helpers from `object` to the real normal-form section classes:

```python
def reduce_exact_row_count(
    row_count: ForallCountSection,
    *,
    domain_size: int,
    ctx: FOLContext,
    rational_cls: type,
) -> CountReductionResult:
    outer_var = row_count.outer_var or ctx.variable("X")
    counted_var = row_count.counted_var or ctx.variable("Y")
```

Also replace `_can_reduce_global_count(gc: object)` and `_can_reduce_row_count(rc: object)` with typed signatures.

**Step 5: Delete the file**

After all imports are gone:

```bash
rm src/wfomc/reduction/_accessors.py
```

**Step 6: Run targeted tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_reduction.py tests/unit/test_engine_compile.py tests/unit/test_fol_package_structure.py -q
```

Expected: PASS.

---

## Task 4: Remove `reduction.normal_form` as a Compatibility Layer

**Files:**

- Delete: `src/wfomc/reduction/normal_form.py`
- Modify callers found by:

```bash
rg -n "reduction\\.normal_form|problem_with_normal_form_sentence|sentence_from_normal_form" src tests
```

**Step 1: Move sentence reconstruction to `C2NormalForm.to_sentence()`**

The implementation should use:

```python
from wfomc.fol import conjunction, count, forall
```

and typed section fields:

```python
count(section.counted_var, section.comparator, section.count, section.body)
forall(section.outer_var, count(...))
```

**Step 2: Replace callers**

Replace:

```python
sentence_from_normal_form(normal_form)
```

with:

```python
normal_form.to_sentence()
```

Replace problem mutation helper with explicit `dataclasses.replace` at the caller:

```python
replace(problem, sentence=normal_form.to_sentence())
```

Only do this when a caller truly needs to materialize a sentence. Prefer keeping `C2NormalForm` as the source of truth.

**Step 3: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_normal_form.py tests/unit/test_reduction.py -q
```

Expected: PASS.

---

## Task 5: Make Engine Feature and Cache Boundaries Typed

**Files:**

- Modify: `src/wfomc/engine/features.py`
- Modify: `src/wfomc/engine/orchestration.py`
- Test: `tests/unit/test_engine_compile.py`
- Test: `tests/unit/test_runtime_cache.py`

**Step 1: Type `analyze_features`**

Change:

```python
def analyze_features(problem: object, normal_form: object | None = None) -> FeatureSet:
```

to:

```python
def analyze_features(problem: Problem, normal_form: C2NormalForm | None = None) -> FeatureSet:
```

**Step 2: Delete legacy sentence scanners**

Remove code that scans:

- `contain_counting_quantifier`
- `contain_modulo_counting_quantifier`
- `cnt_formulas`
- `uni_formula`
- `ext_formulas`
- `preds`
- `consts`
- `quantifier_scope`
- `quantified_formula`

Use typed FOL traversal and `C2NormalForm` methods instead:

```python
has_counting = normal_form.has_counting if normal_form else formula_has_counting(problem.sentence)
has_mod_counting = any(section.comparator == "mod" for section in normal_form.all_count_sections())
has_unary_evidence = problem.has_unary_evidence or problem.has_profile_capacity_constraint
has_global_cardinality = problem.has_cardinality_constraints
```

**Step 3: Move predicate/constant analysis to typed FOL helpers**

If needed, add helpers in `src/wfomc/fol/analysis.py`:

```python
def predicates(formula: Formula) -> frozenset[Predicate]: ...
def constants(formula: Formula) -> frozenset[Constant]: ...
def has_counting(formula: Formula) -> bool: ...
def has_mod_counting(formula: Formula) -> bool: ...
```

Then `engine.features` should call those helpers instead of probing fields.

**Step 4: Replace engine cache key helpers**

In `engine/orchestration.py`, replace:

```python
_problem_key(problem, include_domain_size=True)
```

with:

```python
problem.cache_key_parts(include_domain_size=True)
```

Keep generic `_object_key` only if runtime cache needs opaque values outside `Problem`.

**Step 5: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_engine_compile.py tests/unit/test_runtime_cache.py -q
```

Expected: PASS.

---

## Task 6: Make Algo Core and Materialization Use Explicit Branches

**Files:**

- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/algo/materialization.py`
- Modify: `src/wfomc/algo/materialization_view.py`
- Modify: `src/wfomc/algo/standard/input.py`
- Modify: algorithm `spec.py` files that still accept `object`
- Test: `tests/unit/test_engine_compile.py`
- Test: `tests/test_incremental3_regressions.py`

**Step 1: Replace branch probing**

Replace:

```python
branches = getattr(reduced_problems, "branches", ())
branch = branches[0]
```

with:

```python
branch = reduced_problems.expect_single_branch()
```

**Step 2: Type `resolve_options`**

Change inner resolver signatures from:

```python
def resolve_options(features: object, options: AlgoOptions | None = None) -> AlgoOptions:
```

to:

```python
def resolve_options(features: FeatureSet, options: AlgoOptions | None = None) -> AlgoOptions:
```

Then replace all `getattr(features, "has_*", False)` with direct access:

```python
features.has_unary_evidence
features.has_linear_order
features.has_predk
features.has_circular_pred
features.has_mod_counting
```

**Step 3: Replace materialization problem probing**

In `algo/materialization.py`, replace:

```python
frozenset(getattr(problem, "domain", ()) or ())
dict(getattr(problem, "weights", {}) or {})
getattr(problem, "unary_evidence", None)
```

with:

```python
problem.domain
dict(problem.weights)
problem.unary_evidence
```

**Step 4: Type materialization views**

Replace predicate identity by adding one helper in the owner module:

```python
def predicate_key(predicate: Predicate) -> tuple[str, int]:
    return (predicate.name, predicate.arity)
```

Prefer placing it in `wfomc.fol` or `wfomc.fol.analysis`, not in algo cache files.

**Step 5: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_engine_compile.py tests/test_incremental3_regressions.py -q
```

Expected: PASS.

---

## Task 7: Clean Cell Graph Cache and Input Builders

**Files:**

- Modify: `src/wfomc/algo/cell_graph/cache.py`
- Modify: `src/wfomc/algo/cell_graph/components.py`
- Modify: `src/wfomc/algo/cell_graph/inputs.py`
- Modify: `src/wfomc/algo/materialization.py` if `MaterializedProblem` needs methods
- Test: `tests/unit/test_runtime_cache.py`
- Test: `tests/unary_evidence/test_unary_evidence_partition.py`
- Test: `tests/unary_evidence/test_cell_evidence_allocation.py`

**Step 1: Add cache-key methods to materialized objects**

If `MaterializedProblem` is a dataclass, add:

```python
def cache_key_parts(self) -> tuple[object, ...]:
    return (
        self.materialization_kind,
        repr(self.formula),
        tuple(sorted((predicate_key(pred), repr(pos), repr(neg)) for pred, pos, neg in self.weights)),
        self.evidence_plan.cache_key_parts(),
        self.cardinality_constraints.cache_key_parts(),
    )
```

Add `EvidencePlan.cache_key_parts()` in `evidence.py`.

**Step 2: Delete duplicated cache helpers**

Remove from `cell_graph/cache.py`:

- `_evidence_partition_key`
- `_evidence_profile_key`
- `_literal_key`
- `_predicate_key`

Use owner methods instead.

**Step 3: Remove legacy literal support from input builders**

In `cell_graph/inputs.py`, replace:

```python
_literal_positive(atom)
_literal_positive_atom(atom)
_literal_predicate(atom)
_literal_terms(atom)
```

with typed `Literal` / `Atom` access from `wfomc.fol`.

**Step 4: Make components consume typed inputs**

Replace:

```python
getattr(context, "leq_pred", None)
getattr(cell_graph, "cliques", ())
getattr(profile, "size")
```

with direct access on typed inputs. If there are two cell graph shapes, define two explicit adapters and call them at the boundary instead of probing both shapes everywhere.

**Step 5: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_runtime_cache.py tests/unary_evidence/test_unary_evidence_partition.py tests/unary_evidence/test_cell_evidence_allocation.py -q
```

Expected: PASS or intentional skips only.

---

## Task 8: Delete Evidence Legacy Adapters from Core Evidence

**Files:**

- Modify: `src/wfomc/evidence.py`
- Modify tests importing legacy evidence helpers

Find tests:

```bash
rg -n "from_legacy|unary_evidence_from_legacy|evidence_plan_from_legacy|legacy" tests src/wfomc/evidence.py
```

**Step 1: Delete core legacy conversion functions**

Delete:

- `CellEvidenceAllocation.from_legacy`
- `unary_evidence_from_legacy`
- `evidence_plan_from_legacy`
- `_strategy_from_legacy_plan`
- `_ground_atom_sort_key`

**Step 2: Rewrite tests to use native evidence constructors**

Old pattern:

```python
legacy = ...
evidence = unary_evidence_from_legacy(legacy)
```

New pattern:

```python
evidence = UnaryEvidence(
    (
        GroundUnaryLiteral(predicate=P, constant="a", positive=True),
        GroundUnaryLiteral(predicate=Q, constant="b", positive=False),
    )
)
```

**Step 3: Run evidence tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_evidence_planning.py tests/unary_evidence -q
```

Expected: PASS or intentional skips only.

---

## Task 9: Remove Legacy Sectioned Normal Form Support

**Files:**

- Modify: `src/wfomc/normal_form/c2/normalize.py`
- Modify: `src/wfomc/normal_form/c2/validation.py`
- Test: `tests/unit/test_normal_form.py`
- Test: `tests/unit/test_engine_compile.py`

**Step 1: Delete sectioned sentence detection**

Remove:

- `_is_sectioned_c2_sentence`
- `_normalize_sectioned_c2`
- `_quantifier_chain`
- `_is_universal_scope`
- `_is_counting_scope`

Also remove the early branch:

```python
if _is_sectioned_c2_sentence(sentence):
    return _normalize_sectioned_c2(sentence)
```

**Step 2: Type `normalize`**

Change:

```python
def normalize(sentence: object) -> C2NormalForm:
```

to:

```python
def normalize(sentence: Formula) -> C2NormalForm:
```

Keep private helper `object` annotations only where they truly accept terms/constants/numeric payloads.

**Step 3: Clean validation fallback**

Replace `hasattr(count, "outer_var")` and related probing with explicit overloads or `isinstance` checks against `ForallCountSection` / `CountSection`.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_normal_form.py tests/unit/test_engine_compile.py -q
```

Expected: PASS.

---

## Task 10: Tighten Weights Around `Problem`

**Files:**

- Modify: `src/wfomc/weights.py`
- Test: `tests/unit/test_weights.py`
- Test: `tests/unit/test_arithmetic_context.py`

**Step 1: Type problem-facing functions**

Change:

```python
def collect_symbolic_weight_variables(problem: object) -> tuple[str, ...]:
def build_weight_plan(problem: object, options: WeightOptions) -> WeightPlan:
def build_arithmetic_context(problem: object, options: WeightOptions) -> ArithmeticContext:
```

to:

```python
def collect_symbolic_weight_variables(problem: Problem) -> tuple[str, ...]:
def build_weight_plan(problem: Problem, options: WeightOptions) -> WeightPlan:
def build_arithmetic_context(problem: Problem, options: WeightOptions) -> ArithmeticContext:
```

**Step 2: Replace problem field probing**

Replace:

```python
getattr(problem, "weights", {})
getattr(problem, "cardinality_constraints", None)
getattr(problem, "cardinality_constraint", None)
getattr(problem, "cardinality_constraints", ())
```

with:

```python
problem.weights
problem.cardinality_constraints
```

`cardinality_constraint` and `cardinality_constraints` are legacy field names and should not be supported by active weight planning.

**Step 3: Keep numeric duck typing local**

Do not remove numeric boundary checks like:

```python
getattr(value, "is_number", True)
hasattr(value, "context")
```

Those inspect external symbolic/FLINT values, not WFOMC business models.

**Step 4: Run tests**

```bash
PYTHONPATH=src:. pytest tests/unit/test_weights.py tests/unit/test_arithmetic_context.py -q
```

Expected: PASS.

---

## Task 11: Remove Remaining Legacy FOL/Cell-Graph Blockers

**Files:**

- Modify or delete: `src/wfomc/cell_graph/cell_graph.py`
- Modify or delete: `src/wfomc/cell_graph/formula_ops.py`
- Modify: `src/wfomc/grounding/propositional.py`
- Modify tests importing `wfomc.fol.compat`

Find blockers:

```bash
rg -n "fol\\.compat|legacy_syntax|legacy_utils|QFFormula|AtomicFormula|Const|Pred\\b|from_legacy|legacy_adapter" src tests
```

**Step 1: Decide ownership**

If active algorithms now use `src/wfomc/algo/cell_graph`, delete the old top-level `src/wfomc/cell_graph` package.

If any active algorithm still imports top-level `wfomc.cell_graph`, migrate that import to `wfomc.algo.cell_graph` first.

**Step 2: Rewrite grounding to typed formula**

In `src/wfomc/grounding/propositional.py`, remove imports from `wfomc.fol.compat.legacy_syntax`.

Use typed constructors from `wfomc.fol`:

```python
from wfomc.fol import Atom, Constant, Formula, Predicate, Variable
```

Then ground typed `Formula` nodes only.

**Step 3: Remove tests that pin compat module presence**

Update `tests/unit/test_fol_package_structure.py`:

- delete tests asserting legacy modules live under `fol/compat`;
- delete tests asserting `legacy_adapter` exists;
- replace with tests asserting active modules do not import `wfomc.fol.compat`.

**Step 4: Run broad tests**

```bash
PYTHONPATH=src:. pytest tests/unit tests/unary_evidence/test_algorithm_matrix.py tests/test_incremental3_regressions.py -q
```

Expected: PASS with only intentional skips.

---

## Task 12: Final Scan and Cleanup

**Files:**

- Whole repo

**Step 1: Run import scan**

```bash
rg -n "wfomc\\.reduction\\._accessors|reduction\\._accessors|wfomc\\.fol\\.compat|legacy_syntax|legacy_utils|legacy_adapter" src tests
```

Expected: no active-path hits. Any remaining hit must be in intentionally skipped/deleted tests or docs.

**Step 2: Run active business-field `getattr` scan**

```bash
rg -n "getattr\\([^\\n]*(cnt_formulas|uni_formula|ext_formulas|quantifier_scope|quantified_formula|preds|consts|qf_formula|evidence_profiles|required_unary_preds|cardinality_constraint|cardinality_constraints)" src/wfomc
```

Expected: no active-path hits.

**Step 3: Run focused test suite**

```bash
PYTHONPATH=src:. pytest tests/unit tests/test_incremental3_regressions.py -q
PYTHONPATH=src:. pytest tests/unary_evidence/test_algorithm_matrix.py -q
```

Expected:

- unit and regression tests pass;
- unary evidence matrix passes with only the known intentionally disabled tests skipped.

**Step 4: Run full suite when the focused suite is clean**

```bash
PYTHONPATH=src:. pytest -q
```

Expected: PASS or only documented intentional skips.

---

## Implementation Order

Do this in order:

1. Add failing structure tests.
2. Add owner methods to `Problem`, `C2NormalForm`, `ReducedProblems`, `AlgoOptions`, and evidence classes.
3. Delete `reduction._accessors`.
4. Delete `reduction.normal_form`.
5. Clean engine features and orchestration cache keys.
6. Clean algo core and materialization.
7. Clean cell graph cache/input helpers.
8. Delete evidence legacy adapters.
9. Delete sectioned legacy normal-form support.
10. Tighten `weights.py` problem typing.
11. Remove remaining `fol.compat` blockers.
12. Run final scans and focused/full tests.

This order matters: deleting legacy first will create noisy breakage; adding owner methods first gives each later deletion a direct replacement.
