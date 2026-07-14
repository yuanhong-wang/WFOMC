# Evidence Interface Convergence Plan

Date: 2026-07-10

## Goal

Remove `EvidencePlan` entirely and converge evidence around one clear boundary:

```text
Problem.evidence
  -> reduction
  -> Problem.evidence_constraints
  -> algo-specific materialization
  -> CellEvidenceAllocation / GroundEvidenceInput / future TreewidthEvidenceInput
```

The final state must have no:

- `EvidencePlan`
- `Problem.unary_evidence`
- `Problem.profile_capacity_constraint`
- `UnaryLiteral`
- `wfomc.unary_evidence`
- `evidence/model.py`
- `evidence/binary.py`

## Target Package Layout

```text
src/wfomc/evidence/
  __init__.py
  core.py
  constraints.py
  unary.py
  materialization.py
```

## 1. Add Unified Evidence Input Model

File: `src/wfomc/evidence/core.py`

Keep or add:

```python
@dataclass(frozen=True)
class GroundUnaryLiteral:
    predicate: object
    constant: object
    positive: bool = True


@dataclass(frozen=True)
class UnaryEvidence:
    literals: tuple[GroundUnaryLiteral, ...] = ()


@dataclass(frozen=True)
class GroundBinaryLiteral:
    predicate: object
    left: object
    right: object
    positive: bool = True


@dataclass(frozen=True)
class BinaryEvidence:
    literals: tuple[GroundBinaryLiteral, ...] = ()


@dataclass(frozen=True)
class Evidence:
    unary: UnaryEvidence = UnaryEvidence()
    binary: BinaryEvidence = BinaryEvidence()

    @property
    def is_empty(self) -> bool:
        return self.unary.is_empty and self.binary.is_empty

    def cache_key_parts(self) -> tuple[object, object]:
        return (
            self.unary.cache_key_parts(),
            self.binary.cache_key_parts(),
        )
```

`EvidenceStrategy` can temporarily remain in `core.py`, although semantically it is an algorithm option rather than raw evidence.

## 2. Add Reduction-Level Evidence Constraints

New file: `src/wfomc/evidence/constraints.py`

Move `ProfileCapacityConstraint` out of `evidence/unary.py`:

```python
@dataclass(frozen=True)
class ProfileCapacityConstraint:
    profiles: tuple[EvidenceProfile, ...]
    domain_size: int
    assignment_count: object = 1
```

Add:

```python
@dataclass(frozen=True)
class EvidenceConstraints:
    unary_profile_capacity: ProfileCapacityConstraint | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.unary_profile_capacity is None
            or self.unary_profile_capacity.is_empty
        )

    def cache_key_parts(self) -> tuple[object, ...]:
        return (
            None
            if self.unary_profile_capacity is None
            else self.unary_profile_capacity.cache_key_parts(),
        )
```

`ProfileCapacityConstraint.from_unary_evidence(...)` may remain, but it should import `EvidencePartition` from `evidence.unary`.

## 3. Converge `Problem`

File: `src/wfomc/problem.py`

Replace:

```python
unary_evidence: UnaryEvidence
profile_capacity_constraint: ProfileCapacityConstraint | None
```

with:

```python
evidence: Evidence = field(default_factory=Evidence)
evidence_constraints: EvidenceConstraints = field(default_factory=EvidenceConstraints)
```

Update properties:

```python
@property
def has_unary_evidence(self) -> bool:
    return not self.evidence.unary.is_empty


@property
def has_binary_evidence(self) -> bool:
    return not self.evidence.binary.is_empty


@property
def has_profile_capacity_constraint(self) -> bool:
    c = self.evidence_constraints.unary_profile_capacity
    return c is not None and not c.is_empty
```

Update `cache_key_parts()` to use:

```python
self.evidence.cache_key_parts()
self.evidence_constraints.cache_key_parts()
```

Then remove all `problem.unary_evidence` and `problem.profile_capacity_constraint` call sites.

## 4. Delete `EvidencePlan`

Delete from `src/wfomc/evidence/unary.py`:

```python
class EvidencePlan
EvidencePlan.build(...)
EvidencePlan.build_from_profile_capacity(...)
```

All references must migrate to either `EvidenceConstraints` or algo-specific materialization inputs.

## 5. Replace Reduction Evidence Planning

Current file: `src/wfomc/reduction/evidence_planning.py`

This file currently builds `EvidencePlan`. Delete it or rewrite it as pure constraints/reduction logic.

Acceptable short-term path: keep the filename but remove `EvidencePlan` usage.

Suggested function:

```python
def reduce_evidence_constraints(
    problem: Problem,
    strategy: EvidenceStrategy,
) -> Problem:
    if strategy is EvidenceStrategy.LIFTED_PROFILES:
        return reduce_unary_evidence_to_profile_capacity(problem)

    if strategy is EvidenceStrategy.CCS:
        return reduce_unary_evidence_to_cardinality_constraints(problem)

    return problem
```

Delete:

```python
build_evidence_plan(...)
merge_cardinality_constraints(... evidence_plan ...)
```

`merge_cardinality_constraints` should no longer receive an `EvidencePlan`. CCS should merge evidence-induced cardinality constraints during the CCS reduction itself.

## 6. Update `reduction/core.py`

Update field access:

```python
evidence = problem.evidence.unary
```

Profile-capacity reduction:

```python
return replace(
    problem,
    evidence_constraints=replace(
        problem.evidence_constraints,
        unary_profile_capacity=ProfileCapacityConstraint.from_unary_evidence(
            evidence,
            frozenset(problem.domain),
        ),
    ),
    evidence=replace(problem.evidence, unary=UnaryEvidence()),
)
```

CCS reduction must no longer call `EvidencePlan.build(...)`.

Expose a unary helper:

```python
from wfomc.evidence.unary import build_unary_ccs_encoding

encoding = build_unary_ccs_encoding(evidence, frozenset(problem.domain))
```

Then:

```python
sentence=conjunction(problem.sentence, encoding.formula_patch)
cardinality_constraints=combine_cardinality_constraints(
    problem.cardinality_constraints,
    cardinality_constraints_from_simple_constraints(encoding.cardinality_constraints),
)
evidence=replace(problem.evidence, unary=UnaryEvidence())
```

## 7. Update `evidence/unary.py`

Keep:

```python
EvidenceProfile
EvidencePartition
CellEvidenceAllocation
CellConfigCoefficientBasis
CellEvidenceView
UnaryEvidencePartition  # only if still used
organize_evidence
```

Delete if unused:

```python
UnaryEvidencePlan
UnaryEvidenceStrategy
```

Add replacement for `EvidencePlan.build(..., CCS)`:

```python
@dataclass(frozen=True)
class UnaryCcsEncoding:
    formula_patch: Formula
    cardinality_constraints: tuple[tuple[object, str, int], ...]
    correction_factor: object


def build_unary_ccs_encoding(
    evidence: UnaryEvidence,
    domain: frozenset[object],
) -> UnaryCcsEncoding:
    partition = EvidencePartition.from_evidence(evidence, domain)
    formula_patch, cardinality_constraints = _ccs_formula_and_constraints(partition)
    return UnaryCcsEncoding(
        formula_patch=formula_patch,
        cardinality_constraints=cardinality_constraints,
        correction_factor=partition.assignment_count,
    )
```

Add replacement for `EvidencePlan.build_from_profile_capacity(...)`:

```python
def partition_from_profile_capacity(
    constraint: ProfileCapacityConstraint,
) -> EvidencePartition:
    return EvidencePartition(
        profiles=constraint.profiles,
        domain_size=constraint.domain_size,
        assignment_count=constraint.assignment_count,
    )
```

## 8. Update Materialization State

File: `src/wfomc/algo/materialization.py`

Replace:

```python
evidence_plan: EvidencePlan
```

with:

```python
evidence_constraints: EvidenceConstraints
```

If the algorithm needs lifted profile partition:

```python
constraint = state.problem.evidence_constraints.unary_profile_capacity
partition = partition_from_profile_capacity(constraint)
```

Delete `_materialize_evidence()` dependency on `evidence_plan.formula_patch`; CCS should already be encoded by reduction.

## 9. Update Algo Inputs

Files:

```text
src/wfomc/algo/materialization.py
src/wfomc/algo/cell_graph/types.py
src/wfomc/algo/cell_graph/components.py
src/wfomc/algo/cell_graph/inputs.py
src/wfomc/algo/standard/input.py
src/wfomc/algo/tail_signature/input.py
```

Replace generic:

```python
evidence_plan: EvidencePlan
```

with either:

```python
evidence_constraints: EvidenceConstraints
```

or more specific algorithm-owned inputs:

```python
unary_profile_partition: EvidencePartition | None
cell_evidence_allocation: CellEvidenceAllocation | None
```

Do not pass a generic `EvidencePlan`.

## 10. Update Cell-Graph Components

File: `src/wfomc/algo/cell_graph/components.py`

Replace:

```python
reduced.evidence_plan.strategy
reduced.evidence_plan.partition
reduced.evidence_plan.required_unary_predicates
```

with:

```python
constraint = reduced.evidence_constraints.unary_profile_capacity
```

If empty:

```python
if constraint is None or constraint.is_empty:
    return None
```

Otherwise:

```python
partition = partition_from_profile_capacity(constraint)
return CellEvidenceAllocation.from_partition(partition, cells)
```

Compute required predicates from profiles:

```python
def required_unary_predicates_from_partition(partition):
    return frozenset(
        literal.predicate
        for profile in partition.profiles
        for literal in profile.literals
    )
```

## 11. Update Tail Signature

File: `src/wfomc/algo/tail_signature/solve.py`

Replace:

```python
_profile_capacities_from_evidence_plan(evidence_plan)
```

with:

```python
_profile_capacities_from_evidence_constraints(evidence_constraints)
```

Read directly from:

```python
constraint = evidence_constraints.unary_profile_capacity
constraint.profiles
constraint.domain_size
constraint.assignment_count
```

Required predicates are derived from profiles.

## 12. Update Parser and Problem Construction

File:

```text
src/wfomc/parser/transformers/fol.py
```

Parser can still produce `UnaryEvidence`.

Problem construction should use:

```python
Problem(
    evidence=Evidence(unary=parsed_unary_evidence),
)
```

not:

```python
Problem(unary_evidence=...)
```

## 13. Update Feature Analysis

File:

```text
src/wfomc/engine/features.py
```

Existing properties can remain if `Problem` exposes them:

```python
problem.has_unary_evidence
problem.has_profile_capacity_constraint
```

Add if needed:

```python
problem.has_binary_evidence
```

## 14. Clean Exports

Files:

```text
src/wfomc/evidence/__init__.py
src/wfomc/__init__.py
```

`wfomc.evidence.__all__` should export:

```python
"Evidence",
"EvidenceStrategy",
"GroundUnaryLiteral",
"UnaryEvidence",
"GroundBinaryLiteral",
"BinaryEvidence",
"EvidenceConstraints",
"ProfileCapacityConstraint",
"EvidenceProfile",
"EvidencePartition",
"CellEvidenceAllocation",
"CellConfigCoefficientBasis",
"build_unary_ccs_encoding",
"partition_from_profile_capacity",
```

Do not export:

```text
EvidencePlan
UnaryLiteral
UnaryEvidencePlan  # if deleted
UnaryEvidenceStrategy # if deleted
```

## 15. Delete Files and Old Entrypoints

Must not exist:

```text
src/wfomc/unary_evidence.py
src/wfomc/evidence/model.py
src/wfomc/evidence/binary.py
```

Clean pycache:

```bash
find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
```

## 16. Update Tests

Update:

```text
tests/unit/test_evidence_partition.py
tests/unit/test_evidence_planning.py
tests/unit/test_evidence_execution.py
tests/unit/test_cell_graph_inputs.py
tests/unit/test_tail_signature_adapter.py
tests/unit/reduction/test_core_contract.py
tests/unit/reduction/test_unary_evidence_reduction.py
tests/unary_evidence/*
tests/unit/test_fol_package_structure.py
```

Replace:

```python
EvidencePlan
```

with:

```python
EvidenceConstraints
ProfileCapacityConstraint
EvidencePartition
CellEvidenceAllocation
```

Replace:

```python
Problem(unary_evidence=...)
```

with:

```python
Problem(evidence=Evidence(unary=...))
```

Replace:

```python
profile_capacity_constraint=...
```

with:

```python
evidence_constraints=EvidenceConstraints(unary_profile_capacity=...)
```

## 17. Required Scans

After implementation:

```bash
rg -n "\bEvidencePlan\b|wfomc\.unary_evidence|\bUnaryLiteral\b|evidence\.model|evidence\.binary|profile_capacity_constraint|unary_evidence" src tests
```

Expected:

- `EvidencePlan`: no hits
- `wfomc.unary_evidence`: no hits
- `UnaryLiteral`: no hits
- `evidence.model` / `evidence.binary`: no hits
- `profile_capacity_constraint`: only allowed as `EvidenceConstraints.unary_profile_capacity`
- `unary_evidence`: no longer a `Problem` field

Stricter scan:

```bash
rg -n "problem\.unary_evidence|problem\.profile_capacity_constraint|evidence_plan" src tests
```

Expected: no hits.

## 18. Required Tests

Focused:

```bash
PYTHONPATH=src:. pytest \
  tests/unit/test_evidence_partition.py \
  tests/unit/test_evidence_planning.py \
  tests/unit/test_evidence_execution.py \
  tests/unit/test_cell_graph_inputs.py \
  tests/unit/test_tail_signature_adapter.py \
  tests/unit/reduction/test_core_contract.py \
  tests/unit/reduction/test_unary_evidence_reduction.py \
  tests/unary_evidence \
  tests/unit/test_fol_package_structure.py \
  -q
```

Wide:

```bash
PYTHONPATH=src:. pytest tests/unit tests/test_incremental3_regressions.py -q
```

Algorithm matrix:

```bash
PYTHONPATH=src:. pytest tests/unary_evidence/test_algorithm_matrix.py -q
```

## Final Acceptance Criteria

All must be true:

- No `EvidencePlan`
- No `Problem.unary_evidence`
- No `Problem.profile_capacity_constraint`
- No `UnaryLiteral`
- No `wfomc.unary_evidence`
- No `evidence/model.py`
- No `evidence/binary.py`
- Evidence flow is:

```text
Problem.evidence
  -> reduction
  -> Problem.evidence_constraints
  -> algo-specific materialization
  -> CellEvidenceAllocation / GroundEvidenceInput / future TreewidthEvidenceInput
```
