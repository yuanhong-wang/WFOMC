"""Source, domain, compilation, and execution problem models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from wfomc.cardinality_constraints import CardinalityConstraints
from wfomc.evidence import Evidence
from wfomc.evidence.profile import ProfileCapacityConstraint

if TYPE_CHECKING:
    from wfomc.algo.core import AlgoInput, AlgoName, AlgoOptions, PreparedBranch
    from wfomc.arithmetic import ArithmeticContext
    from wfomc.engine.features import FeatureSet
    from wfomc.fol import Formula
    from wfomc.fol.normal_form import C2NormalForm
    from wfomc.weights import CompiledWeightMapping, RawWeightMapping


# ---------------------------------------------------------------------------
# Public source and domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    """One concrete finite domain, separate from the reusable logical query."""

    elements: frozenset[object] = frozenset()
    # Number of elements participating in CIRCULAR_PRED. It may be smaller
    # than the full domain when auxiliary elements are also present.
    circular_order_size: int | None = None

    def __post_init__(self) -> None:
        if self.circular_order_size is not None and not (
            0 <= self.circular_order_size <= len(self.elements)
        ):
            raise ValueError(
                "Domain.circular_order_size must be non-negative and no "
                "larger than the domain"
            )

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def size(self) -> int:
        return len(self.elements)

    @classmethod
    def of_size(cls, size: int, *, prefix: str = "domain") -> "Domain":
        """Create a typed domain with stable generated constant names."""

        if size < 0:
            raise ValueError("Domain size must be non-negative")
        from wfomc.fol import FOLContext

        context = FOLContext()
        return cls(
            frozenset(
                context.constant(f"{prefix}{index}") for index in range(size)
            )
        )

    def cache_key_parts(self) -> tuple[object, ...]:
        return (
            tuple(sorted(map(str, self.elements))),
            self.circular_order_size,
        )


@dataclass(frozen=True)
class Problem:
    """Reusable logical WFOMC query with no concrete finite domain."""

    sentence: "Formula"
    weights: Mapping[object, tuple[object, object]] = field(default_factory=dict)
    cardinality_constraints: CardinalityConstraints = field(
        default_factory=CardinalityConstraints
    )
    evidence: Evidence = field(default_factory=Evidence)
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from wfomc.fol import Formula

        if not isinstance(self.sentence, Formula):
            raise TypeError("Problem.sentence must be a typed Formula")

    @property
    def has_unary_evidence(self) -> bool:
        return not self.evidence.unary.is_empty

    @property
    def has_binary_evidence(self) -> bool:
        return not self.evidence.binary.is_empty

    @property
    def has_cardinality_constraints(self) -> bool:
        return not self.cardinality_constraints.is_empty

    @property
    def has_profile_capacity_constraint(self) -> bool:
        return False

    @property
    def internal_weight_symbols(self) -> tuple[str, ...]:
        return ()

    def declared_predicate_names(self) -> frozenset[str]:
        """Names reserved by formula, weights, evidence, or constraints."""

        from wfomc.fol import Predicate, predicates

        declared = {predicate.name for predicate in predicates(self.sentence)}

        def add(predicate: object) -> None:
            declared.add(
                predicate.name if isinstance(predicate, Predicate) else str(predicate)
            )

        for predicate in self.weights:
            add(predicate)
        for literal in self.evidence.unary.literals:
            add(literal.predicate)
        for literal in self.evidence.binary.literals:
            add(literal.predicate)
        for constraint in self.cardinality_constraints.constraints:
            for term in constraint.terms:
                add(term.predicate)
        return frozenset(declared)

    def required_domain_constants(self) -> frozenset[object]:
        """Constants referenced by the formula or ground evidence."""

        from wfomc.fol import constants

        required = set(constants(self.sentence))
        required.update(
            literal.to_ground_literal().terms[0]
            for literal in self.evidence.unary.literals
        )
        for literal in self.evidence.binary.literals:
            ground = literal.to_ground_literal()
            required.update(ground.terms)
        return frozenset(required)

    def cache_key_parts(self) -> tuple[object, ...]:
        return (
            repr(self.sentence),
            tuple(sorted((repr(k), repr(v)) for k, v in self.weights.items())),
            self.evidence.cache_key_parts(),
            self.cardinality_constraints.cache_key_parts(),
            tuple(sorted((str(k), repr(v)) for k, v in self.options.items())),
        )


@dataclass(frozen=True)
class ProblemInstance:
    """Parsed/convenience pairing of one logical problem and one domain."""

    problem: Problem
    domain: Domain = Domain()


# ---------------------------------------------------------------------------
# Reusable engine artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledProblem:
    """Domain-free compilation selected for one algorithm and option set."""

    problem: Problem
    feature_set: "FeatureSet"
    algo: "AlgoName"
    algo_options: "AlgoOptions"
    # Algorithms store reusable compiled branches here.
    branches: tuple[object, ...] = ()


@dataclass(frozen=True)
class ProblemExecution:
    """One concrete-domain instantiation of a reusable compilation."""

    compiled_problem: CompiledProblem
    domain: Domain
    prepared_branches: tuple["PreparedBranch", ...]

    @property
    def algo_inputs(self) -> tuple["AlgoInput", ...]:
        return tuple(branch.algo_input for branch in self.prepared_branches)

    @property
    def algo_input(self) -> "AlgoInput | None":
        return self.algo_inputs[0] if self.algo_inputs else None


# ---------------------------------------------------------------------------
# Concrete per-domain branch data used by algorithm input instantiation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledBranchInstance:
    """Concrete QF/numeric branch consumed by algorithm input builders."""

    sentence: "Formula"
    arithmetic: "ArithmeticContext"
    domain: frozenset[object] = frozenset()
    weights: "CompiledWeightMapping" = field(default_factory=dict)
    evidence: Evidence = field(default_factory=Evidence)
    profile_capacity_constraint: ProfileCapacityConstraint | None = None
    circular_order_size: int | None = None

    def __post_init__(self) -> None:
        from wfomc.fol import Formula, is_quantifier_free

        if not isinstance(self.sentence, Formula):
            raise TypeError("CompiledBranchInstance.sentence must be a typed Formula")
        if not is_quantifier_free(self.sentence):
            raise ValueError("CompiledBranchInstance.sentence must be quantifier-free")

    @property
    def has_unary_evidence(self) -> bool:
        return not self.evidence.unary.is_empty

    @property
    def has_binary_evidence(self) -> bool:
        return not self.evidence.binary.is_empty

    @property
    def has_profile_capacity_constraint(self) -> bool:
        constraint = self.profile_capacity_constraint
        return constraint is not None and not constraint.is_empty

    @property
    def has_cardinality_constraints(self) -> bool:
        return False

    @property
    def internal_weight_symbols(self) -> tuple[str, ...]:
        return tuple(
            symbol
            for symbol in self.arithmetic.symbolic_variables
            if symbol not in self.arithmetic.output_symbols
        )


__all__ = [
    "CompiledBranchInstance",
    "CompiledProblem",
    "Domain",
    "Problem",
    "ProblemExecution",
    "ProblemInstance",
]
