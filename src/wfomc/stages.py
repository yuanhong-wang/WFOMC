"""Neutral feature, reduction, compilation, and branch-stage data."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import TYPE_CHECKING

from wfomc.cardinality_constraints import (
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.evidence import Evidence
from wfomc.evidence.profile import EvidenceProfile, ProfileCapacityConstraint
from wfomc.problem import Problem

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext
    from wfomc.fol.literals import Literal
    from wfomc.fol.normal_form.c2.norm_form import C2NormalForm
    from wfomc.fol.syntax import Formula, Predicate
    from wfomc.weights import CompiledWeightMapping


# ---------------------------------------------------------------------------
# Feature-analysis output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSet:
    """Domain-free source or reduced-branch capabilities."""

    has_c2_counting: bool = False
    has_mod_counting: bool = False
    has_unary_evidence: bool = False
    has_binary_evidence: bool = False
    leq_predicate: "Predicate | None" = None
    predecessor_predicates: tuple[tuple[int, "Predicate"], ...] = ()
    circular_predecessor_predicate: "Predicate | None" = None
    named_constants: tuple[str, ...] = ()

    @property
    def has_named_constants(self) -> bool:
        return bool(self.named_constants)

    @property
    def has_linear_order(self) -> bool:
        return self.leq_predicate is not None

    @property
    def has_predk(self) -> bool:
        return bool(self.predecessor_predicates)

    @property
    def has_circular_pred(self) -> bool:
        return self.circular_predecessor_predicate is not None


# ---------------------------------------------------------------------------
# Domain-parametric reduction data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainExpr:
    """Small exact arithmetic expression evaluated from a domain size."""

    op: str
    args: tuple[object, ...] = ()

    @classmethod
    def constant(cls, value: int | Fraction) -> "DomainExpr":
        return cls("constant", (Fraction(value),))

    @classmethod
    def size(cls) -> "DomainExpr":
        return cls("size")

    def evaluate(self, domain_size: int) -> Fraction:
        values = tuple(
            arg.evaluate(domain_size) if isinstance(arg, DomainExpr) else arg
            for arg in self.args
        )
        if self.op == "constant":
            return Fraction(values[0])
        if self.op == "size":
            return Fraction(domain_size)
        if self.op == "add":
            return Fraction(values[0]) + Fraction(values[1])
        if self.op == "sub":
            return Fraction(values[0]) - Fraction(values[1])
        if self.op == "mul":
            return Fraction(values[0]) * Fraction(values[1])
        if self.op == "div":
            return Fraction(values[0]) / Fraction(values[1])
        if self.op == "floordiv":
            return Fraction(math.floor(Fraction(values[0]) / Fraction(values[1])))
        if self.op == "pow":
            exponent = _exact_int(Fraction(values[1]), "power exponent")
            return Fraction(values[0]) ** exponent
        if self.op == "factorial":
            value = _exact_int(Fraction(values[0]), "factorial argument")
            return Fraction(math.factorial(value))
        if self.op == "min":
            return min(Fraction(value) for value in values)
        raise ValueError(f"Unknown domain expression operation: {self.op!r}")

    def __add__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("add", (self, _domain_expr(other)))

    def __sub__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("sub", (self, _domain_expr(other)))

    def __mul__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("mul", (self, _domain_expr(other)))

    def __truediv__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("div", (self, _domain_expr(other)))

    def __floordiv__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("floordiv", (self, _domain_expr(other)))

    def power(self, exponent: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("pow", (self, _domain_expr(exponent)))

    def factorial(self) -> "DomainExpr":
        return DomainExpr("factorial", (self,))

    @classmethod
    def minimum(cls, *values: "DomainExpr") -> "DomainExpr":
        if len(values) == 1:
            return values[0]
        return cls("min", tuple(values))


N = DomainExpr.size()


@dataclass(frozen=True)
class ReducedCardinalityConstraint:
    """Linear cardinality constraint with a domain-sized right-hand side."""

    terms: tuple[CardinalityTerm, ...]
    comparator: Comparator
    rhs: DomainExpr
    modulus: int | None = None

    def instantiate(self, domain_size: int) -> LinearCardinalityConstraint:
        return LinearCardinalityConstraint(
            terms=self.terms,
            comparator=self.comparator,
            rhs=_exact_int(self.rhs.evaluate(domain_size), "cardinality rhs"),
            modulus=self.modulus,
        )


@dataclass(frozen=True)
class ReducedProfileConstraint:
    """Unary-evidence profiles whose capacities may depend on the domain."""

    profiles: tuple[tuple[frozenset["Literal"], DomainExpr], ...]

    def profile_sizes(self, domain_size: int) -> tuple[int, ...]:
        """Evaluate all capacities, including structurally absent zero profiles."""

        sizes = tuple(
            _exact_int(size.evaluate(domain_size), "evidence profile size")
            for _literals, size in self.profiles
        )
        if any(size < 0 for size in sizes):
            raise ValueError("Evidence profile sizes must be non-negative.")
        return sizes

    def active_profile_indices(self, domain_size: int) -> tuple[int, ...]:
        """Return the profiles present in one concrete domain."""

        return tuple(
            index
            for index, size in enumerate(self.profile_sizes(domain_size))
            if size > 0
        )

    def structural_constraint(
        self,
        active_profile_indices: tuple[int, ...] | None = None,
    ) -> ProfileCapacityConstraint:
        """Return a size-insensitive constraint for cell-graph construction."""

        indices = (
            tuple(range(len(self.profiles)))
            if active_profile_indices is None
            else active_profile_indices
        )
        return ProfileCapacityConstraint(
            profiles=tuple(
                EvidenceProfile(self.profiles[index][0], 1) for index in indices
            ),
            domain_size=len(indices),
        )

    def instantiate(self, domain_size: int) -> ProfileCapacityConstraint:
        sizes = self.profile_sizes(domain_size)
        profiles = tuple(
            EvidenceProfile(literals, size)
            for (literals, _expression), size in zip(self.profiles, sizes)
            if size > 0
        )
        assignment_count = Fraction(math.factorial(domain_size), 1)
        for profile in profiles:
            assignment_count /= math.factorial(profile.size)
        return ProfileCapacityConstraint(
            profiles=profiles,
            domain_size=domain_size,
            assignment_count=assignment_count,
        )

    def assignment_count_expr(self) -> DomainExpr:
        denominator = DomainExpr.constant(1)
        for _literals, size in self.profiles:
            denominator = denominator * size.factorial()
        return N.factorial() / denominator


# ---------------------------------------------------------------------------
# Reduction decoder data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivideDecoderSpec:
    coefficient: DomainExpr


@dataclass(frozen=True)
class CardinalityDecoderSpec:
    constraints: tuple[ReducedCardinalityConstraint, ...]
    predicate_markers: tuple[tuple[object, str, int], ...]


DecoderStep = DivideDecoderSpec | CardinalityDecoderSpec


@dataclass(frozen=True)
class DecoderSpec:
    """Data-only composition of reduction corrections."""

    steps: tuple[DecoderStep, ...] = ()

    def append(self, step: DecoderStep) -> "DecoderSpec":
        return DecoderSpec(self.steps + (step,))


@dataclass(frozen=True)
class ReducedProblem:
    """Domain-free logical branch produced by staged reductions."""

    normal_form: "C2NormalForm"
    weights: dict[object, tuple[object, object]] = field(default_factory=dict)
    cardinality_constraints: tuple[ReducedCardinalityConstraint, ...] = ()
    evidence: Evidence = field(default_factory=Evidence)
    profile_constraint: ReducedProfileConstraint | None = None
    # Unary CCS markers; unmarked elements form the empty evidence profile.
    ccs_profile_markers: tuple[object, ...] = ()
    ccs_unmarked_size: DomainExpr | None = None
    internal_weight_symbols: tuple[str, ...] = ()
    internal_weight_degree_limits: tuple[tuple[str, DomainExpr], ...] = ()
    decoder_spec: DecoderSpec = DecoderSpec()
    min_domain_size: int = 0

    @property
    def has_unary_evidence(self) -> bool:
        return not self.evidence.unary.is_empty

    @property
    def has_binary_evidence(self) -> bool:
        return not self.evidence.binary.is_empty

    @property
    def has_profile_capacity_constraint(self) -> bool:
        return self.profile_constraint is not None

# ---------------------------------------------------------------------------
# Numeric compilation and concrete branch data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingProblem:
    """Domain-free source problem and numeric data prepared for grounding."""

    problem: Problem
    arithmetic: "ArithmeticContext"
    weights: dict[object, tuple[object, object]]
    feature_set: FeatureSet


@dataclass(frozen=True)
class CompiledReducedBranch:
    """Domain-free numeric branch consumed by a staged input builder."""

    reduced_problem: ReducedProblem
    sentence: "Formula"
    arithmetic: "ArithmeticContext"
    weights: dict[object, tuple[object, object]]
    feature_set: FeatureSet


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

def _domain_expr(value: int | DomainExpr) -> DomainExpr:
    return value if isinstance(value, DomainExpr) else DomainExpr.constant(value)


def _exact_int(value: Fraction, label: str) -> int:
    if value.denominator != 1:
        raise ValueError(f"{label} must be integral, got {value}")
    return value.numerator


__all__ = [
    "CardinalityDecoderSpec",
    "CompiledBranchInstance",
    "CompiledReducedBranch",
    "DecoderSpec",
    "DivideDecoderSpec",
    "DomainExpr",
    "FeatureSet",
    "GroundingProblem",
    "N",
    "ReducedCardinalityConstraint",
    "ReducedProblem",
    "ReducedProfileConstraint",
]
