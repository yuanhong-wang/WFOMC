"""Native problem models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from wfomc.cardinality_constraints import CardinalityConstraints
from wfomc.evidence import Evidence
from wfomc.evidence.profile import ProfileCapacityConstraint

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext
    from wfomc.fol import Formula
    from wfomc.fol.normal_form import C2NormalForm
    from wfomc.weights import CompiledWeightMapping, RawWeightMapping


# ---------------------------------------------------------------------------
# Public source problem
#
# ``Problem`` is the only problem type exposed by the package-level API. It is
# parser/user input: a typed source formula, raw weights, evidence, and declared
# constraints. Reduction artifacts must not be added to this class.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Problem:
    """Native execution input.

    Carries typed artifacts (a typed formula, an exact ``CardinalityConstraints``,
    and structured ``Evidence``) so the engine never converts to the
    legacy ``WFOMCProblem`` object model.
    """

    sentence: "Formula"
    domain: frozenset[object] = frozenset()
    weights: Mapping[object, tuple[object, object]] = field(default_factory=dict)
    cardinality_constraints: CardinalityConstraints = field(default_factory=CardinalityConstraints)
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

    def weight_items(self) -> tuple[tuple[object, object, object], ...]:
        return tuple(
            (predicate, positive, negative)
            for predicate, (positive, negative) in self.weights.items()
        )

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

    def cache_key_parts(self, *, include_domain_size: bool) -> tuple[object, ...]:
        domain_part = (
            len(self.domain) if include_domain_size
            else tuple(sorted(map(str, self.domain)))
        )
        return (
            repr(self.sentence),
            domain_part,
            tuple(
                sorted((repr(k), repr(v)) for k, v in self.weights.items())
            ),
            self.evidence.cache_key_parts(),
            self.cardinality_constraints.cache_key_parts(),
            tuple(sorted((str(k), repr(v)) for k, v in self.options.items())),
        )


# ---------------------------------------------------------------------------
# Internal pipeline stages
#
# These two flat dataclasses intentionally live beside ``Problem`` so the three
# representations can be compared in one place. They are internal contracts,
# not additional public API models. Cross-stage conversion belongs to the
# owning pipeline module (``reduction.core`` or ``algo.core``), never here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReducedProblem:
    """Logical-reduction state; ``normal_form`` is always the authority."""

    normal_form: "C2NormalForm"
    domain: frozenset[object] = frozenset()
    weights: "RawWeightMapping" = field(default_factory=dict)
    cardinality_constraints: CardinalityConstraints = field(
        default_factory=CardinalityConstraints
    )
    evidence: Evidence = field(default_factory=Evidence)
    profile_capacity_constraint: ProfileCapacityConstraint | None = None
    internal_weight_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from wfomc.fol.normal_form import C2NormalForm

        if not isinstance(self.normal_form, C2NormalForm):
            raise TypeError("ReducedProblem.normal_form must be C2NormalForm")

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
        return not self.cardinality_constraints.is_empty


@dataclass(frozen=True)
class CompiledProblem:
    """Common QF/numeric input consumed by algorithm-owned builders."""

    sentence: "Formula"
    arithmetic: "ArithmeticContext"
    domain: frozenset[object] = frozenset()
    weights: "CompiledWeightMapping" = field(default_factory=dict)
    evidence: Evidence = field(default_factory=Evidence)
    profile_capacity_constraint: ProfileCapacityConstraint | None = None

    def __post_init__(self) -> None:
        from wfomc.fol import Formula, is_quantifier_free

        if not isinstance(self.sentence, Formula):
            raise TypeError("CompiledProblem.sentence must be a typed Formula")
        if not is_quantifier_free(self.sentence):
            raise ValueError("CompiledProblem.sentence must be quantifier-free")

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


__all__ = ["CompiledProblem", "Problem", "ReducedProblem"]
