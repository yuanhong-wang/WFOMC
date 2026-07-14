"""Reduce raw unary evidence into solver-specific representations."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING

from wfomc.evidence.data import UnaryEvidence
from wfomc.evidence.profile import EvidenceProfile, ProfileCapacityConstraint
from wfomc.fol import FOLContext, Formula, Literal

if TYPE_CHECKING:
    from wfomc.cardinality_constraints import CardinalityConstraints
    from wfomc.problem import ReducedProblem
    from wfomc.reduction.core import ProblemWithDecoder


@dataclass(frozen=True)
class UnaryCcsEncoding:
    """CCS formula patch, cardinality constraints, and correction factor."""

    formula_patch: Formula
    cardinality_constraints: tuple[tuple[object, str, int], ...]
    correction_factor: Fraction


def reduce_unary_evidence_to_profile_capacity(
    problem: "ReducedProblem",
) -> "ReducedProblem":
    """Reduce unary evidence to exactly one profile-capacity constraint."""

    unary = problem.evidence.unary
    if unary.is_empty:
        return problem
    if problem.profile_capacity_constraint is not None:
        raise ValueError(
            "Cannot reduce unary evidence to profile_capacity_constraint when "
            "profile_capacity_constraint is already set"
        )

    return replace(
        problem,
        profile_capacity_constraint=build_profile_capacity_constraint(
            unary,
            problem.domain,
        ),
        evidence=replace(problem.evidence, unary=UnaryEvidence()),
    )


def reduce_unary_evidence_to_cardinality_constraints(
    problem: "ReducedProblem",
) -> "ReducedProblem | ProblemWithDecoder":
    """Reduce unary evidence to cardinality constraints using CCS encoding."""

    from wfomc.cardinality_constraints import combine_cardinality_constraints
    from wfomc.fol import conjunction, true
    from wfomc.reduction.core import ProblemWithDecoder, divide_decoder

    unary = problem.evidence.unary
    if unary.is_empty:
        return problem

    encoding = build_unary_ccs_encoding(unary, problem.domain)
    ccs_constraints = _cardinality_constraints_from_simple_constraints(
        encoding.cardinality_constraints
    )
    qf_formula = problem.normal_form.qf_formula
    if qf_formula is None:
        qf_formula = true()
    reduced = replace(
        problem,
        normal_form=replace(
            problem.normal_form,
            qf_formula=conjunction(
                qf_formula,
                encoding.formula_patch,
            ),
        ),
        cardinality_constraints=combine_cardinality_constraints(
            problem.cardinality_constraints,
            ccs_constraints,
        ),
        evidence=replace(problem.evidence, unary=UnaryEvidence()),
    )
    return ProblemWithDecoder(reduced, divide_decoder(encoding.correction_factor))


def _cardinality_constraints_from_simple_constraints(
    constraints: tuple[tuple[object, str, int], ...],
) -> "CardinalityConstraints":
    """Convert the small CCS output shape into the problem's public model."""

    from wfomc.cardinality_constraints import (
        CardinalityConstraints,
        CardinalityTerm,
        Comparator,
        LinearCardinalityConstraint,
    )

    return CardinalityConstraints(
        tuple(
            LinearCardinalityConstraint(
                terms=(CardinalityTerm(predicate),),
                comparator=Comparator(comparator),
                rhs=rhs,
            )
            for predicate, comparator, rhs in constraints
        )
    )


def build_profile_capacity_constraint(
    evidence: UnaryEvidence,
    domain: frozenset[object],
) -> ProfileCapacityConstraint:
    """Group ground unary literals by their domain-element profile."""

    element_to_literals: dict[object, set[Literal]] = {}
    for literal in evidence.literals:
        ground_literal = literal.to_ground_literal()
        constant = ground_literal.terms[0]
        if constant not in domain:
            raise ValueError(f"Evidence must be consistent with the domain: {constant}")
        normalized = literal.to_profile_literal()
        element_literals = element_to_literals.setdefault(constant, set())
        if ~normalized in element_literals:
            raise ValueError(
                f"Evidence must be consistent for {constant}: {literal.predicate}"
            )
        element_literals.add(normalized)

    sizes: dict[frozenset[Literal], int] = {}
    for element in sorted(domain, key=str):
        profile = frozenset(element_to_literals.get(element, set()))
        sizes[profile] = sizes.get(profile, 0) + 1

    profiles = tuple(
        EvidenceProfile(profile, sizes[profile])
        for profile in sorted(sizes, key=_profile_sort_key)
    )
    assignment_count = Fraction(math.factorial(len(domain)), 1)
    for profile in profiles:
        assignment_count /= math.factorial(profile.size)
    return ProfileCapacityConstraint(
        profiles=profiles,
        domain_size=len(domain),
        assignment_count=assignment_count,
    )


def build_unary_ccs_encoding(
    evidence: UnaryEvidence,
    domain: frozenset[object],
) -> UnaryCcsEncoding:
    """Encode unary evidence with auxiliary profile predicates."""

    constraint = build_profile_capacity_constraint(evidence, domain)
    formula_patch, cardinality_constraints = _ccs_formula_and_constraints(constraint)
    return UnaryCcsEncoding(
        formula_patch=formula_patch,
        cardinality_constraints=cardinality_constraints,
        correction_factor=constraint.assignment_count,
    )


def _ccs_formula_and_constraints(
    constraint: ProfileCapacityConstraint,
) -> tuple[Formula, tuple[tuple[object, str, int], ...]]:
    from wfomc.fol import conjunction, disjunction, implies, neg, true

    ctx = FOLContext()
    formula = true()
    aux_predicates = []
    cardinality_constraints = []

    for idx, profile in enumerate(constraint.profiles):
        aux_predicate = ctx.predicate(f"EvidenceProfile_{idx}", 1)
        aux_predicates.append(aux_predicate)
        aux_atom = ctx.atom(aux_predicate, ctx.variable("X"))

        if profile.literals:
            profile_parts = tuple(
                _literal_to_formula(literal, ctx)
                for literal in sorted(profile.literals, key=_literal_sort_key)
            )
            profile_formula = (
                profile_parts[0]
                if len(profile_parts) == 1
                else conjunction(*profile_parts)
            )
            formula = conjunction(formula, implies(aux_atom, profile_formula))
        cardinality_constraints.append((aux_predicate, "=", profile.size))

    aux_atoms = tuple(ctx.atom(pred, ctx.variable("X")) for pred in aux_predicates)
    at_least_one = disjunction(*aux_atoms)
    no_two = true()
    for left_idx in range(len(aux_atoms)):
        for right_idx in range(left_idx):
            no_two = conjunction(
                no_two,
                disjunction(
                    neg(aux_atoms[left_idx]),
                    neg(aux_atoms[right_idx]),
                ),
            )
    return conjunction(formula, at_least_one, no_two), tuple(cardinality_constraints)


def _literal_to_formula(literal: Literal, ctx: FOLContext) -> Formula:
    atom = ctx.atom(literal.predicate, ctx.variable("X"))
    return atom if literal.positive else ctx.neg(atom)


def _literal_sort_key(literal: Literal) -> tuple[str, bool]:
    return (str(literal.predicate), literal.positive)


def _profile_sort_key(profile: frozenset[Literal]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted(_literal_sort_key(literal) for literal in profile))


__all__ = [
    "UnaryCcsEncoding",
    "build_profile_capacity_constraint",
    "build_unary_ccs_encoding",
    "reduce_unary_evidence_to_cardinality_constraints",
    "reduce_unary_evidence_to_profile_capacity",
]
