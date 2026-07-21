"""Unary-evidence reduction."""

from __future__ import annotations

from dataclasses import replace

from wfomc.cardinality_constraints import CardinalityTerm, Comparator
from wfomc.evidence.data import UnaryEvidence
from wfomc.evidence.profile import ProfileCapacityConstraint
from wfomc.fol import FOLContext, Formula, Literal
from wfomc.options import EvidenceStrategy
from wfomc.stages import (
    DivideDecoderSpec,
    DomainExpr,
    N,
    ReducedCardinalityConstraint,
    ReducedProblem,
    ReducedProfileConstraint,
)


def reduce_unary_evidence(
    problem: ReducedProblem,
    strategy: EvidenceStrategy,
) -> ReducedProblem:
    """Consume unary evidence according to one resolved strategy."""

    unary = problem.evidence.unary
    if unary.is_empty or strategy in (
        EvidenceStrategy.NONE,
        EvidenceStrategy.GROUND_UNITS,
    ):
        return problem
    nonempty_profiles = _observed_profiles(unary)
    observed_nonempty = sum(size for _literals, size in nonempty_profiles)
    profile = ReducedProfileConstraint(
        tuple(
            (literals, DomainExpr.constant(size))
            for literals, size in nonempty_profiles
        )
        + ((frozenset(), N - observed_nonempty),)
    )
    return _apply_profile_strategy(problem, profile, strategy)


def _observed_profiles(
    evidence: UnaryEvidence,
) -> tuple[tuple[frozenset[Literal], int], ...]:
    element_to_literals: dict[object, set[Literal]] = {}
    for literal in evidence.literals:
        ground = literal.to_ground_literal()
        constant = ground.terms[0]
        normalized = literal.to_profile_literal()
        element_literals = element_to_literals.setdefault(constant, set())
        if ~normalized in element_literals:
            raise ValueError(
                f"Evidence must be consistent for {constant}: {literal.predicate}"
            )
        element_literals.add(normalized)
    sizes: dict[frozenset[Literal], int] = {}
    for literals in element_to_literals.values():
        profile = frozenset(literals)
        sizes[profile] = sizes.get(profile, 0) + 1
    return tuple(
        (profile, sizes[profile]) for profile in sorted(sizes, key=_profile_sort_key)
    )


def _apply_profile_strategy(
    problem: ReducedProblem,
    profile: ReducedProfileConstraint,
    strategy: EvidenceStrategy,
) -> ReducedProblem:
    cleared_evidence = replace(problem.evidence, unary=UnaryEvidence())
    if strategy is EvidenceStrategy.LIFTED_PROFILES:
        return replace(
            problem,
            profile_constraint=profile,
            evidence=cleared_evidence,
        )
    if strategy is not EvidenceStrategy.CCS:
        raise ValueError(f"Unsupported staged evidence strategy: {strategy}")

    from wfomc.fol import conjunction, true

    encoded_profiles = tuple(item for item in profile.profiles if item[0])
    structural = ReducedProfileConstraint(encoded_profiles).structural_constraint()
    formula_patch, simple_constraints = _build_ccs_formula_and_constraints(
        structural,
        cover_domain=False,
    )
    qf_formula = problem.normal_form.qf_formula
    if qf_formula is None:
        qf_formula = true()
    profile_constraints = tuple(
        ReducedCardinalityConstraint(
            terms=(CardinalityTerm(predicate),),
            comparator=Comparator(comparator),
            rhs=size,
        )
        for (predicate, comparator, _rhs), (_literals, size) in zip(
            simple_constraints,
            encoded_profiles,
        )
    )
    unmarked_sizes = tuple(size for literals, size in profile.profiles if not literals)
    if len(unmarked_sizes) != 1:
        raise RuntimeError("Unary CCS expects exactly one unmarked profile")
    return replace(
        problem,
        normal_form=replace(
            problem.normal_form,
            qf_formula=conjunction(qf_formula, formula_patch),
        ),
        cardinality_constraints=(problem.cardinality_constraints + profile_constraints),
        evidence=cleared_evidence,
        ccs_profile_markers=tuple(
            predicate for predicate, _comparator, _rhs in simple_constraints
        ),
        ccs_unmarked_size=unmarked_sizes[0],
        decoder_spec=problem.decoder_spec.append(
            DivideDecoderSpec(profile.assignment_count_expr())
        ),
    )


def _build_ccs_formula_and_constraints(
    constraint: ProfileCapacityConstraint,
    *,
    cover_domain: bool = True,
) -> tuple[Formula, tuple[tuple[object, str, int], ...]]:
    """Encode concrete evidence profiles with unary marker predicates."""

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
    at_least_one = disjunction(*aux_atoms) if cover_domain else true()
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
    return tuple(
        sorted((str(literal.predicate), literal.positive) for literal in profile)
    )


__all__ = ["reduce_unary_evidence"]
