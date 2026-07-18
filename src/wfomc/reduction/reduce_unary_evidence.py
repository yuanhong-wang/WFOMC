"""Unary-evidence CCS helpers."""

from __future__ import annotations

from wfomc.evidence.profile import ProfileCapacityConstraint
from wfomc.fol import FOLContext, Formula, Literal


def build_ccs_formula_and_constraints(
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


__all__ = ["build_ccs_formula_and_constraints"]
