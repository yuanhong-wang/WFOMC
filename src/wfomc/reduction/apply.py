"""Apply the explicit domain-free reduction sequence."""

from __future__ import annotations

from wfomc.options import EvidenceStrategy, ExistentialStrategy
from wfomc.problem import Problem
from wfomc.reduction.cardinality import (
    encode_cardinality_constraints,
    to_reduced_cardinality_constraint,
)
from wfomc.reduction.counting import reduce_counting_quantifiers
from wfomc.reduction.existentials import reduce_existentials
from wfomc.reduction.unary_evidence import reduce_unary_evidence
from wfomc.stages import ReducedProblem


def reduce_problem(
    problem: Problem,
    *,
    evidence_strategy: EvidenceStrategy | None = None,
    existential_strategy: ExistentialStrategy | None = None,
    lower_counting: bool = True,
) -> ReducedProblem:
    """Normalize and apply the selected reductions without binding a domain."""

    reduced = _normalize_problem(problem)
    reduced = reduce_unary_evidence(
        reduced,
        evidence_strategy or EvidenceStrategy.NONE,
    )
    # Lifted algorithms lower existing counting sections here. Incremental3
    # skips this pass, then may turn existentials into native counting sections.
    if lower_counting:
        reduced = reduce_counting_quantifiers(reduced)
    reduced = reduce_existentials(
        reduced,
        strategy=existential_strategy,
    )
    return encode_cardinality_constraints(reduced)


def _normalize_problem(problem: Problem) -> ReducedProblem:
    """Create the initial domain-free reduction state."""

    from wfomc.fol.normal_form import normalize, validate_normal_form

    normal_form = normalize(
        problem.sentence,
        reserved_predicate_names=problem.declared_predicate_names(),
    )
    validate_normal_form(normal_form)
    required = problem.required_domain_constants()
    minimum = max(
        len(required),
        1 if normal_form.requires_nonempty_domain else 0,
    )
    return ReducedProblem(
        normal_form=normal_form,
        weights=dict(problem.weights),
        cardinality_constraints=tuple(
            to_reduced_cardinality_constraint(constraint)
            for constraint in problem.cardinality_constraints.constraints
        ),
        evidence=problem.evidence,
        min_domain_size=minimum,
    )


__all__ = ["reduce_problem"]
