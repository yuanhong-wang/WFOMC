"""Existential-quantifier reduction."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING

from wfomc.fol import (
    Atom,
    FOLContext,
    Formula,
    Quantifier,
    QuantifierKind,
    conjunction,
    disjunction,
    neg,
    predicates,
)
from wfomc.fol.normal_form import CountSection, ForallCountSection
from wfomc.options import ExistentialStrategy

if TYPE_CHECKING:
    from wfomc.stages import ReducedProblem


def reduce_existentials(
    problem: "ReducedProblem",
    *,
    strategy: ExistentialStrategy | None,
) -> "ReducedProblem":
    strategy = strategy or ExistentialStrategy.SKOLEM
    if strategy is ExistentialStrategy.COUNTING:
        return _reduce_existentials_to_counts(problem)
    if strategy is not ExistentialStrategy.SKOLEM:
        raise ValueError(f"Unsupported existential strategy: {strategy!r}")

    from wfomc.fol import true

    normal_form = problem.normal_form
    qf_formula = normal_form.qf_formula
    if qf_formula is None:
        qf_formula = true()
    qf_formula, weights = _reduce_skolem_existentials(
        qf_formula,
        (*normal_form.forall_exists, *normal_form.exists),
        reserved_predicate_names=frozenset(map(str, problem.weights)),
    )
    merged_weights = dict(problem.weights)
    merged_weights.update(weights)
    return replace(
        problem,
        normal_form=replace(
            normal_form,
            qf_formula=qf_formula,
            forall_exists=(),
            exists=(),
        ),
        weights=merged_weights,
    )


def _reduce_existentials_to_counts(
    problem: "ReducedProblem",
) -> "ReducedProblem":
    """Represent ``exists`` as an explicit ``count >= 1`` section."""

    normal_form = problem.normal_form
    qf_formula = normal_form.qf_formula
    ctx = FOLContext()
    used_names = _normal_form_predicate_names(problem)
    fresh_index = 0
    forall_counts = list(normal_form.forall_counts)
    counts = list(normal_form.counts)

    def count_body(body: Formula, variables: tuple[object, ...]) -> Atom:
        nonlocal fresh_index, qf_formula
        if isinstance(body, Atom) and body.terms == variables:
            return body
        while True:
            name = f"__existential_count{fresh_index}"
            fresh_index += 1
            if name not in used_names:
                used_names.add(name)
                break
        marker = ctx.atom(ctx.predicate(name, len(variables)), *variables)
        definition = marker.equivalent(body)
        qf_formula = (
            definition
            if qf_formula is None
            else conjunction(qf_formula, definition)
        )
        return marker

    for formula in normal_form.forall_exists:
        outer_var = formula.variables[0]
        inner = formula.body
        counted_var = inner.variables[0]
        forall_counts.append(
            ForallCountSection(
                comparator=">=",
                count=1,
                body=count_body(inner.body, (outer_var, counted_var)),
                outer_var=outer_var,
                counted_var=counted_var,
            )
        )

    for formula in normal_form.exists:
        counted_var = formula.variables[0]
        counts.append(
            CountSection(
                comparator=">=",
                count=1,
                body=count_body(formula.body, (counted_var,)),
                counted_var=counted_var,
            )
        )

    return replace(
        problem,
        normal_form=replace(
            normal_form,
            qf_formula=qf_formula,
            forall_exists=(),
            exists=(),
            forall_counts=tuple(forall_counts),
            counts=tuple(counts),
        ),
    )


def _normal_form_predicate_names(problem: "ReducedProblem") -> set[str]:
    normal_form = problem.normal_form
    names = {str(predicate) for predicate in problem.weights}
    formulas = [
        formula
        for formula in (
            normal_form.qf_formula,
            *normal_form.forall_exists,
            *normal_form.exists,
        )
        if formula is not None
    ]
    formulas.extend(section.body for section in normal_form.counts)
    formulas.extend(section.body for section in normal_form.forall_counts)
    for definition in normal_form.count_definitions:
        formulas.extend((definition.marker, definition.section.body))
    for formula in formulas:
        names.update(predicate.name for predicate in predicates(formula))
    return names


def _reduce_skolem_existentials(
    formula: Formula,
    existential_formulas: tuple[Formula, ...],
    *,
    reserved_predicate_names: frozenset[str] = frozenset(),
) -> tuple[Formula, dict[object, tuple[object, object]]]:
    ctx = FOLContext()
    qf = formula
    weights: dict[object, tuple[object, object]] = {}
    used_names = set(reserved_predicate_names)
    for part in (formula, *existential_formulas):
        used_names.update(predicate.name for predicate in predicates(part))
    fresh_index = 0
    for existential in existential_formulas:
        outer_var = None
        inner = existential
        if isinstance(inner, Quantifier) and inner.kind == QuantifierKind.FORALL:
            if len(inner.variables) == 1:
                outer_var = inner.variables[0]
            inner = inner.body
        if isinstance(inner, Quantifier) and inner.kind == QuantifierKind.EXISTS:
            body = inner.body
            while True:
                skolem_name = f"__skolem{fresh_index}"
                fresh_index += 1
                if skolem_name not in used_names:
                    used_names.add(skolem_name)
                    break
            sp = ctx.predicate(skolem_name, 1 if outer_var is not None else 0)
            sa = ctx.atom(sp, outer_var) if outer_var is not None else ctx.atom(sp)
            qf = conjunction(qf, disjunction(sa, neg(body)))
            weights[sp] = (Fraction(1, 1), Fraction(-1, 1))
    return qf, weights


__all__ = ["reduce_existentials"]
