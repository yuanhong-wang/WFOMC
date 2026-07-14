"""Counting-quantifier reduction: C2 counting → UFO² + cardinality constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import reduce
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.fol import FOLContext
from wfomc.fol import (
    Formula,
    conjunction,
    disjunction,
    false as _false,
    neg,
    true as _true,
)
from wfomc.fol.normal_form import C2NormalForm
from wfomc.reduction.core import ProblemWithDecoder, divide_decoder

if TYPE_CHECKING:
    from wfomc.algo.core import AlgoOptions
    from wfomc.arithmetic import ArithmeticValue
    from wfomc.fol.normal_form import C2NormalForm
    from wfomc.fol.normal_form.c2.norm_form import CountSection, ForallCountSection
    from wfomc.problem import ReducedProblem


def reduce_counting_quantifiers(
    problem: "ReducedProblem",
    *,
    options: "AlgoOptions",
) -> ProblemWithDecoder:
    from wfomc.cardinality_constraints import combine_cardinality_constraints

    normal_form = problem.normal_form
    qf_formula = normal_form.qf_formula
    if qf_formula is None:
        qf_formula = _true()
    counting = reduce_counting(
        normal_form,
        domain_size=len(problem.domain),
        rational_cls=Fraction,
        reserved_predicate_names=(str(predicate) for predicate in problem.weights),
    )
    weights = dict(problem.weights)
    weights.update(counting.weight_map())
    reduced_normal_form = replace(
        normal_form,
        qf_formula=conjunction(qf_formula, counting.formula_patch),
        counts=(),
        forall_counts=(),
        count_definitions=(),
    )
    reduced_problem = replace(
        problem,
        normal_form=reduced_normal_form,
        weights=weights,
        cardinality_constraints=combine_cardinality_constraints(
            problem.cardinality_constraints,
            counting.cardinality_constraints,
        ),
    )
    return ProblemWithDecoder(reduced_problem, divide_decoder(counting.repeat_factor))


@dataclass(frozen=True)
class CountingReduction:
    formula_patch: Formula
    weights: tuple[tuple[object, tuple["ArithmeticValue", "ArithmeticValue"]], ...]
    cardinality_constraints: CardinalityConstraints
    repeat_factor: int = 1

    def weight_map(
        self,
    ) -> dict[object, tuple["ArithmeticValue", "ArithmeticValue"]]:
        return dict(self.weights)


def can_reduce_counting_to_ufo2(normal_form: "C2NormalForm") -> bool:
    if not normal_form.has_counting:
        return True
    if normal_form.count_definitions:
        return False
    return all(
        _can_reduce_global_count(gc) for gc in normal_form.counts
    ) and all(_can_reduce_row_count(rc) for rc in normal_form.forall_counts)


def reduce_counting(
    normal_form: "C2NormalForm",
    *,
    domain_size: int,
    rational_cls: type,
    reserved_predicate_names: Iterable[str] = (),
) -> CountingReduction:
    if not normal_form.has_counting:
        return CountingReduction(
            formula_patch=_true(),
            weights=(),
            cardinality_constraints=CardinalityConstraints(),
        )
    if not can_reduce_counting_to_ufo2(normal_form):
        raise ValueError("Counting sections not reducible to UFO2 + cardinality")

    ctx = FOLContext()
    formula_patch = _true()
    weights: dict[object, tuple[object, object]] = {}
    constraints: list[LinearCardinalityConstraint] = []
    repeat_factor = 1
    fresh_predicate = _fresh_predicate_factory(
        normal_form,
        ctx,
        reserved_predicate_names,
    )

    for gc in normal_form.counts:
        atom = _count_body_atom(gc.body, arity=1, label="global count")
        constraints.append(
            LinearCardinalityConstraint(
                terms=(CardinalityTerm(_count_body_pred(atom), 1),),
                comparator=Comparator(str(gc.comparator)),
                rhs=int(gc.count),
            )
        )

    for rc in normal_form.forall_counts:
        rf, rw, rcst, rp = reduce_exact_row_count(
            rc,
            domain_size=domain_size,
            ctx=ctx,
            rational_cls=rational_cls,
            fresh_predicate=fresh_predicate,
        )
        formula_patch = conjunction(formula_patch, rf)
        weights.update(rw)
        constraints.append(rcst)
        repeat_factor *= rp

    return CountingReduction(
        formula_patch=formula_patch,
        weights=_sort_weight_items(weights),
        cardinality_constraints=CardinalityConstraints(tuple(constraints)),
        repeat_factor=repeat_factor,
    )


def reduce_exact_row_count(
    row_count: "ForallCountSection",
    *,
    domain_size: int,
    ctx: FOLContext,
    rational_cls: type,
    fresh_predicate: Callable[[str, int], object] | None = None,
) -> tuple[
    Formula, dict[object, tuple[object, object]], LinearCardinalityConstraint, int
]:
    body = _count_body_atom(row_count.body, arity=2, label="row count")
    count = int(row_count.count)
    outer_var = row_count.outer_var
    counted_var = row_count.counted_var
    formula = _true()
    weights: dict[object, tuple[object, object]] = {}
    repeat_factor = (math.factorial(count)) ** domain_size
    if fresh_predicate is None:
        fresh_predicate = _standalone_fresh_predicate(ctx)

    aux_pred = fresh_predicate("__aux", 2)
    aux_atom = ctx.atom(aux_pred, outer_var, counted_var)
    formula = conjunction(formula, body.equivalent(aux_atom))

    sub_aux_atoms = [
        ctx.atom(fresh_predicate(f"{aux_pred.name}_{i}", 2), outer_var, counted_var)
        for i in range(count)
    ]
    for i in range(count):
        for j in range(i):
            formula = conjunction(
                formula,
                disjunction(neg(sub_aux_atoms[i]), neg(sub_aux_atoms[j])),
            )
    or_sub = reduce(
        lambda left, right: disjunction(left, right), sub_aux_atoms, _false()
    )
    formula = conjunction(formula, or_sub.equivalent(aux_atom))

    skolem_preds = []
    skolem_axioms = _true()
    for index, sa in enumerate(sub_aux_atoms):
        sp = fresh_predicate(f"__sk_{index}", 1)
        skolem_preds.append(sp)
        skolem_axioms = conjunction(
            skolem_axioms,
            disjunction(ctx.atom(sp, outer_var), neg(sa)),
        )
        weights[sp] = (rational_cls(1, 1), rational_cls(-1, 1))

    c_preds = [fresh_predicate(f"__C_{j}", 1) for j in range(count + 1)]
    gamma_body = _true()
    for j in range(count + 1):
        ta = [ctx.atom(skolem_preds[h], outer_var) for h in range(j)]
        fa = [neg(ctx.atom(skolem_preds[h], outer_var)) for h in range(j, count)]
        ct = reduce(lambda left, right: conjunction(left, right), ta, _true())
        cf = reduce(lambda left, right: conjunction(left, right), fa, _true())
        gamma_body = conjunction(
            gamma_body,
            ctx.atom(c_preds[j], outer_var).equivalent(conjunction(ct, cf)),
        )
    final_disj = reduce(
        lambda left, right: disjunction(left, right),
        (ctx.atom(p, outer_var) for p in c_preds),
        _false(),
    )
    for j, p in enumerate(c_preds):
        weights[p] = (rational_cls(math.comb(count, j), 1), rational_cls(1, 1))

    return (
        reduce(
            lambda left, right: conjunction(left, right),
            [formula, skolem_axioms, gamma_body, final_disj],
            _true(),
        ),
        weights,
        LinearCardinalityConstraint(
            terms=(CardinalityTerm(aux_pred, 1),),
            comparator=Comparator.EQ,
            rhs=domain_size * count,
        ),
        repeat_factor,
    )


def _fresh_predicate_factory(
    normal_form: C2NormalForm,
    ctx: FOLContext,
    reserved_predicate_names: Iterable[str],
) -> Callable[[str, int], object]:
    from wfomc.fol import predicates

    used_names = set(reserved_predicate_names)
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
        used_names.update(predicate.name for predicate in predicates(formula))
    return _fresh_predicate_from_names(ctx, used_names)


def _standalone_fresh_predicate(
    ctx: FOLContext,
) -> Callable[[str, int], object]:
    return _fresh_predicate_from_names(ctx, set())


def _fresh_predicate_from_names(
    ctx: FOLContext,
    used_names: set[str],
) -> Callable[[str, int], object]:
    def fresh(base: str, arity: int) -> object:
        index = 0
        name = base
        while name in used_names:
            index += 1
            name = f"{base}_{index}"
        used_names.add(name)
        return ctx.predicate(name, arity)

    return fresh


# ---------------------------------------------------------------------------
# Count-body validation helpers (moved from duck-typing accessor layer)
# ---------------------------------------------------------------------------


def _is_count_body_atom(body: object, *, arity: int) -> bool:
    from wfomc.fol import Atom

    return isinstance(body, Atom) and len(body.terms) == arity


def _count_body_atom(body: object, *, arity: int, label: str) -> object:
    if _is_count_body_atom(body, arity=arity):
        return body
    raise TypeError(
        f"{label} reduction requires an atomic body with arity {arity}, got {body}"
    )


def _count_body_pred(body: object) -> object:
    """Return the predicate from a typed count-body :class:`~wfomc.fol.Atom`."""
    from wfomc.fol import Atom

    if isinstance(body, Atom):
        return body.predicate
    raise TypeError(f"Expected typed Atom count body, got {body!r}")


def _sort_weight_items(
    weights: dict[object, tuple[object, object]],
) -> tuple[tuple[object, tuple[object, object]], ...]:
    return tuple(sorted(weights.items(), key=lambda item: str(item[0])))


def _can_reduce_global_count(gc: "CountSection") -> bool:
    return (
        gc.comparator != "mod"
        and isinstance(gc.count, int)
        and _is_count_body_atom(gc.body, arity=1)
    )


def _can_reduce_row_count(rc: "ForallCountSection") -> bool:
    return (
        rc.comparator == "="
        and isinstance(rc.count, int)
        and _is_count_body_atom(rc.body, arity=2)
    )


__all__ = [
    "CountingReduction",
    "can_reduce_counting_to_ufo2",
    "reduce_counting",
    "reduce_counting_quantifiers",
    "reduce_exact_row_count",
]
