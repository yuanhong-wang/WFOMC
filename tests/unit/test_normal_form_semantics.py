"""Small-domain semantic checks for C2 definitional normalization."""

from __future__ import annotations

from itertools import product

import pytest

from wfomc.fol import (
    And,
    Atom,
    BoolConst,
    CountingQuantifier,
    Eq,
    FOLContext,
    Iff,
    Implies,
    Not,
    Or,
    Quantifier,
    QuantifierKind,
    Variable,
    free_vars,
)
from wfomc.fol.normal_form import C2NormalForm, normalize
from wfomc.fol.normal_form.c2.norm_form import CountSection, ForallCountSection


Interpretation = dict[object, frozenset[tuple[int, ...]]]


@pytest.mark.parametrize("domain_size", [0, 1, 2])
def test_structure_preserving_normalization_matches_global_count_semantics(
    domain_size: int,
):
    ctx = FOLContext()
    x = ctx.variable("X")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 0)
    exact = ctx.count(x, "=", 1, p(x))
    formulas = (
        ctx.iff(exact, q()),
        ctx.neg(ctx.count(x, "mod", (1, 2), p(x))),
    )
    domain = tuple(range(domain_size))

    for interpretation in _interpretations(domain, (p, q)):
        for formula in formulas:
            assert _evaluate(formula, interpretation, domain) == _normal_form_holds(
                normalize(formula), interpretation, domain
            )


@pytest.mark.parametrize("domain_size", [0, 1, 2])
def test_structure_preserving_normalization_matches_row_count_semantics(
    domain_size: int,
):
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    p = ctx.predicate("P", 1)
    relation = ctx.predicate("R", 2)
    formula = ctx.forall(
        x,
        ctx.disjunction(p(x), ctx.count(y, "=", 1, relation(x, y))),
    )
    normal_form = normalize(formula)
    domain = tuple(range(domain_size))

    for interpretation in _interpretations(domain, (p, relation)):
        assert _evaluate(formula, interpretation, domain) == _normal_form_holds(
            normal_form, interpretation, domain
        )


def _interpretations(
    domain: tuple[int, ...], predicates: tuple[object, ...]
):
    ground_atoms = tuple(
        (predicate, terms)
        for predicate in predicates
        for terms in product(domain, repeat=predicate.arity)
    )
    for values in product((False, True), repeat=len(ground_atoms)):
        interpretation: dict[object, set[tuple[int, ...]]] = {
            predicate: set() for predicate in predicates
        }
        for (predicate, terms), value in zip(ground_atoms, values):
            if value:
                interpretation[predicate].add(terms)
        yield {
            predicate: frozenset(extension)
            for predicate, extension in interpretation.items()
        }


def _normal_form_holds(
    normal_form: C2NormalForm,
    interpretation: Interpretation,
    domain: tuple[int, ...],
) -> bool:
    extended = _extend_count_markers(normal_form, interpretation, domain)
    if normal_form.qf_formula is not None:
        variables = tuple(sorted(free_vars(normal_form.qf_formula), key=str))
        if not all(
            _evaluate(
                normal_form.qf_formula,
                extended,
                domain,
                dict(zip(variables, values)),
            )
            for values in product(domain, repeat=len(variables))
        ):
            return False
    if not all(
        _evaluate(formula, extended, domain)
        for formula in (*normal_form.forall_exists, *normal_form.exists)
    ):
        return False
    if not all(
        _count_section_holds(section, extended, domain)
        for section in normal_form.counts
    ):
        return False
    return all(
        _forall_count_section_holds(section, extended, domain)
        for section in normal_form.forall_counts
    )


def _extend_count_markers(
    normal_form: C2NormalForm,
    interpretation: Interpretation,
    domain: tuple[int, ...],
) -> Interpretation:
    extended = dict(interpretation)
    pending = list(normal_form.count_definitions)
    marker_predicates = {definition.marker.predicate for definition in pending}
    while pending:
        ready = [
            definition
            for definition in pending
            if definition.section.body.predicate not in marker_predicates
            or definition.section.body.predicate in extended
        ]
        if not ready:
            raise AssertionError("cyclic count definitions")
        for definition in ready:
            section = definition.section
            if isinstance(section, CountSection):
                extension = (
                    frozenset({()})
                    if _count_section_holds(section, extended, domain)
                    else frozenset()
                )
            else:
                extension = frozenset(
                    (outer,)
                    for outer in domain
                    if _row_count_holds(section, outer, extended, domain)
                )
            extended[definition.marker.predicate] = extension
            pending.remove(definition)
    return extended


def _count_section_holds(
    section: CountSection,
    interpretation: Interpretation,
    domain: tuple[int, ...],
) -> bool:
    count = sum(
        _evaluate(
            section.body,
            interpretation,
            domain,
            {section.counted_var: value},
        )
        for value in domain
    )
    return _compare_count(count, section.comparator, section.count)


def _forall_count_section_holds(
    section: ForallCountSection,
    interpretation: Interpretation,
    domain: tuple[int, ...],
) -> bool:
    return all(
        _row_count_holds(section, outer, interpretation, domain)
        for outer in domain
    )


def _row_count_holds(
    section: ForallCountSection,
    outer: int,
    interpretation: Interpretation,
    domain: tuple[int, ...],
) -> bool:
    count = sum(
        _evaluate(
            section.body,
            interpretation,
            domain,
            {section.outer_var: outer, section.counted_var: counted},
        )
        for counted in domain
    )
    return _compare_count(count, section.comparator, section.count)


def _evaluate(
    formula: object,
    interpretation: Interpretation,
    domain: tuple[int, ...],
    env: dict[object, int] | None = None,
) -> bool:
    env = {} if env is None else env
    if isinstance(formula, BoolConst):
        return formula.value
    if isinstance(formula, Atom):
        terms = tuple(env[term] if isinstance(term, Variable) else term for term in formula.terms)
        return terms in interpretation.get(formula.predicate, frozenset())
    if isinstance(formula, Eq):
        left = env[formula.left] if isinstance(formula.left, Variable) else formula.left
        right = env[formula.right] if isinstance(formula.right, Variable) else formula.right
        return left == right
    if isinstance(formula, Not):
        return not _evaluate(formula.body, interpretation, domain, env)
    if isinstance(formula, And):
        return all(_evaluate(arg, interpretation, domain, env) for arg in formula.args)
    if isinstance(formula, Or):
        return any(_evaluate(arg, interpretation, domain, env) for arg in formula.args)
    if isinstance(formula, Implies):
        return not _evaluate(formula.left, interpretation, domain, env) or _evaluate(
            formula.right, interpretation, domain, env
        )
    if isinstance(formula, Iff):
        return _evaluate(formula.left, interpretation, domain, env) == _evaluate(
            formula.right, interpretation, domain, env
        )
    if isinstance(formula, Quantifier):
        variable = formula.variables[0]
        values = (
            _evaluate(formula.body, interpretation, domain, {**env, variable: value})
            for value in domain
        )
        return all(values) if formula.kind is QuantifierKind.FORALL else any(values)
    if isinstance(formula, CountingQuantifier):
        count = sum(
            _evaluate(
                formula.body,
                interpretation,
                domain,
                {**env, formula.variable: value},
            )
            for value in domain
        )
        return _compare_count(count, formula.comparator, formula.count)
    raise TypeError(f"Unsupported formula in reference evaluator: {formula!r}")


def _compare_count(value: int, comparator: str, count: object) -> bool:
    if comparator == "mod":
        remainder, modulus = count
        return value % modulus == remainder
    return {
        "=": value == count,
        "!=": value != count,
        "<": value < count,
        "<=": value <= count,
        ">": value > count,
        ">=": value >= count,
    }[comparator]
