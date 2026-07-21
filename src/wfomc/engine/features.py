"""Domain-free source and reduced-problem feature analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wfomc.fol import Formula, FormulaKind, Predicate, constants
from wfomc.fol import predicates as _formula_predicates
from wfomc.fol import walk
from wfomc.stages import FeatureSet

if TYPE_CHECKING:
    from wfomc.fol.normal_form.c2.norm_form import C2NormalForm
    from wfomc.problem import Problem
    from wfomc.stages import ReducedProblem


def analyze_problem_features(problem: "Problem") -> FeatureSet:
    """Analyze capabilities intrinsic to one source problem."""

    return _analyze_features(
        sentence=problem.sentence,
        has_unary_evidence=problem.has_unary_evidence,
        has_binary_evidence=problem.has_binary_evidence,
    )


def analyze_reduced_features(problem: "ReducedProblem") -> FeatureSet:
    """Analyze capabilities of one logical branch after reduction."""

    from wfomc.fol import true

    sentence = problem.normal_form.qf_formula
    if sentence is None:
        sentence = true()
    return _analyze_features(
        sentence=sentence,
        has_unary_evidence=(
            problem.has_unary_evidence or problem.has_profile_capacity_constraint
        ),
        has_binary_evidence=problem.has_binary_evidence,
        normal_form=problem.normal_form,
    )


def _analyze_features(
    *,
    sentence: Formula,
    has_unary_evidence: bool,
    has_binary_evidence: bool,
    normal_form: "C2NormalForm | None" = None,
) -> FeatureSet:
    predicates = set(_formula_predicates(sentence))
    predicate_arities = _sentence_predicate_arities(sentence)
    named_constants = tuple(sorted(str(constant) for constant in constants(sentence)))
    leq, predecessor_predicates, circular = _order_predicates(
        predicates,
        predicate_arities,
    )

    return FeatureSet(
        has_c2_counting=_has_c2_counting(normal_form, sentence),
        has_mod_counting=_formula_has_mod_counting(sentence),
        has_unary_evidence=has_unary_evidence,
        has_binary_evidence=has_binary_evidence,
        leq_predicate=leq,
        predecessor_predicates=predecessor_predicates,
        circular_predecessor_predicate=circular,
        named_constants=named_constants,
    )


def _has_c2_counting(
    normal_form: "C2NormalForm | None",
    sentence: Formula,
) -> bool:
    if normal_form is not None and normal_form.has_counting:
        return True
    return _formula_has_counting(sentence)


def _sentence_predicate_arities(formula: Formula) -> dict[str, set[int]]:
    arities: dict[str, set[int]] = {}
    for node in walk(formula):
        if node.op == FormulaKind.ATOM:
            args = node.args
            if args:
                arities.setdefault(_predicate_name(args[0]), set()).add(len(args) - 1)

    for predicate in _formula_predicates(formula):
        arity = _predicate_arity(predicate)
        if arity is not None:
            arities.setdefault(_predicate_name(predicate), set()).add(arity)
    return arities


def _order_predicates(
    predicates: set[object],
    predicate_arities: dict[str, set[int]],
) -> tuple[Predicate | None, tuple[tuple[int, Predicate], ...], Predicate | None]:
    leq = None
    predecessor_by_order: dict[int, Predicate] = {}
    circular = None
    for predicate in predicates:
        if not isinstance(predicate, Predicate) or not _predicate_has_arity(
            predicate, 2, predicate_arities
        ):
            continue
        name = predicate.name
        if name == "LEQ":
            leq = predicate
        elif name == "CIRCULAR_PRED":
            circular = predicate
        elif name == "PRED":
            predecessor_by_order[1] = predicate
        elif name.startswith("PRED") and name[4:].isdigit():
            predecessor_by_order[int(name[4:])] = predicate

    if circular is not None and not predecessor_by_order:
        predecessor_by_order[1] = circular
    if leq is None and (predecessor_by_order or circular is not None):
        from wfomc.fol import FOLContext

        leq = FOLContext().predicate("LEQ", 2)
    return leq, tuple(sorted(predecessor_by_order.items())), circular


def _predicate_name(predicate: object) -> str:
    return predicate.name if isinstance(predicate, Predicate) else str(predicate)


def _predicate_arity(predicate: object) -> int | None:
    return predicate.arity if isinstance(predicate, Predicate) else None


def _predicate_has_arity(
    predicate: object,
    arity: int,
    predicate_arities: dict[str, set[int]],
) -> bool:
    explicit_arity = _predicate_arity(predicate)
    if explicit_arity is not None:
        return explicit_arity == arity
    return arity in predicate_arities.get(_predicate_name(predicate), set())


def _formula_has_mod_counting(formula: Formula) -> bool:
    return any(
        node.op == FormulaKind.COUNT and node.args[1] == "mod"
        for node in walk(formula)
    )


def _formula_has_counting(formula: Formula) -> bool:
    return any(node.op == FormulaKind.COUNT for node in walk(formula))


__all__ = [
    "analyze_problem_features",
    "analyze_reduced_features",
]
