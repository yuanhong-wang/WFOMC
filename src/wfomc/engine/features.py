"""Feature analysis types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from wfomc.fol import Formula, FormulaKind, Predicate, constants as _formula_constants
from wfomc.fol import predicates as _formula_predicates
from wfomc.fol import walk
from wfomc.weights import collect_symbolic_weight_variables

if TYPE_CHECKING:
    from wfomc.fol.normal_form import C2NormalForm
    from wfomc.problem import Problem


@dataclass(frozen=True)
class FeatureSet:
    has_c2_counting: bool = False
    has_mod_counting: bool = False
    has_global_cardinality: bool = False
    has_unary_evidence: bool = False
    has_binary_evidence: bool = False
    has_named_constants: bool = False
    leq_predicate: Predicate | None = None
    predecessor_predicates: tuple[tuple[int, Predicate], ...] = ()
    circular_predecessor_predicate: Predicate | None = None
    requires_symbolic_weights: bool = False
    symbolic_weight_variables: tuple[str, ...] = ()
    named_constants: tuple[str, ...] = ()

    @property
    def has_linear_order(self) -> bool:
        return self.leq_predicate is not None

    @property
    def has_predk(self) -> bool:
        return bool(self.predecessor_predicates)

    @property
    def has_circular_pred(self) -> bool:
        return self.circular_predecessor_predicate is not None


def analyze_features(
    problem: "Problem",
    normal_form: "C2NormalForm | None" = None,
) -> FeatureSet:
    sentence = problem.sentence
    text = str(sentence)
    formulas = tuple(_sentence_formulas(sentence))
    predicates = _sentence_predicates(sentence, formulas)
    predicate_arities = _sentence_predicate_arities(formulas)
    named_constants = tuple(
        sorted(str(const) for const in _sentence_constants(formulas))
    )
    leq, predecessor_predicates, circular = _order_predicates(
        predicates,
        predicate_arities,
    )

    has_c2_counting = _has_c2_counting(sentence, normal_form, text, formulas)
    symbolic_weight_variables = collect_symbolic_weight_variables(problem)
    return FeatureSet(
        has_c2_counting=has_c2_counting,
        has_mod_counting=_has_mod_counting(sentence, text, formulas),
        has_global_cardinality=_has_global_cardinality(problem, normal_form),
        has_unary_evidence=_has_unary_evidence(problem),
        has_binary_evidence=problem.has_binary_evidence,
        has_named_constants=bool(named_constants),
        leq_predicate=leq,
        predecessor_predicates=predecessor_predicates,
        circular_predecessor_predicate=circular,
        requires_symbolic_weights=bool(symbolic_weight_variables),
        symbolic_weight_variables=symbolic_weight_variables,
        named_constants=named_constants,
    )


def _has_c2_counting(
    sentence: object,
    normal_form: "C2NormalForm | None",
    text: str,
    formulas: tuple[Formula, ...] = (),
) -> bool:
    if normal_form is not None and normal_form.has_counting:
        return True
    for formula in formulas:
        if _formula_has_counting(formula):
            return True
    return bool(not formulas and "exists_" in text)


def _has_mod_counting(
    sentence: object,
    text: str,
    formulas: tuple[Formula, ...] = (),
) -> bool:
    for formula in formulas:
        if _formula_has_mod_counting(formula):
            return True
    return bool(not formulas and "mod" in text)


def _has_global_cardinality(
    problem: "Problem", normal_form: "C2NormalForm | None",
) -> bool:
    if problem.has_cardinality_constraints:
        return True
    if normal_form is not None and normal_form.counts:
        return True
    return False


def _has_unary_evidence(problem: "Problem") -> bool:
    return problem.has_unary_evidence or problem.has_profile_capacity_constraint


def _sentence_formulas(sentence: Formula) -> Iterable[Formula]:
    yield sentence


def _sentence_constants(formulas: Iterable[Formula]) -> set[object]:
    constants = set()
    for formula in formulas:
        constants.update(_formula_constants(formula))
    return constants


def _sentence_predicates(
    sentence: Formula, formulas: Iterable[Formula]
) -> set[object]:
    predicates = set()
    for formula in formulas:
        predicates.update(_formula_predicates(formula))
    return predicates


def _sentence_predicate_arities(formulas: Iterable[Formula]) -> dict[str, set[int]]:
    arities: dict[str, set[int]] = {}
    for formula in formulas:
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
    predicates: Iterable[object],
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
    if formula.op == FormulaKind.COUNT:
        comparator = formula.args[1]
        return comparator == "mod"
    for child in walk(formula):
        if child is not formula and child.op == FormulaKind.COUNT:
            comparator = child.args[1]
            if comparator == "mod":
                return True
    return False


def _formula_has_counting(formula: Formula) -> bool:
    if formula.op == FormulaKind.COUNT:
        return True
    return any(
        child is not formula and child.op == FormulaKind.COUNT
        for child in walk(formula)
    )


__all__ = ["FeatureSet", "analyze_features"]
