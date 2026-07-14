"""Context-managed construction for framework FOL syntax nodes."""

from __future__ import annotations

from collections.abc import Iterable

from .syntax import (
    And,
    Atom,
    BoolConst,
    CountingQuantifier,
    Eq,
    Formula,
    FormulaKind,
    Iff,
    Implies,
    ModCount,
    Not,
    Or,
    Quantifier,
    QuantifierKind,
    flatten_kind,
)
from .syntax import Constant, Predicate, Sort, Variable


class FOLContext:
    """Arena for interned symbols, terms, formula nodes, and analysis caches."""

    def __init__(self) -> None:
        self._sorts: dict[str, Sort] = {}
        self._variables: dict[tuple[str, Sort | None], Variable] = {}
        self._constants: dict[tuple[str, Sort | None], Constant] = {}
        self._predicates: dict[
            tuple[str, int, tuple[Sort | None, ...]],
            Predicate,
        ] = {}
        self._formulas: dict[tuple[object, tuple[object, ...]], Formula] = {}
        self._next_node_id = 0
        self._analysis_cache: dict[tuple[str, int], object] = {}

    def sort(self, name: str) -> Sort:
        if name not in self._sorts:
            self._sorts[name] = Sort(str(name))
        return self._sorts[name]

    def variable(self, name: str, sort: Sort | None = None) -> Variable:
        key = (str(name), sort)
        if key not in self._variables:
            self._variables[key] = Variable(str(name), sort, _context=self)
        return self._variables[key]

    def vars(self, names: str | Iterable[str]) -> tuple[Variable, ...]:
        if isinstance(names, str):
            names = names.split()
        return tuple(self.variable(name) for name in names)

    def constant(self, name: str, sort: Sort | None = None) -> Constant:
        key = (str(name), sort)
        if key not in self._constants:
            self._constants[key] = Constant(str(name), sort, _context=self)
        return self._constants[key]

    def constants(self, names: str | Iterable[str]) -> tuple[Constant, ...]:
        if isinstance(names, str):
            names = names.split()
        return tuple(self.constant(name) for name in names)

    def predicate(
        self,
        name: str,
        arity: int,
        *,
        input_sorts: tuple[Sort | None, ...] = (),
    ) -> Predicate:
        key = (str(name), int(arity), tuple(input_sorts))
        if key not in self._predicates:
            self._predicates[key] = Predicate(
                str(name),
                int(arity),
                tuple(input_sorts),
                _context=self,
            )
        return self._predicates[key]

    def true(self) -> Formula:
        return self._intern_formula(BoolConst(True))

    def false(self) -> Formula:
        return self._intern_formula(BoolConst(False))

    def atom(self, predicate: str | Predicate, *terms: object) -> Formula:
        if isinstance(predicate, str):
            predicate = self.predicate(predicate, len(terms))
        if len(terms) != predicate.arity:
            raise ValueError(
                f"Mismatching number of arguments and predicate {predicate}: "
                f"{len(terms)} != {predicate.arity}"
            )
        return self._intern_formula(Atom(predicate, tuple(terms)))

    def eq(self, left: object, right: object) -> Formula:
        return self._intern_formula(Eq(left, right))

    def neg(self, formula: object) -> Formula:
        return self._intern_formula(Not(formula))

    def conjunction(self, *formulas: object) -> Formula:
        flattened = flatten_kind(FormulaKind.AND, formulas)
        if not flattened:
            return self.true()
        # filter out true (identity) and short-circuit on false (annihilator)
        kept: list[object] = []
        for formula in flattened:
            if isinstance(formula, BoolConst):
                if not formula.value:
                    return self.false()
                continue  # true — skip
            kept.append(formula)
        if not kept:
            return self.true()
        if len(kept) == 1 and isinstance(kept[0], Formula):
            return kept[0]
        return self._intern_formula(And(tuple(kept)))

    def disjunction(self, *formulas: object) -> Formula:
        flattened = flatten_kind(FormulaKind.OR, formulas)
        if not flattened:
            return self.false()
        # filter out false (identity) and short-circuit on true (annihilator)
        kept: list[object] = []
        for formula in flattened:
            if isinstance(formula, BoolConst):
                if formula.value:
                    return self.true()
                continue  # false — skip
            kept.append(formula)
        if not kept:
            return self.false()
        if len(kept) == 1 and isinstance(kept[0], Formula):
            return kept[0]
        return self._intern_formula(Or(tuple(kept)))

    def implies(self, left: object, right: object) -> Formula:
        return self._intern_formula(Implies(left, right))

    def iff(self, left: object, right: object) -> Formula:
        return self._intern_formula(Iff(left, right))

    def forall(self, variables: object, body: object) -> Formula:
        return self._quantifier(QuantifierKind.FORALL, variables, body)

    def exists(self, variables: object, body: object) -> Formula:
        return self._quantifier(QuantifierKind.EXISTS, variables, body)

    def count(
        self,
        variable: object,
        comparator: str,
        count_param: object,
        body: object,
    ) -> Formula:
        _validate_count(comparator, count_param)
        return self._intern_formula(
            CountingQuantifier(variable, str(comparator), count_param, body)
        )

    def _quantifier(
        self,
        kind: QuantifierKind,
        variables: object,
        body: object,
    ) -> Formula:
        if isinstance(variables, Variable):
            variable_tuple = (variables,)
        elif isinstance(variables, str):
            variable_tuple = (self.variable(variables),)
        elif isinstance(variables, tuple):
            variable_tuple = variables
        elif isinstance(variables, list):
            variable_tuple = tuple(variables)
        else:
            variable_tuple = (variables,)
        if not variable_tuple:
            raise ValueError("Quantifier requires at least one variable")
        return self._intern_formula(Quantifier(kind, tuple(variable_tuple), body))

    def _intern_formula(self, formula: Formula) -> Formula:
        key = _formula_key(formula)
        existing = self._formulas.get(key)
        if existing is not None:
            return existing
        node = formula
        object.__setattr__(node, "_node_id", self._next_node_id)
        object.__setattr__(node, "_context", self)
        self._next_node_id += 1
        self._formulas[key] = node
        return node


def _validate_count(comparator: str, count_param: object) -> None:
    if comparator == "mod":
        if isinstance(count_param, ModCount):
            return
        if not isinstance(count_param, tuple) or len(count_param) != 2:
            raise ValueError("Modulo counting requires a (remainder, modulus) tuple")
        ModCount(int(count_param[0]), int(count_param[1]))
        return
    if isinstance(count_param, int) and count_param < 0:
        raise ValueError("Counting parameter must be non-negative")


def _formula_key(formula: Formula) -> tuple[object, tuple[object, ...]]:
    if isinstance(formula, Atom):
        return (Atom, (formula.predicate, formula.terms))
    if isinstance(formula, Eq):
        return (Eq, (formula.left, formula.right))
    if isinstance(formula, Not):
        return (Not, (formula.body,))
    if isinstance(formula, And):
        return (And, tuple(formula.args))
    if isinstance(formula, Or):
        return (Or, tuple(formula.args))
    if isinstance(formula, Implies):
        return (Implies, (formula.left, formula.right))
    if isinstance(formula, Iff):
        return (Iff, (formula.left, formula.right))
    if isinstance(formula, Quantifier):
        return (Quantifier, (formula.kind, formula.variables, formula.body))
    if isinstance(formula, CountingQuantifier):
        return (
            CountingQuantifier,
            (formula.variable, formula.comparator, formula.count, formula.body),
        )
    return (type(formula), tuple(formula.args))


_DEFAULT_CONTEXT: FOLContext | None = None


def default_context() -> FOLContext:
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = FOLContext()
    return _DEFAULT_CONTEXT


def current_context() -> FOLContext:
    return default_context()


def context_for(*values: object) -> FOLContext:
    for value in values:
        if isinstance(value, (Formula, Variable, Constant, Predicate)):
            if isinstance(value._context, FOLContext):
                return value._context
    return default_context()


__all__ = [
    "FOLContext",
    "context_for",
    "current_context",
    "default_context",
]
