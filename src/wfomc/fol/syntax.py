"""Typed relational FOL syntax nodes.

This module contains only data definitions for the framework-owned FOL
language: symbols, terms, and formula nodes. Ergonomic constructors live in
``wfomc.fol.dsl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, TypeAlias


@dataclass(frozen=True, slots=True)
class Sort:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    sort: Sort | None = None
    _context: object | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and other.name == self.name

    def __hash__(self) -> int:
        return hash((self.name,))

    def __str__(self) -> str:
        return self.name

    def __deepcopy__(self, memo: dict[int, object]) -> "Variable":
        return self


@dataclass(frozen=True, slots=True)
class Constant:
    name: str
    sort: Sort | None = None
    _context: object | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constant) and other.name == self.name

    def __hash__(self) -> int:
        return hash((self.name,))

    def __str__(self) -> str:
        return self.name

    def __deepcopy__(self, memo: dict[int, object]) -> "Constant":
        return self


Term: TypeAlias = Variable | Constant


@dataclass(frozen=True, slots=True)
class Predicate:
    name: str
    arity: int
    input_sorts: tuple[Sort | None, ...] = ()
    _context: object | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.arity < 0:
            raise ValueError("Predicate arity must be non-negative")
        if self.input_sorts and len(self.input_sorts) != self.arity:
            raise ValueError(
                f"Predicate {self.name} has {len(self.input_sorts)} input sorts "
                f"for arity {self.arity}"
            )

    def __call__(self, *args: object) -> object:
        from .context import context_for

        return context_for(self, *args).atom(self, *args)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Predicate)
            and other.name == self.name
            and other.arity == self.arity
        )

    def __hash__(self) -> int:
        return hash((self.name, self.arity))

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    def identity(self) -> tuple[str, int]:
        return (self.name, self.arity)

    def cache_key_parts(self) -> tuple[str, int]:
        return self.identity()

    def __deepcopy__(self, memo: dict[int, object]) -> "Predicate":
        return self


class FormulaKind(str, Enum):
    ATOM = "atom"
    EQ = "eq"
    NOT = "not"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    IFF = "iff"
    FORALL = "forall"
    EXISTS = "exists"
    COUNT = "count"
    BOOL = "bool"

    def __str__(self) -> str:
        return self.value


class QuantifierKind(str, Enum):
    FORALL = "forall"
    EXISTS = "exists"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ModCount:
    remainder: int
    modulus: int

    def __post_init__(self) -> None:
        if self.modulus <= 0:
            raise ValueError("Modulo count modulus must be positive")
        if self.remainder < 0:
            raise ValueError("Modulo count remainder must be non-negative")

    def __iter__(self):
        yield self.remainder
        yield self.modulus


@dataclass(frozen=True, slots=True)
class Formula:
    _node_id: int = field(
        default=-1,
        compare=False,
        hash=False,
        repr=False,
        kw_only=True,
    )
    _context: object | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
        kw_only=True,
    )

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def op(self) -> FormulaKind:
        raise NotImplementedError

    @property
    def args(self) -> tuple[object, ...]:
        raise NotImplementedError

    def preds(self) -> frozenset[object]:
        from .analysis import predicates

        return predicates(self)

    def consts(self) -> frozenset[object]:
        from .analysis import constants

        return constants(self)

    def free_vars(self) -> frozenset[object]:
        from .analysis import free_vars

        return free_vars(self)

    def vars(self) -> frozenset[object]:
        return self.free_vars()

    def cached_analysis(self, key: str, compute):
        """Read or populate this node's owning-context analysis cache."""

        from .context import FOLContext

        if not isinstance(self._context, FOLContext) or self.node_id < 0:
            return compute()
        cache_key = (key, self.node_id)
        if cache_key not in self._context._analysis_cache:
            self._context._analysis_cache[cache_key] = compute()
        return self._context._analysis_cache[cache_key]

    def implies(self, other: object) -> "Formula":
        from .context import context_for

        return context_for(self, other).implies(self, other)

    def equivalent(self, other: object) -> "Formula":
        from .context import context_for

        return context_for(self, other).iff(self, other)

    def __and__(self, other: object) -> "Formula":
        from .context import context_for

        return context_for(self, other).conjunction(self, other)

    def __rand__(self, other: object) -> "Formula":
        from .context import context_for

        return context_for(other, self).conjunction(other, self)

    def __or__(self, other: object) -> "Formula":
        from .context import context_for

        return context_for(self, other).disjunction(self, other)

    def __ror__(self, other: object) -> "Formula":
        from .context import context_for

        return context_for(other, self).disjunction(other, self)

    def __invert__(self) -> "Formula":
        from .context import context_for

        return context_for(self).neg(self)

    def __str__(self) -> str:
        from .pretty import format_formula

        return format_formula(self)

    __repr__ = __str__

    def __deepcopy__(self, memo: dict[int, object]) -> "Formula":
        return self


@dataclass(frozen=True, slots=True)
class Atom(Formula):
    predicate: object
    terms: tuple[object, ...] = ()

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.ATOM

    @property
    def args(self) -> tuple[object, ...]:
        return (self.predicate, *self.terms)

    @property
    def pred(self) -> object:
        return self.predicate

    @property
    def positive(self) -> bool:
        return True

    def make_positive(self) -> "Atom":
        return self


@dataclass(frozen=True, slots=True)
class Eq(Formula):
    left: object
    right: object

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.EQ

    @property
    def args(self) -> tuple[object, ...]:
        return (self.left, self.right)


@dataclass(frozen=True, slots=True)
class Not(Formula):
    body: object

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.NOT

    @property
    def args(self) -> tuple[object, ...]:
        return (self.body,)


@dataclass(frozen=True, slots=True)
class And(Formula):
    args: tuple[object, ...]

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.AND


@dataclass(frozen=True, slots=True)
class Or(Formula):
    args: tuple[object, ...]

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.OR


@dataclass(frozen=True, slots=True)
class Implies(Formula):
    left: object
    right: object

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.IMPLIES

    @property
    def args(self) -> tuple[object, ...]:
        return (self.left, self.right)


@dataclass(frozen=True, slots=True)
class Iff(Formula):
    left: object
    right: object

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.IFF

    @property
    def args(self) -> tuple[object, ...]:
        return (self.left, self.right)


@dataclass(frozen=True, slots=True)
class Quantifier(Formula):
    kind: QuantifierKind
    variables: tuple[object, ...]
    body: object

    @property
    def op(self) -> FormulaKind:
        return (
            FormulaKind.FORALL
            if self.kind == QuantifierKind.FORALL
            else FormulaKind.EXISTS
        )

    @property
    def args(self) -> tuple[object, ...]:
        if len(self.variables) == 1:
            return (self.variables[0], self.body)
        return (self.variables, self.body)


@dataclass(frozen=True, slots=True)
class CountingQuantifier(Formula):
    variable: object
    comparator: str
    count: object
    body: object

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.COUNT

    @property
    def args(self) -> tuple[object, ...]:
        return (self.variable, self.comparator, self.count, self.body)


@dataclass(frozen=True, slots=True)
class BoolConst(Formula):
    value: bool

    @property
    def op(self) -> FormulaKind:
        return FormulaKind.BOOL

    @property
    def args(self) -> tuple[object, ...]:
        return (self.value,)

    def __bool__(self) -> bool:
        return self.value


def formula_children(formula: object) -> tuple[object, ...]:
    if not isinstance(formula, Formula):
        return ()
    if isinstance(formula, Not):
        return (formula.body,)
    if isinstance(formula, (And, Or)):
        return tuple(formula.args)
    if isinstance(formula, (Implies, Iff)):
        return (formula.left, formula.right)
    if isinstance(formula, Quantifier):
        return (formula.body,)
    if isinstance(formula, CountingQuantifier):
        return (formula.body,)
    if isinstance(formula, BoolConst):
        return ()
    return ()


def flatten_kind(kind: FormulaKind, formulas: Iterable[object]) -> tuple[object, ...]:
    flattened: list[object] = []
    for formula in formulas:
        if isinstance(formula, Formula) and formula.op == kind:
            flattened.extend(formula.args)
        else:
            flattened.append(formula)
    return tuple(flattened)


__all__ = [
    "And",
    "Atom",
    "BoolConst",
    "Constant",
    "CountingQuantifier",
    "Eq",
    "Formula",
    "FormulaKind",
    "Iff",
    "Implies",
    "ModCount",
    "Not",
    "Or",
    "Predicate",
    "Quantifier",
    "QuantifierKind",
    "Sort",
    "Term",
    "Variable",
    "flatten_kind",
    "formula_children",
]
