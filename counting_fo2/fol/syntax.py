from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Iterable
from collections import OrderedDict
from PrettyPrint import PrettyPrintTree

from . import boolean_algebra as backend


__all__ = [
    'Pred',
    'Term',
    'Var',
    'Const',
    'Formula',
    'QFFormula',
    'AtomicFormula',
    'Quantifier',
    'Universal',
    'Existential',
    'Counting',
    'QuantifiedFormula',
    'CompoundFormula',
    'Conjunction',
    'Disjunction',
    'Implication',
    'Equivalence',
    'Negation',
    'BinaryFormula',
    'SCOTT_PREDICATE_PREFIX',
    'AUXILIARY_PRED_NAME',
    'TSEITIN_PRED_NAME',
    'SKOLEM_PRED_NAME',
    'EVIDOM_PRED_NAME',
    'PREDS_FOR_EXISTENTIAL',
    'pretty_print',
    'X', 'Y', 'Z',
    'U', 'V', 'W',
    'top', 'bot',
]


class FOLSyntaxError(Exception):
    pass


class Term(object):
    """
    First-order logic terms, including constants and variables
    """
    name: str


SCOTT_PREDICATE_PREFIX = '@scott'
AUXILIARY_PRED_NAME = '@aux'
TSEITIN_PRED_NAME = '@tseitin'
SKOLEM_PRED_NAME = '@skolem'
EVIDOM_PRED_NAME = '@evidom'
PREDS_FOR_EXISTENTIAL = [
    TSEITIN_PRED_NAME, SKOLEM_PRED_NAME, EVIDOM_PRED_NAME
]


RESERVED_PRED_NAMES: tuple[str] = (
    'true',
    'false',
    SCOTT_PREDICATE_PREFIX,
    AUXILIARY_PRED_NAME,
    TSEITIN_PRED_NAME,
    SKOLEM_PRED_NAME,
    EVIDOM_PRED_NAME
)

RESERVED_VAR_NAMES: tuple[str] = (
)

@dataclass(frozen=True)
class Pred:
    """
    Predicate
    """
    name: str
    arity: int

    def __post_init__(self):
        if self.name.split('_')[0].lower() in RESERVED_PRED_NAMES:
            raise FOLSyntaxError("Predicate name cannot be %s" % self.name)
        if self.arity < 0:
            raise FOLSyntaxError("Arity must be a natural number")

    def __call__(self, *args: Term):
        # NOTE(hack): the callable obj cannot be the column of dataframe
        # if len(args) == 0 or not isinstance(args[0], (Var, Const)):
        #     return self
        if self.arity != len(args):
            raise FOLSyntaxError(
                "Mismatching number of arguments and predicate %s: %s != %s", str(self), self.arity, len(args))
        return AtomicFormula(pred=self, args=tuple(args), positive=True)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Var(Term):
    """
    Variable
    """
    name: str

    def __post_init__(self):
        if self.name in RESERVED_VAR_NAMES:
            raise FOLSyntaxError("Variable name cannot be %s" % self.name)

    def substitute(self, substitution: dict[Var, Term]) -> Term:
        if self in substitution:
            return substitution[self]
        return self

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Const(Term):
    """
    Constant
    """
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


class Formula(object):
    """
    Base class for first-order logic formulas
    """
    ...


@dataclass(frozen=True)
class QFFormula(Formula):
    """
    Quantifier-free formula
    """
    expr: backend.Expr

    def __invert__(self) -> QFFormula:
        return QFFormula(backend.Not(self.expr))

    def __or__(self, other: Formula) -> Formula:
        if isinstance(other, (Top, Bot)):
            return other | self
        if isinstance(other, QFFormula):
            return QFFormula(backend.Or(self.expr, other.expr))
        return other | self

    def __and__(self, other: Formula) -> Formula:
        if isinstance(other, (Top, Bot)):
            return other & self
        if isinstance(other, QFFormula):
            return QFFormula(backend.And(self.expr, other.expr))
        return other & self

    def implies(self, other: Formula) -> Formula:
        if isinstance(other, (Top, Bot)):
            return ~self | other
        if isinstance(other, QFFormula):
            return QFFormula(backend.Implies(self.expr, other.expr))
        return ~self | other

    def equivalent(self, other: Formula) -> Formula:
        if isinstance(other, (Top, Bot)):
            return (~self | other) & (self | ~other)
        if isinstance(other, QFFormula):
            return QFFormula(backend.Equivalent(self.expr, other.expr))
        else:
            return other.equivalent(self)

    def __str__(self) -> str:
        return str(self.expr)

    def __repr__(self) -> str:
        return str(self)

    def atoms(self) -> frozenset[AtomicFormula]:
        return backend.get_atoms(self.expr)

    def terms(self) -> Iterable[Term]:
        for atom in self.atoms():
            for term in atom.args:
                yield term

    def vars(self) -> frozenset[Var]:
        return frozenset(filter(lambda x: isinstance(x, Var), self.terms()))

    def free_vars(self) -> frozenset[Var]:
        return self.vars()

    def consts(self) -> frozenset[Const]:
        return frozenset(filter(lambda x: isinstance(x, Const), self.terms()))

    def preds(self) -> frozenset[Pred]:
        return frozenset(atom.pred for atom in self.atoms())

    def satisfiable(self) -> bool:
        return backend.satisfiable(self.expr)

    def models(self) -> Iterable[frozenset[AtomicFormula]]:
        """
        Yield all models of the formula

        :rtype Iterable[frozenset[Lit]]: models
        """
        if not self.satisfiable():
            raise RuntimeError("Formula is not satisfiable")

        for model in backend.get_models(self.expr):
            yield frozenset(
                backend.get_atom(symbol) if value else ~backend.get_atom(symbol)
                for symbol, value in model.items()
            )

    def substitute(self, substitution: dict[Term, Term]) -> QFFormula:
        atom_substitutions = OrderedDict()
        for atom in self.atoms():
            atom_substitutions[atom.expr] = atom.substitute(substitution).expr
        return QFFormula(backend.substitute(self.expr, atom_substitutions))

    def sub_nullary_atoms(self, substitution: dict[AtomicFormula, bool]) -> QFFormula:
        substitution = dict((atom.expr, value) for atom, value in substitution.items())
        return QFFormula(backend.substitute(self.expr, substitution))

    def simplify(self) -> QFFormula:
        return QFFormula(backend.simplify(self.expr))


@dataclass(frozen=True)
class AtomicFormula(QFFormula):
    """
    Atomic formula, i.e. a predicate applied to a tuple of terms.
    It is a subclass of QFFormula.
    It actually acts as a literal.
    """
    pred: Pred
    args: tuple[Term]
    positive: bool
    expr: backend.Expr = field(init=False, default=None,
                               hash=False, compare=False)

    def __post_init__(self):
        if len(self.args) != self.pred.arity:
            raise FOLSyntaxError(
                "Number of terms does not match the predicate's arity")
        atom = self
        if not self.positive:
            atom = ~self
        expr = backend.get_symbol(atom)
        expr = expr if self.positive else backend.Not(expr)
        object.__setattr__(self, 'expr', expr)

    @functools.lru_cache(maxsize=None)
    def __invert__(self):
        return AtomicFormula(self.pred, self.args, not self.positive)

    @functools.lru_cache(maxsize=None)
    def make_positive(self):
        if self.positive:
            return self
        return AtomicFormula(self.pred, self.args, True)

    def __str__(self):
        s = '{}({})'.format(self.pred,
                               ','.join([str(arg) for arg in self.args]))
        return s if self.positive else '~' + s

    def __repr__(self):
        return str(self)

    def vars(self) -> frozenset[Var]:
        return frozenset(filter(lambda x: isinstance(x, Var), self.args))

    def consts(self) -> frozenset[Const]:
        return frozenset(filter(lambda x: isinstance(x, Const), self.args))

    def substitute(self, substitution: dict[Term, Term]) -> AtomicFormula:
        substituted_args = []
        for arg in self.args:
            if arg in substitution:
                substituted_args.append(substitution[arg])
            else:
                substituted_args.append(arg)
        return AtomicFormula(self.pred, tuple(substituted_args), self.positive)

    def simplify(self) -> QFFormula:
        return self


@dataclass(frozen=True)
class Bot(QFFormula):
    expr: backend.Expr = field(init=False, default=None)

    def __and__(self, other: Formula) -> Formula:
        return self

    def __or__(self, other: Formula) -> Formula:
        return other

    def __invert__(self) -> QFFormula:
        return top

    def implies(self, other: Formula) -> Formula:
        return top

    def equivalent(self, other: Formula) -> Formula:
        return ~other

    def __str__(self) -> str:
        return '⊥'

    def preds(self) -> frozenset[Pred]:
        return frozenset()


@dataclass(frozen=True)
class Top(QFFormula):
    expr: backend.Expr = field(init=False, default=None)

    def __and__(self, other: Formula) -> Formula:
        return other

    def __or__(self, other: Formula) -> Formula:
        return self

    def __invert__(self) -> QFFormula:
        return bot

    def implies(self, other: Formula) -> Formula:
        return other

    def equivalent(self, other: Formula) -> Formula:
        return other

    def __str__(self) -> str:
        return '⊤'

    def preds(self) -> frozenset[Pred]:
        return frozenset()


@dataclass(frozen=True)
class Quantifier(object):
    quantifier: str = field(init=False)
    quantified_var: Var

    def __str__(self):
        return '{} {}'.format(self.quantifier, self.quantified_var)

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Universal(Quantifier):
    def __post_init__(self):
        object.__setattr__(self, 'quantifier', '\\forall')

    def complement(self) -> Existential:
        return Existential(self.quantified_var)


@dataclass(frozen=True)
class Existential(Quantifier):
    def __post_init__(self):
        object.__setattr__(self, 'quantifier', '\\exists')

    def complement(self) -> Universal:
        return Universal(self.quantified_var)


@dataclass(frozen=True)
class Counting(Quantifier):
    comparator: str
    count_param: int

    def __post_init__(self):
        assert self.comparator in ['='], 'Only equality is supported'
        object.__setattr__(self, 'quantifier', '\\exists')

    def complement(self) -> Counting:
        raise FOLSyntaxError('Complement of counting quantifier is not supported')

    def __str__(self):
        return '{}_{{{}{}}} {}'.format(
            self.quantifier, self.comparator,
            self.count_param, self.quantified_var
        )


class QuantifiedFormula(Formula):
    """
    Quantified formula, e.g. \\forall x P(x),
    \\exists x P(x) and \\exists_{=2} x P(x)
    """
    def __init__(self, quantifier_scope: Quantifier, quantified_formula: Formula):
        self.quantifier_scope = quantifier_scope
        self.quantified_formula = quantified_formula

    def vars(self) -> frozenset[Var]:
        return self.quantified_formula.vars()

    @property
    def quantified_var(self) -> Var:
        return self.quantifier_scope.quantified_var

    def free_vars(self) -> frozenset[Var]:
        return self.quantified_formula.free_vars() - frozenset([self.quantified_var])

    def rename(self, substitution: dict[Term, Term]) -> QuantifiedFormula:
        # filter out the quantified variable
        substitution = {k: v for k, v in substitution.items() if k in self.free_vars()}
        inverse_substitution = {v: k for k, v in substitution.items()}
        if self.quantified_var in inverse_substitution:
            raise FOLSyntaxError('Subsituting variable {} with {} will cause collision in the formula: {}'.format(
                inverse_substitution[self.quantified_var],
                self.quantified_var, self
            )
        )
        quantifier_scope = self.quantifier_scope.rename_quantified_var(
            substitution.get(self.quantified_var, self.quantified_var)
        )
        if isinstance(self.quantified_formula, QFFormula):
            return QuantifiedFormula(quantifier_scope, self.quantified_formula.substitute(substitution))
        elif isinstance(self.quantified_formula, QuantifiedFormula):
            return QuantifiedFormula(quantifier_scope, self.quantified_formula.rename(substitution))
        else:
            raise FOLSyntaxError('Compound quantified formula is not supported')

    def consts(self) -> frozenset[Const]:
        return self.quantified_formula.consts()

    def atoms(self) -> frozenset[AtomicFormula]:
        return self.quantified_formula.atoms()

    def __invert__(self) -> QuantifiedFormula:
        # return QuantifiedFormula(self.quantifier_scope.complement(), ~self.formula)
        if isinstance(self.quantified_formula, QFFormula):
            return QuantifiedFormula(self.quantifier_scope.complement(), ~self.quantified_formula)
        return Negation(self)

    def __or__(self, other: Formula) -> Formula:
        if isinstance(other, QFFormula) and \
                self.quantified_var not in other.vars() and \
                not isinstance(self.quantifier_scope, Counting):
            return QuantifiedFormula(self.quantifier_scope, self.quantified_formula | other)
        return Disjunction(self, other)

    def __and__(self, other: Formula) -> Formula:
        if isinstance(other, QFFormula) and \
                self.quantified_var not in other.vars() and \
                not isinstance(self.quantifier_scope, Counting):
            return QuantifiedFormula(self.quantifier_scope, self.quantified_formula & other)
        return Conjunction(self, other)

    def implies(self, other: Formula) -> Formula:
        return Implication(self, other)

    def equivalent(self, other: Formula) -> Formula:
        return Equivalence(self, other)

    def ext_uni_vars(self) -> tuple[frozenset[Var], frozenset[Var]]:
        all_vars = self.vars()
        if self.exist is None:
            ext_vars = None
        else:
            ext_vars = self.exist.quantified_vars
        return (ext_vars, all_vars - ext_vars)

    def preds(self) -> frozenset[Pred]:
        return self.quantified_formula.preds()

    def is_exist(self) -> bool:
        return self.exist is not None

    def __str__(self):
        return '{}: {}'.format(self.quantifier_scope, str(self.quantified_formula))

    def __repr__(self):
        return str(self)


class CompoundFormula(Formula):
    def __init__(self, op_name: str, op: Callable[..., Formula]):
        self.op_name: str = op_name
        self.op: Callable[..., Formula] = op

    def __invert__(self):
        return Negation(self)

    def __or__(self, other):
        return Disjunction(self, other)

    def __and__(self, other):
        return Conjunction(self, other)

    def implies(self, other):
        return Implication(self, other)

    def equivalent(self, other):
        return Equivalence(self, other)


class Negation(CompoundFormula):
    def __init__(self, formula: Formula) -> None:
        super().__init__('~', lambda x: ~x)
        self.sub_formula: Formula = formula

    def __str__(self):
        return f'~{self.sub_formula}'

    def __repr__(self):
        return str(self)

    def vars(self) -> frozenset[Var]:
        return self.sub_formula.vars()


class BinaryFormula(CompoundFormula):
    def __init__(self, op_name: str, op: Callable[[Formula, Formula], Formula],
                 left: Formula, right: Formula) -> None:
        super().__init__(op_name, op)
        self.left_formula: Formula = left
        self.right_formula: Formula = right

    def __str__(self):
        return f'({self.left_formula}) {self.op_name} ({self.right_formula})'

    def __repr__(self):
        return str(self)

    def vars(self) -> frozenset[Var]:
        return self.left_formula.vars() | self.right_formula.vars()


class Conjunction(BinaryFormula):
    def __init__(self, left: Formula, right: Formula) -> None:
        super().__init__('&', lambda x, y: x & y,
                         left, right)


class Disjunction(BinaryFormula):
    def __init__(self, left: Formula, right: Formula) -> None:
        super().__init__('|', lambda x, y: x | y,
                         left, right)


class Implication(BinaryFormula):
    def __init__(self, left: Formula, right: Formula) -> None:
        super().__init__('->', lambda x, y: x.implies(y),
                         left, right)


class Equivalence(BinaryFormula):
    def __init__(self, left: Formula, right: Formula) -> None:
        super().__init__('<->', lambda x, y: x.equivalent(y),
                         left, right)


def pretty_print(formula: Formula) -> None:
    def get_children(formula: Formula) -> list[Formula]:
        if isinstance(formula, QFFormula):
            return []
        elif isinstance(formula, QuantifiedFormula):
            return [formula.quantified_formula]
        elif isinstance(formula, Negation):
            return [formula.sub_formula]
        elif isinstance(formula, BinaryFormula):
            return [formula.left_formula, formula.right_formula]

    def get_value(formula: Formula) -> str:
        if isinstance(formula, QFFormula):
            return str(formula)
        elif isinstance(formula, QuantifiedFormula):
            return f'{formula.quantifier_scope}:'
        elif isinstance(formula, Negation):
            return '~'
        elif isinstance(formula, BinaryFormula):
            return formula.op_name

    pt = PrettyPrintTree(get_children, get_value, return_instead_of_print=True)
    return pt(formula)


X, Y, Z = Var('X'), Var('Y'), Var('Z')
U, V, W = Var('U'), Var('V'), Var('W')
a, b, c = Const('a'), Const('b'), Const('c')
top, bot = Top(), Bot()
