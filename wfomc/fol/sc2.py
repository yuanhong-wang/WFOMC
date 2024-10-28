from __future__ import annotations

from functools import reduce
from logzero import logger
from typing import Callable, Union

from wfomc.fol.utils import new_predicate, new_scott_predicate
from .syntax import *
from .syntax import FOLSyntaxError


class SC2(Formula):
    def __init__(self, uni_formula: QuantifiedFormula = None,
                       ext_formulas: list[QuantifiedFormula] = None,
                       cnt_formulas: list[QuantifiedFormula] = None):
        self.uni_formula: QuantifiedFormula = uni_formula
        self.ext_formulas: list[QuantifiedFormula] = ext_formulas or []
        self.cnt_formulas: list[QuantifiedFormula] = cnt_formulas or []
        self.index = 0

    def contain_existential_quantifier(self) -> bool:
        return len(self.ext_formulas) > 0

    def contain_counting_quantifier(self) -> bool:
        return len(self.cnt_formulas) > 0

    def append_ext(self, formula: QuantifiedFormula):
        self.ext_formulas.append(formula)

    def append_cnt(self, formula: QuantifiedFormula):
        self.cnt_formulas.append(formula)

    def preds(self):
        p = self.uni_formula.preds() if self.uni_formula is not None else set()
        return reduce(lambda x, y: x.union(y), map(lambda f: f.preds(), self.ext_formulas), p)

    def pred_by_name(self, name: str) -> Union[Pred, None]:
        for pred in self.preds():
            if pred.name == name:
                return pred
        return None

    def __str__(self) -> str:
        s = ''
        if self.uni_formula is not None:
            s += f'Universally quantified formula: {self.uni_formula}'
        if self.ext_formulas:
            s += '\nExistentially quantified formulas:\n'
            s += '\n'.join(map(str, self.ext_formulas))
        if self.cnt_formulas:
            s += '\nCounting quantified formulas:\n'
            s += '\n'.join(map(str, self.cnt_formulas))
        return s

    def __repr__(self) -> str:
        return str(self)


def dfs(formula: Formula, f: Callable[[Formula], Formula]) -> Formula:
    if isinstance(formula, QFFormula):
        formula, additional_formulas = f(formula)
        return formula, additional_formulas
    elif isinstance(formula, QuantifiedFormula):
        quantified_formula, additional_formulas = dfs(formula.quantified_formula, f)
        formula.quantified_formula = quantified_formula
        formula, a_formulas = f(formula)
        additional_formulas.extend(a_formulas)
        return formula, additional_formulas
    else:
        if isinstance(formula, Negation):
            sub_formula, additional_formulas = dfs(formula.sub_formula, f)
            formula = ~sub_formula
            formula, a_formulas = f(formula)
            additional_formulas.extend(a_formulas)
            return formula, additional_formulas
        else:
            left_formula, additional_formulas = dfs(formula.left_formula, f)
            right_formula, a_formulas = dfs(formula.right_formula, f)
            formula = formula.op(left_formula, right_formula)
            additional_formulas.extend(a_formulas)
            formula, a_formulas = f(formula)
            additional_formulas.extend(a_formulas)
            return formula, additional_formulas


def bfs(formula: Formula, f: Callable[[Formula], Formula]) -> Formula:
    formula, additional_formulas = f(formula)
    if isinstance(formula, QFFormula):
        return formula, additional_formulas
    elif isinstance(formula, QuantifiedFormula):
        formula.quantified_formula, a_formulas = bfs(formula.quantified_formula, f)
        additional_formulas.extend(a_formulas)
        return formula, additional_formulas
    else:
        if isinstance(formula, Negation):
            sub_formula, a_formulas = bfs(formula.sub_formula, f)
            formula = ~sub_formula
            additional_formulas.extend(a_formulas)
            return formula, additional_formulas
        else:
            left_formula, a_formulas = bfs(formula.left_formula, f)
            additional_formulas.extend(a_formulas)
            right_formula, a_formulas = bfs(formula.right_formula, f)
            formula = formula.op(left_formula, right_formula)
            additional_formulas.extend(a_formulas)
            return formula, additional_formulas


def transformer(name: str, formula_clses: tuple[type[Formula]]):
    def decorator(f):
        def wrapper(formula: Formula):
            if isinstance(formula, formula_clses):
                logger.debug("%s: %s", name, formula)
                formula = f(formula)
                if isinstance(formula, tuple):
                    return formula
                return formula, []
            return formula, []
        return wrapper
    return decorator


@transformer("Convert implication and equivalence", (Implication, Equivalence))
def convert_implies_equiv(formula: Formula) -> Formula:
    """
    After this transformation, the formula should not contain any implication or equivalence
    """
    if isinstance(formula, Implication):
        return ~formula.left_formula | formula.right_formula
    elif isinstance(formula, Equivalence):
        return (~formula.left_formula | formula.right_formula) & \
            (~formula.right_formula | formula.left_formula)


@transformer("Push negation", (Negation,))
def push_negation(formula: Formula) -> Formula:
    """
    After this transformation, the negation only appears in quantifer-free formulas
    """
    sub_formula = formula.sub_formula
    if isinstance(sub_formula, Negation):
        # NOTE: in case of multiple negations, e.g., ~~~A(x), we remove them in one go
        return push_negation(~sub_formula.sub_formula)
    elif isinstance(sub_formula, BinaryFormula):
        if isinstance(sub_formula, Conjunction):
            return ~sub_formula.left_formula | ~sub_formula.right_formula
        elif isinstance(sub_formula, Disjunction):
            return ~sub_formula.left_formula & ~sub_formula.right_formula
    elif isinstance(sub_formula, QuantifiedFormula):
        return QuantifiedFormula(
            sub_formula.quantifier_scope.complement(),
            ~sub_formula.quantified_formula
        )


@transformer("Push quantifier-free formula", (BinaryFormula,))
def push_qfformula(formula: Formula) -> Formula:
    """
    After this transformation, the quantifier-free formula only appears in quantified formulas
    That is, the resulting formula only contains quantified formulas and compound formulas
    """
    left = formula.left_formula
    right = formula.right_formula
    if isinstance(right, QFFormula):
        left, right = right, left
    if not isinstance(left, QFFormula):
        return formula
    if isinstance(right, BinaryFormula):
        _left = right.left_formula
        _right = right.right_formula
        if isinstance(right, type(formula)):
            #       &             |
            #      / \    or     / \
            #     F   &         F   |
            if isinstance(_left, BinaryFormula):
                return formula.op(_left, formula.op(left, _right))
            else:
                return formula.op(_right, formula.op(left, _left))
        else:
            # distribute quantifier over conjunction/disjunction
            #       &             |
            #      / \    or     / \
            #     F   |         F   &
            return right.op(
                formula.op(left, _left),
                formula.op(left, _right)
            )
    elif isinstance(right, QuantifiedFormula):
        #       &             |
        #      / \    or     / \
        #     F  Qx:        F  Qx:
        if right.quantified_var not in left.vars():
            return QuantifiedFormula(
                right.quantifier_scope,
                formula.op(right.quantified_formula, left)
            )
        else:
            raise FOLSyntaxError(
                'Not support variable renaming yet, '
                'please rename the variable {} in {}'.format(
                    right.quantified_var, left)
            )
    else:
        raise FOLSyntaxError(
            'Should not reach here'
        )


@transformer("Pop quantifier", (Conjunction, Disjunction))
def pop_quantifier_once(formula: Formula) -> Formula:
    left = formula.left_formula
    right = formula.right_formula
    if isinstance(right, QuantifiedFormula):
        left, right = right, left
    # assert isinstance(left, QuantifiedFormula)
    if not isinstance(left, QuantifiedFormula):
        return formula
    if isinstance(right, QuantifiedFormula):
        if left.quantifier_scope == right.quantifier_scope and \
                isinstance(right.quantifier_scope, Universal):
            return QuantifiedFormula(
                left.quantifier_scope,
                formula.op(left.quantified_formula, right.quantified_formula)
            )
    elif isinstance(right, QFFormula) and not isinstance(left.quantifier_scope, Counting):
        if left.quantified_var not in right.vars():
            return QuantifiedFormula(
                left.quantifier_scope,
                formula.op(left.quantified_formula, right)
            )
        else:
            raise FOLSyntaxError(
                'Not support variable renaming yet, '
                'please rename the variable {} in {}'.format(
                    left.quantified_var, right)
            )

    return formula


@transformer("Distribute quantifier", (QuantifiedFormula, ))
def distribute_quantifier(formula: Formula) -> Formula:
    """
    After this transformation, the quantified formula should not contain any compound formula
    """
    quantified_formula = formula.quantified_formula
    if isinstance(quantified_formula, CompoundFormula):
        if (
            isinstance(quantified_formula, Conjunction) and
            isinstance(formula.quantifier_scope, Universal)
        ) or (
            isinstance(quantified_formula, Disjunction) and
            isinstance(formula.quantifier_scope, Existential)
        ):
            return quantified_formula.op(
                QuantifiedFormula(
                    formula.quantifier_scope,
                    quantified_formula.left_formula
                ),
                QuantifiedFormula(
                    formula.quantifier_scope,
                    quantified_formula.right_formula
                )
            )
        else:
            raise FOLSyntaxError(
                'Not support quantifier distribution for {}'.format(
                    quantified_formula)
            )
    return formula


@transformer("Replace disjunction", (Disjunction, ))
def replace_disjunction(formula: Formula) -> Formula:
    left = formula.left_formula
    right = formula.right_formula
    additional_formulas = []
    if isinstance(left, QuantifiedFormula):
        vars = left.free_vars()
        aux_pred = new_predicate(len(vars), AUXILIARY_PRED_NAME)
        aux_atom = aux_pred(*vars)
        additional_formulas.append(left.equivalent(aux_atom))
        left = aux_atom
    if isinstance(right, QuantifiedFormula):
        vars = right.free_vars()
        aux_pred = new_predicate(len(vars), AUXILIARY_PRED_NAME)
        aux_atom = aux_pred(*vars)
        additional_formulas.append(right.equivalent(aux_atom))
        right = aux_atom
    return left | right, additional_formulas


@transformer('Remove existential quantifier', (QuantifiedFormula,))
def remove_existential_quantifier(formula: Formula) -> Formula:
    # NOTE: only remove existential quantifier, we don't want
    # to complicate the resulting formulas
    assert not isinstance(formula.quantified_formula, CompoundFormula), \
        'Compound formula should be distributed already'
    # NOTE: here we only support FO2, i.e., quantified formulas
    # with two recursive structures
    additional_formulas = []
    if isinstance(formula.quantifier_scope, Existential):
        quantified_formula = formula.quantified_formula
        if isinstance(quantified_formula, QFFormula):
            return formula
            # free_vars = formula.free_vars()
            # aux_pred = new_scott_predicate(len(free_vars))
            # aux_atom = aux_pred(*free_vars)
            # additional_formula = formula.equivalent(aux_atom)
            # for var in free_vars:
            #     additional_formula = QuantifiedFormula(
            #         Universal(var), additional_formula
            #     )
            # additional_formulas.append(additional_formula)
            # return aux_atom, additional_formulas
        elif isinstance(quantified_formula.quantifier_scope, (Universal, Existential)):
            # \exists X \forall Y: f(X,Y) ==>
            # \exists X: S(X) & \forall X\forall Y: (S(X) <-> f(X,Y))
            quantified_var = formula.quantified_var
            aux_pred = new_scott_predicate(1)
            aux_atom = aux_pred(quantified_var)
            additional_formula = quantified_formula.equivalent(aux_atom)
            additional_formula = QuantifiedFormula(
                Universal(quantified_var), additional_formula
            )
            additional_formulas.append(additional_formula)
            formula.quantified_formula = aux_atom
            # formula, a_formulas = remove_quantifier(formula)
            # additional_formulas.extend(a_formulas)
            return formula, additional_formulas
        else:
            raise FOLSyntaxError(
                'Not support quantified formula with more than two quantifiers'
            )
    return formula, additional_formulas


def rename_variables(formula: Formula, depth: int,
                     default_vars: list[Var],
                     substitution: dict[Var, Term]) -> Formula:
    if isinstance(formula, QFFormula):
        return formula.substitute(substitution)
    elif isinstance(formula, QuantifiedFormula):
        quantified_var = formula.quantified_var
        if isinstance(formula.quantifier_scope, Counting):
            quantifier_scope = type(formula.quantifier_scope)(
                default_vars[depth],
                formula.quantifier_scope.comparator,
                formula.quantifier_scope.count_param
            )
        else:
            quantifier_scope = type(formula.quantifier_scope)(
                default_vars[depth]
            )
        substitution[quantified_var] = default_vars[depth]
        quantified_formula = rename_variables(
            formula.quantified_formula,
            depth+1,
            default_vars,
            substitution
        )
        return QuantifiedFormula(quantifier_scope, quantified_formula)
    else:
        if isinstance(formula, Negation):
            return ~rename_variables(formula.sub_formula, depth, default_vars, substitution)
        elif isinstance(formula, BinaryFormula):
            return formula.op(
                rename_variables(formula.left_formula, depth, default_vars, substitution),
                rename_variables(formula.right_formula, depth, default_vars, substitution)
            )


def pop_quantifier(formula: Formula) -> Formula:
    # poping quantifier twice is enough for FO2
    formula, _ = dfs(formula, pop_quantifier_once)
    formula, _ = dfs(formula, pop_quantifier_once)
    return formula


@transformer("Check all conjunction", (BinaryFormula, ))
def check_all_conjunction(formula: Formula) -> bool:
    if not isinstance(formula, Conjunction):
        raise FOLSyntaxError(
            f'Found {type(formula).__name__} formula: {formula}'
        )
    return formula


def standardize(formula: Formula) -> Formula:
    """
    Standardize the given formula to a compound of quantified formulas
    """
    formula, _ = dfs(formula, convert_implies_equiv)
    logger.debug("After convert implication and equivalence: %s", formula)
    formula, _ = bfs(formula, push_negation)
    logger.debug("After push negation: %s", formula)
    formula, _ = bfs(formula, distribute_quantifier)
    logger.debug("After distribute quantifier: %s", formula)
    # formula, _ = bfs(formula, push_qfformula)
    # logger.debug("After push quantified-free formula: %s", formula.pretty())
    # formula, _ = bfs(formula, pop_quantifier)
    # logger.debug("After pop quantifier: %s", formula.pretty())
    return formula


def to_sc2(formula: Formula) -> SC2:
    """
    The formula must satisify that a compound formula has at most one quantifier-free subformula
    """
    if isinstance(formula, QFFormula):
        raise FOLSyntaxError(
            f'Found quantified-free formula: {formula}'
        )
    # elif isinstance(formula, QuantifiedFormula):
    #     if isinstance(formula.quantifier_scope, Universal):
    #         return SNF(uni_formula=formula)
    #     elif isinstance(formula.quantifier_scope, Existential):
    #         return SNF(ext_formulas=[formula])
    #     else:
    #         raise FOLSyntaxError(
    #             f'Found counting quantifier {formula.quantifier_scope}'
    #         )

    logger.debug("Before standardize: %s", formula)
    formula = standardize(formula)
    logger.debug("After standardize: \n%s", pretty_print(formula))
    formula, additional_formulas = dfs(formula, replace_disjunction)
    for additional_formula in additional_formulas:
        formula = formula & additional_formula
    formula = standardize(formula)
    logger.debug("After replace disjunction: \n%s", pretty_print(formula))
    formula, additional_formulas = dfs(formula, remove_existential_quantifier)
    scott_formula = formula # top
    for additional_formula in additional_formulas:
        scott_formula = scott_formula & additional_formula
    logger.debug("After remove existential quantifier: \n%s", pretty_print(scott_formula))
    formula = standardize(scott_formula)
    logger.debug("After standardize: \n%s", pretty_print(formula))
    formula = rename_variables(
        formula, 0, [U, V, W], {}
    )
    formula = rename_variables(
        formula, 0, [X, Y, Z], {}
    )
    logger.debug("After rename variables: \n%s", pretty_print(formula))
    # TODO: disable due to https://github.com/lucienwang1009/lifted_sampling_fo2/issues/8
    # formula = pop_quantifier(formula)
    # logger.debug("After pop quantifier: \n%s", pretty_print(formula))
    # here, the formula must only contain conjunctions
    bfs(formula, check_all_conjunction)

    sc2 = SC2()
    uni_formulas: list[QFFormula] = []
    uni_quantifier_scopes: list[Universal] = []
    def collect_formula(formula: Formula,
                        quantifier_scopes: tuple[Quantifier] = ()):
        if isinstance(formula, QuantifiedFormula):
            quantifier_scopes += (formula.quantifier_scope, )
            collect_formula(formula.quantified_formula, quantifier_scopes)
        elif isinstance(formula, QFFormula):
            if all(isinstance(quantifier_scope, Universal) for quantifier_scope in quantifier_scopes):
                uni_formulas.append(formula)
                uni_quantifier_scopes.append(quantifier_scopes)
            elif any(isinstance(quantifier_scope, Counting) for quantifier_scope in quantifier_scopes):
                collected_formula = reduce(
                    lambda x, y: QuantifiedFormula(y, x),
                    quantifier_scopes[::-1],
                    formula
                )
                if len(quantifier_scopes) == 2 and \
                    isinstance(quantifier_scopes[0], Universal) and \
                        isinstance(quantifier_scopes[1], Counting):
                            sc2.append_cnt(collected_formula)
                else:
                    raise FOLSyntaxError(f"Not support fomula \"{collected_formula}\"")
            else:
                collected_formula = reduce(
                    lambda x, y: QuantifiedFormula(y, x),
                    quantifier_scopes[::-1],
                    formula
                )
                sc2.append_ext(collected_formula)
        elif isinstance(formula, Conjunction):
            collect_formula(formula.left_formula, quantifier_scopes)
            collect_formula(formula.right_formula, quantifier_scopes)
        else:
            raise FOLSyntaxError(
                f'Found {type(formula).__name__} formula: {formula}'
            )
    collect_formula(formula)
    uni_formula = top
    for formula in uni_formulas:
        uni_formula &= formula
    if uni_formula == top:
        sc2.uni_formula = top
    else:
        max_uni_quantifier_scope = max(uni_quantifier_scopes, key=len)
        sc2.uni_formula = reduce(
            lambda x, y: QuantifiedFormula(y, x),
            max_uni_quantifier_scope[::-1],
            uni_formula
        )
    return sc2
