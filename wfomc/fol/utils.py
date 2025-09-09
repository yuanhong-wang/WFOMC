from __future__ import annotations
from functools import reduce
import math
from collections import defaultdict
from .syntax import *

PREDICATES = defaultdict(list)

PERMITTED_VAR_NAMES = range(ord('A'), ord('Z') + 1)
def new_var(exclude: frozenset[Var]) -> Var:
    for c in PERMITTED_VAR_NAMES:
        v = Var(chr(c))
        if v not in exclude:
            return v
    raise RuntimeError(
        "No more variables available"
    )


def new_predicate(arity: int, name: str) -> Pred:
    global PREDICATES
    p = Pred('{}{}'.format(name, len(PREDICATES[name])), arity)
    PREDICATES[name].append(p)
    return p


def get_predicates(name: str) -> list[Pred]:
    return PREDICATES[name]


def new_scott_predicate(arity: int) -> Pred:
    return new_predicate(arity, SCOTT_PREDICATE_PREFIX)


def pad_vars(vars: frozenset[Var], arity: int) -> frozenset[Var]:
    if arity > 3:
        raise RuntimeError(
            "Not support arity > 3"
        )
    ret_vars = set(vars)
    default_vars = [X, Y, Z]
    idx = 0
    while(len(ret_vars) < arity):
        ret_vars.add(default_vars[idx])
        idx += 1
    return frozenset(list(ret_vars)[:arity])


def exactly_one_qf(preds: list[Pred]) -> QFFormula:
    if len(preds) == 1:
        return top
    lits = [p(X) for p in preds]
    # p1(x) v p2(x) v ... v pm(x)
    formula = reduce(lambda x, y: x | y, lits) & \
        exclusive_qf(preds)
    return formula


def exactly_one(preds: list[Pred]) -> QuantifiedFormula:
    if len(preds) == 1:
        return top
    return QuantifiedFormula(Universal(X), exactly_one_qf(preds))


def exclusive_qf(preds: list[Pred]) -> QFFormula:
    if len(preds) == 1:
        return top
    lits = [p(X) for p in preds]
    formula = top
    for i, l1 in enumerate(lits):
        for j, l2 in enumerate(lits):
            if i < j:
                formula = formula & ~(l1 & l2)
    return formula


def exclusive(preds: list[Pred]) -> QuantifiedFormula:
    if len(preds) == 1:
        return top
    return QuantifiedFormula(Universal(X), exclusive_qf(preds))


def new_predicate(arity: int, pred_name: str, used_pred_names: set[str] = None) -> Pred:
    # (This helper function should already exist in utils.py, keep it as is)
    # ...
    # (Assuming the rest of the function is here)
    i = 0
    name = f'{pred_name}{i}'
    if used_pred_names is None:
        return Pred(name, arity)
    while name in used_pred_names:
        i += 1
        name = f'{pred_name}{i}'
    return Pred(name, arity)


def convert_counting_formula(formula: QuantifiedFormula, domain: set) -> \
        tuple[QFFormula, list[QuantifiedFormula], tuple, int]:
    """
    Translates a counting formula to a universally quantified formula,
    existentially quantified formulas, a cardinality constraint, and a repeat factor.

    This new version handles both unary (single layer) and binary (double layer) counting formulas.
    """
    # 检查是单层还是一元公式
    # 灵感来源于 dr_context.py 中的 _split_layer 方法
    if not isinstance(formula.quantified_formula, QuantifiedFormula):
        # ========= 处理一元计数公式 (例如 ∃=k X: P(X)) =========
        inner_formula = formula.quantified_formula

        # 确保内部是一元原子公式
        if not (isinstance(inner_formula, AtomicFormula) and inner_formula.pred.arity == 1):
            raise TypeError(f"Unary counting quantifier requires a unary atomic formula inside, but got {inner_formula}")

        quantifier_scope = formula.quantifier_scope
        comparator = quantifier_scope.comparator
        count_param = int(quantifier_scope.count_param)

        # 直接在该谓词上创建基数约束
        predicate_to_constrain = inner_formula.pred
        cardinality_constraint = (predicate_to_constrain, comparator, count_param)

        # 对于纯一元约束，没有额外的公式或重复因子
        return top, [], cardinality_constraint, 1

    else:
        # ========= 处理二元计数公式 (例如 ∀X ∃=k Y: R(X,Y)) - 沿用旧逻辑 =========
        cnt_quantified_formula = formula.quantified_formula.quantified_formula

        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
        aux_atom = aux_pred(X, Y)

        # f(X,Y) <=> aux(X,Y)
        uni_formula = cnt_quantified_formula.equivalent(aux_atom)

        # ∃=k Y: aux(X,Y)
        quantifier_scope = formula.quantified_formula.quantifier_scope
        comparator = quantifier_scope.comparator
        count_param = int(quantifier_scope.count_param)

        cardinality_constraint = (aux_pred, comparator, count_param)

        return uni_formula, [], cardinality_constraint, 1