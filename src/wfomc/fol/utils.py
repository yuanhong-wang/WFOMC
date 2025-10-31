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


def new_predicate(arity: int, pred_name: str, used_pred_names: set[str] = None) -> Pred:
    """Creates a new predicate with a unique name."""
    global PREDICATES
    # If used_pred_names is provided, use a simple indexed naming scheme
    if used_pred_names is not None:
        i = 0
        name = f'{pred_name}{i}'
        while name in used_pred_names:
            i += 1
            name = f'{pred_name}{i}'
        return Pred(name, arity)

    # Otherwise, use the global PREDICATES dictionary for uniqueness
    name = f'{pred_name}{len(PREDICATES[pred_name])}'
    p = Pred(name, arity)
    PREDICATES[pred_name].append(p)
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
    while (len(ret_vars) < arity):
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


def convert_counting_formula(formula: QuantifiedFormula, domain: set) -> \
        tuple[QFFormula, list[QuantifiedFormula], tuple, int]:
    """
    这个函数负责实现Beat论文中的引理 4，也就是将 ∀X ∃=k Y: R(X,Y) 分解。分解过程中，它会生成 k 个独立的、需要被满足的存在量词公式 ∀X ∃Y: fᵢ(X,Y)。
    """

    inner_formula = formula.quantified_formula
    # Case 1: Unary counting formula, e.g. ∃=k X: φ(X)
    if not isinstance(inner_formula, QuantifiedFormula): # 通过判断内部公式是否还包含量词，来区分是单层还双层结构

        # If inner formula is not a simple unary atom, introduce an auxiliary predicate.
        if not (isinstance(inner_formula, AtomicFormula) and inner_formula.pred.arity == 1):
            raise TypeError(f"Unary counting quantifier requires a unary atomic formula inside, but got {inner_formula}")
        # 从量词作用域中提取计数相关的参数
        quantifier_scope = formula.quantifier_scope
        comparator = quantifier_scope.comparator # 比较符，如 '=' 或 '<='
        count_param = int(quantifier_scope.count_param) # 计数值 k

        # Create cardinality constraints directly on this predicate
        predicate_to_constrain = inner_formula.pred # 确定需要施加基数约束的谓词
        cardinality_constraint = (predicate_to_constrain, comparator, count_param) # 直接根据提取的参数创建基数约束

        # For pure unary constraints, there are no additional formulas or repeat factors # 对于一元约束，转换非常直接，不需要引入新的复杂公式或重复因子
        return top, [], cardinality_constraint, 1 # 返回：一个恒为真的公式，一个空的公式列表，创建的基数约束，以及重复因子 1

    else:
        # ========= Handling the binary counting formula (for example, ∀X ∃=k Y: R(X,Y)) - Following the old logic =========
        uni_formula = top # 初始化返回值：一个恒为真的全称量词公式和一个空的存在量词公式列表
        ext_formulas = []
        # 提取内层的计数公式部分和计数量词
        cnt_quantified_formula = formula.quantified_formula.quantified_formula
        cnt_quantifier = formula.quantified_formula.quantifier_scope
        count_param = cnt_quantifier.count_param # 计数值 k

        repeat_factor = (math.factorial(count_param)) ** len(domain) # 计算重复因子，用于修正因转换引入的重复计数。其值为 (k!)^n

        # Follow the steps in "A Complexity Upper Bound for
        # Some Weighted First-Order Model Counting Problems With Counting Quantifiers"
        # 步骤 (2): 引入一个二元辅助谓词 aux(X,Y)，并添加公理使其与原始内层公式等价
        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
        aux_atom = aux_pred(X, Y)
        uni_formula = uni_formula & (cnt_quantified_formula.equivalent(aux_atom))
        # 步骤 (3): 引入 k 个新的“子辅助”谓词 aux_i(X,Y)
        sub_aux_preds, sub_aux_atoms = [], []
        for i in range(count_param): # 创建第 i 个子辅助谓词
            aux_pred_i = new_predicate(2, f'{aux_pred.name}_')
            aux_atom_i = aux_pred_i(X, Y)
            sub_aux_preds.append(aux_pred_i)
            sub_aux_atoms.append(aux_atom_i)
            sub_ext_formula = QuantifiedFormula(Existential(Y), aux_atom_i) # 为每个子辅助谓词生成一个简单的存在量词公式 ∀X ∃Y: aux_i(X,Y)
            sub_ext_formula = QuantifiedFormula(Universal(X), sub_ext_formula)
            ext_formulas.append(sub_ext_formula) # 将这个新公式添加到列表中，等待后续的skolem
        # 步骤 (4): 添加排他性公理，确保对于任意 (X,Y)，最多只有一个 aux_i 为真
        for i in range(count_param):
            for j in range(i):
                uni_formula = uni_formula & (~sub_aux_atoms[i] | ~sub_aux_atoms[j])
        # 步骤 (5): 添加覆盖性公理，确保 aux(X,Y) 为真当且仅当 k 个 aux_i 中至少有一个为真
        or_sub_aux_atoms = QFFormula(False)
        for atom in sub_aux_atoms:
            or_sub_aux_atoms = or_sub_aux_atoms | atom
        uni_formula = uni_formula & or_sub_aux_atoms.equivalent(aux_atom)
        # 步骤 (6): 创建全局基数约束，要求 aux 谓词的总数为 n * k
        cardinality_constraint = (aux_pred, '=', len(domain) * count_param)

        return uni_formula, ext_formulas, cardinality_constraint, repeat_factor