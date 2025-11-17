"""
定义 m-odd-degree-graph 问题的 WFOMC 输入。
手动创建，然后拼接。而不是从文件解析，以确保正确性。
"""

from wfomc.fol.syntax import QuantifiedFormula, Universal, Counting
from wfomc.fol.syntax import Conjunction, Implication
from wfomc.problems import CardinalityConstraint
from wfomc.fol.syntax import Const, X, Y, QFFormula, AtomicFormula, Pred
from wfomc.fol.syntax import Pred, Const, X, Y
from wfomc.fol.sc2 import SC2
from wfomc.problems import WFOMCProblem


def create_odd_input():
    # 1. 定义谓词 (Define Predicates)
    E = Pred('E', 2)
    Odd = Pred('Odd', 1)

    # 2. 定义公式的各个部分 (Define parts of the formula)
    # \forall X: (~E(X,X))
    f1 = QuantifiedFormula(Universal(X), ~E(X, X))

    # \forall X: (\forall Y: (E(X,Y) -> E(Y,X)))
    inner_f2 = QuantifiedFormula(Universal(Y), Implication(E(X, Y), E(Y, X)))
    f2 = QuantifiedFormula(Universal(X), inner_f2)

    # \forall X: (Odd(X) -> (\exists_{1 mod 2} Y: (E(X, Y))))
    f3_part1 = QuantifiedFormula(
        Universal(X),
        Implication(
            Odd(X),
            QuantifiedFormula(Counting(Y, 'mod', (1, 2)), E(X, Y))
        )
    )

    # \forall X: ((\exists_{1 mod 2} Y: (E(X, Y))) -> Odd(X))
    f3_part2 = QuantifiedFormula(
        Universal(X),
        Implication(
            QuantifiedFormula(Counting(Y, 'mod', (1, 2)), E(X, Y)),
            Odd(X)
        )
    )

    # \exists_{=2} X: (Odd(X))
    f4 = QuantifiedFormula(Counting(X, '=', 2), Odd(X))

    # 3. 创建 Sentence 对象 (Create Sentence object)
    sentence = SC2()
    # 使用 Conjunction (&) 合并所有纯粹的 forall 公式
    sentence.uni_formula = Conjunction(f1, f2)
    # 放入所有包含 counting 的公式
    sentence.cnt_formulas = [f3_part1, f3_part2, f4]

    # 4. 创建基数约束 (Create Cardinality Constraint)
    card_constraint = CardinalityConstraint()
    card_constraint.add({E: 1.0}, '=', 2.0)

    # 5. 创建最终的 WFOMCProblem 对象 (Create the final WFOMCProblem object)
    return WFOMCProblem(
        domain=set(),  # 稍后填充
        sentence=sentence,
        weights={},  # 您的脚本不处理权重
        cardinality_constraint=card_constraint,
        unary_evidence={}
    )
