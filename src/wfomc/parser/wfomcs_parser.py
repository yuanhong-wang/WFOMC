from __future__ import annotations

from fractions import Fraction

from lark import Lark
from wfomc.fol.sc2 import SC2, to_sc2
from wfomc.fol.syntax import Const, Formula, Pred
from wfomc.network.constraint import CardinalityConstraint
from wfomc.parser.cardinality_constraints_parser import CCTransfomer

from wfomc.parser.fol_parser import FOLTransformer
from wfomc.parser.wfomcs_grammar import grammar
from wfomc.problems import WFOMCProblem
from wfomc.utils import Rational


class WFOMSTransformer(FOLTransformer, CCTransfomer):

    def domain_elements(self, args):
        return list(args)

    def int_domain(self, args):
        return int(args[0])

    def element(self, args):
        return args[0].value

    def set_domain(self, args):
        return set(Const(i) for i in args[0])

    def domain_name(self, args):
        return args[0].value

    def domain(self, args):
        domain_name, domain_spec = args
        if isinstance(domain_spec, int):
            domain_spec = set(
                Const(f'{domain_name}{i}') for i in range(domain_spec)
            )
        return (domain_name, domain_spec)

    def weightings(self, args):
        return dict(args)

    def weighting(self, args):
        return (args[2], (
            Rational(
                Fraction(args[0]).numerator,
                Fraction(args[0]).denominator
            ),
            Rational(
                Fraction(args[1]).numerator,
                Fraction(args[1]).denominator
            )
        ))

    def weight(self, args):
        return str(args[0])

    def wfomcs(self, args) -> \
            tuple[SC2, set[Const], dict[Pred, tuple[Rational, Rational]],
                  CardinalityConstraint]:
        sentence = args[0]
        domain = args[1][1]
        weightings = args[2]
        cardinality_constraints = args[3]
        try:
            sentence = to_sc2(sentence)
        except:
            raise ValueError('Sentence must be a valid SC2 formula.')

        ccs: list[tuple[dict[Pred, float], str, float]] = list()
        if len(cardinality_constraints) > 0:
            for cc in cardinality_constraints:
                new_expr = dict()
                expr, comparator, param = cc
                for pred_name, coef in expr.items():
                    pred = self.name2pred.get(pred_name, None)
                    if not pred:
                        raise ValueError(f'Predicate {pred_name} not found')
                    new_expr[pred] = coef
                ccs.append((new_expr, comparator, param))
            cardinality_constraint = CardinalityConstraint(ccs)
        else:
            cardinality_constraint = None

        return sentence, domain, weightings, cardinality_constraint


def parse(text: str) -> \
        tuple[SC2, set[Const], dict[Pred, tuple[Rational, Rational]], CardinalityConstraint]:
    """
    将 WFOMS 文本解析成一个 WFOMCProblem 对象。

    Args:
        text (str): 包含 WFOMS 问题定义的文本字符串。

    Returns:
        WFOMCProblem: 一个包含了解析后的公式、论域、权重和约束的结构化对象。
    """
    # 1. 初始化解析器
    wfomcs_parser = Lark(grammar,
                        start='wfomcs') # 创建一个 Lark 解析器实例。`grammar` 是一个预定义的变量，包含了 WFOMS 语言的完整语法规则。`start='wfomcs'` 指定了解析过程应该从语法的 'wfomcs' 规则开始。
    # 2. 解析文本
    tree = wfomcs_parser.parse(text) # 使用上一步创建的解析器来处理输入的 `text` 字符串。如果文本符合语法规则，`.parse()` 方法会返回一个解析树（Tree），它以层级结构表示了输入文本的语法结构。
    # 3. 转换解析树
    (
        sentence,
        domain,
        weightings,
        cardinality_constraint
    ) = WFOMSTransformer().transform(tree) # `WFOMSTransformer` 是一个自定义类，它会遍历解析树 `tree`。对于树中的每个节点（对应一条语法规则），它会调用一个同名的方法来处理该节点，将原始的文本标记转换成更有意义的 Python 对象（如公式、集合、字典等）。`.transform()` 方法最终返回一个元组，包含了转换后的核心组件。
    # 4. 关联谓词对象与权重
    pred_weightings = dict(
        (sentence.pred_by_name(pred), weights)
        for pred, weights in weightings.items()
    ) # `weightings` 字典的键是谓词的字符串名称，但算法需要的是谓词对象本身。这段代码创建一个新的字典 `pred_weightings`。它遍历 `weightings` 字典，对于每个谓词名称，它使用 `sentence.pred_by_name(pred)`,从解析好的公式 `sentence` 中查找并获取对应的谓词（Pred）对象，然后将这个对象作为新字典的键。
    # 5. 创建并返回问题对象
    return WFOMCProblem(
        sentence,
        domain,
        pred_weightings,
        cardinality_constraint
    ) # 使用所有解析和处理过的信息，创建一个 `WFOMCProblem` 类的实例。 这个对象将所有部分封装在一起，方便后续的算法直接使用。


if __name__ == '__main__':
    parse(r"""
\forall X: (~E(X,X)) &
\forall X: (\forall Y: (E(X,Y) -> E(Y,X))) &
\forall X: (\exists_{<=3} Y: (E(X,Y)))
V = 5
    """)
#     wfoms = parse(r'''
# \forall X: (\forall Y: (E(X,Y) -> E(Y,X))) &
# \forall X: (~E(X,X)) &
# \forall X: (\exists Y: (E(X,Y)))
#
# v = 10
#     ''')
