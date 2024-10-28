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
        return set(args[0])

    def domain_name(self, args):
        return args[0].value

    def domain(self, args):
        domain_name, domain_spec = args
        if isinstance(domain_spec, int):
            domain_spec = set(f'{domain_name}{i}' for i in range(domain_spec))
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
    Parse wfoms text into WFOMSContext
    """
    wfomcs_parser = Lark(grammar,
                        start='wfomcs')
    tree = wfomcs_parser.parse(text)
    (
        sentence,
        domain,
        weightings,
        cardinality_constraint
    ) = WFOMSTransformer().transform(tree)
    pred_weightings = dict(
        (sentence.pred_by_name(pred), weights)
        for pred, weights in weightings.items()
    )
    return WFOMCProblem(
        sentence,
        domain,
        pred_weightings,
        cardinality_constraint
    )


if __name__ == '__main__':
    wfoms = parse(r'''
\forall X: (\forall Y: (E(X,Y) -> E(Y,X))) &
\forall X: (~E(X,X)) &
\forall X: (\exists Y: (E(X,Y)))

vertices = 10
0.1 1 E
0.2 2 F
0.3 3 G
    ''')
