from collections import defaultdict
from fractions import Fraction

from lark import Transformer


class CCTransformer(Transformer):
    # NOTE: only support LRA expr
    """
    cardinality_constraints: cardinality_constraint*
    cardinality_constraint: cc_expr comparator NUMBER
    ?cc_expr: left_parenthesis cc_expr right_parenthesis -> parenthesis
        | cc_atom -> cc_atomic_expr
        | cc_expr "+" cc_atom -> cc_add
        | cc_expr "-" cc_atom -> cc_sub
    cc_atom: "|" predicate "|"
        | NUMBER "|" predicate "|"
    """
    def cardinality_constraints(self, args):
        return list(args)

    def cardinality_constraint(self, args):
        expr, comparator, number = args
        return expr, comparator, _number(number)

    def cc_atom(self, args):
        if len(args) == 1:
            return 1, args[0]
        return _number(args[0][0]), args[1]

    def cc_atomic_expr(self, args):
        coef, pred = args[0]
        expr = defaultdict(int)
        expr[pred] = coef
        return expr

    def cc_add(self, args):
        expr, atom = args
        coef, pred = atom
        expr[pred] += coef
        return expr

    def cc_sub(self, args):
        expr, atom = args
        coef, pred = atom
        expr[pred] -= coef
        return expr


def _number(value):
    rational = Fraction(str(value))
    if rational.denominator == 1:
        return rational.numerator
    return rational


CCTransfomer = CCTransformer


__all__ = ["CCTransformer", "CCTransfomer"]
