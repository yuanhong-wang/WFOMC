

from collections import defaultdict
from lark import Transformer


class CCTransfomer(Transformer):
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
        return expr, comparator, float(number)

    def cc_atom(self, args):
        if len(args) == 1:
            return 1.0, args[0]
        else:
            return float(args[0][0]), args[1]

    def cc_atomic_expr(self, args):
        coef, pred = args[0]
        expr = defaultdict(lambda : 0.0)
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
