from lark import Lark
from wfomc.network.mln import MLN
from wfomc.fol.syntax import Const, Pred
from wfomc.network.constraint import CardinalityConstraint
from wfomc.parser.cardinality_constraints_parser import CCTransfomer
from wfomc.parser.mln_grammar import grammar
from wfomc.utils import Rational


from wfomc.parser.fol_parser import FOLTransformer
from wfomc.problems import MLNProblem

from wfomc.fol.syntax import *

class MLNTransformer(FOLTransformer, CCTransfomer):

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

    def weighting(self, args):
        return float(args[0])

    def rules(self, args):
        rules = args
        weightings = []
        formulas = []
        for w, r in rules:
            weightings.append(w)
            formulas.append(r)
        return weightings, formulas

    def rule(self, args):
        w, r = args[0]
        return w, r

    def hard_rule(self, args):
        return float('inf'), args[0]

    def soft_rule(self, args):
        return args[0], args[1]

    def mln(self, args):
        rules = args[0]
        domain = args[1][1] # Only one definition domain is supported
        cardinality_constraints = args[2]

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

        return rules, domain, cardinality_constraint

def parse(text: str) -> MLNProblem:
    mln_parser = Lark(grammar,
                        start='mln')
    tree = mln_parser.parse(text)
    (rules, domain, cardinality_constraint) = MLNTransformer().transform(tree)

    return MLNProblem(
        rules,
        domain,
        cardinality_constraint
    )
