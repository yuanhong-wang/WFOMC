from lark import Transformer, Lark
from enum import Enum
from wfomc.fol.sc2 import to_sc2
from wfomc.fol.syntax import Existential
from wfomc.fol.utils import exactly_one

from wfomc.parser.fol_grammar import function_free_logic_grammar
from wfomc.fol.syntax import *


QuantifiersEnum = Enum('QuantifiersEnum', ['UNIVERSAL', 'EXISTENTIAL', 'COUNTING'])

Quantifiers = {
    QuantifiersEnum.UNIVERSAL: Universal,
    QuantifiersEnum.EXISTENTIAL: Existential,
    QuantifiersEnum.COUNTING: Counting
}

class FOLTransformer(Transformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name2pred = {}

    def constant(self, args):
        return Const(args[0].value)

    def variable(self, args):
        return Var(args[0].value)

    def terms(self, args):
        return list(args)

    def predicate(self, args):
        return args[0].value

    def atomic_ffl(self, args):
        pred_name, _, terms, _ = args[:]
        if terms is None:
            terms = []
        pred = Pred(pred_name, len(terms))
        self.name2pred[pred_name] = pred
        return pred(*terms)

    def parenthesis(self, args):
        return args[1]

    def disjunction(self, args):
        return args[0] | args[-1]

    def conjunction(self, args):
        return args[0] & args[-1]

    def implication(self, args):
        return args[0].implies(args[-1])

    def equivalence(self, args):
        return args[0].equivalent(args[-1])

    def negation(self, args):
        return ~args[-1]

    def universal_quantifier(self, args):
        return (QuantifiersEnum.UNIVERSAL, [])

    def existential_quantifier(self, args):
        return (QuantifiersEnum.EXISTENTIAL, [])

    def equality(self, args):
        return '='

    def nequality(self, args):
        return '!='

    def le(self, args):
        return '<='

    def ge(self, args):
        return '>='

    def lt(self, args):
        return '<'

    def gt(self, args):
        return '>'

    def count_parameter(self, args):
        param = int(args[0])
        assert param >= 0, "Counting parameter must be non-negative"
        return param

    def counting_quantifier(self, args):
        return QuantifiersEnum.COUNTING, args

    def quantifier_variable(self, args):
        quantifier = args[0][0]
        if quantifier == QuantifiersEnum.UNIVERSAL:
            return Universal(args[1])
        elif quantifier == QuantifiersEnum.EXISTENTIAL:
            return Existential(args[1])
        elif quantifier == QuantifiersEnum.COUNTING:
            comparator, count_param = args[0][1]
            return Counting(args[1], comparator, count_param)

    def quantification(self, args):
        quantifier, _, formula, _ = args[:]
        return QuantifiedFormula(quantifier, formula)

    def exactlyone(self, args):
        predicates = args[1]
        predicates = [Pred(p, 1) for p in predicates]
        self.name2pred.update(
            (p.name, p) for p in predicates
        )
        return exactly_one(predicates)

    def predicates(self, args):
        return list(args)


def parse(text: str) -> Formula:
    fol_parser = Lark(function_free_logic_grammar, start='ffl')
    tree = fol_parser.parse(text)
    transformer = FOLTransformer()
    formula = transformer.transform(tree)
    return formula


if __name__ == '__main__':
    text = r"""
        # \forall X: (\forall Y: ((fr(X,Y) -> fr(Y,X))))
        \forall X: (\forall Y: (fr(X,Y) & sm(X) -> sm(Y)))
        & \forall Y: (P(Y) <-> \exists X: (fr(X,Y)))
        & \exists X: (\exists Y: (fr(X,Y) -> P(X)))
        # | \forall X: (P(X) -> \exists_{=2} Y: (fr(X,Y)))
        | \exists Y: (\forall X: (fr(X, Y)))
        # & \forall X: ((\forall Y: (fr(X,Y))) -> Q(X))
        # \forall X: ((P(X) -> A) <-> (~~~\forall Z: (~~fr(X,Z)) -> \forall Q: (R(Q))))
        # & \forall X: (P(X) <-> \exists Y: (Q(Y) & R(X,Y)))
    """
    text = r"""
        (A -> (\forall X: (P(X)) & \forall Y: (Q(Y))))
        & \forall X: (P(X) -> \exists Y: (f(X,Y)))
        | \forall X: (\exists Y: (f(X,Y)) <-> \forall Y: (f(X, Y)))
    """
    text = r"""
        \forall X: (\forall Y: (A(X,Y)) | \forall Y: (B(X,Y)))
    """
    fol_parser = Lark(function_free_logic_grammar,
                      start='ffl')
    tree = fol_parser.parse(text)
    formula = FOLTransformer().transform(tree)
    formula = to_sc2(formula)
    print(formula)
    # print(formula.atoms())
    # print(formula.vars())
    # print(formula.consts())
    # print(formula.formula)
    # print(list(formula.formula.formula.models()))
    # print(formula.substitute({Var('X'): Const('tommy')}))
