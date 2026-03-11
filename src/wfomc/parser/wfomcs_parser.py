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
    Parse the WFOMS text into a WFOMCProblem object. 
    Args:
        text (str): A text string containing the definition of the WFOMS problem. 
    Returns:
        WFOMC Problem: A structured object that contains the derived formula, domain, weights, and constraints.
    """
    # 1. Initialize the parser
    wfomcs_parser = Lark(grammar,
                        start='wfomcs') # Create a Lark parser instance. `grammar` is a predefined variable containing the complete grammar rules for the WFOMS language. `start='wfomcs'` specifies that the parsing process should begin from the 'wfomcs' rule in the grammar.
    # 2. Parse the text
    tree = wfomcs_parser.parse(text) # Use the parser created in the previous step to process the input `text` string. If the text conforms to the grammar rules, the `.parse()` method returns a parse tree (Tree) that hierarchically represents the syntactic structure of the input text.
    # 3. Transform the parse tree
    (
        sentence,
        domain,
        weightings,
        cardinality_constraint
    ) = WFOMSTransformer().transform(tree) # `WFOMSTransformer` is a custom class that traverses the parse tree `tree`. For each node in the tree (corresponding to a grammar rule), it calls a method with the same name to process that node, converting the raw text tokens into more meaningful Python objects (such as formulas, sets, dictionaries, etc.). The `.transform()` method ultimately returns a tuple containing the core components after transformation.
    # 4. Associate predicate objects with weights
    pred_weightings = dict(
        (sentence.pred_by_name(pred), weights)
        for pred, weights in weightings.items()
    ) # The keys in the `weightings` dictionary are predicate names as strings, but the algorithm requires the predicate objects themselves. This code creates a new dictionary `pred_weightings`. It iterates over the `weightings` dictionary, and for each predicate name, it uses `sentence.pred_by_name(pred)` to look up and retrieve the corresponding predicate (Pred) object from the parsed formula `sentence`, then uses this object as the key in the new dictionary.
    # 5. Create and return the problem object
    return WFOMCProblem(
        sentence,
        domain,
        pred_weightings,
        cardinality_constraint
    ) # Using all the parsed and processed information, create an instance of the `WFOMCProblem` class. This object encapsulates all parts together, making it convenient for subsequent algorithms to use directly.


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
