from copy import deepcopy
from fractions import Fraction
import math

from wfomc.fol import SC2, to_sc2, AtomicFormula, Const, Pred, top, AUXILIARY_PRED_NAME, \
    Formula, QuantifiedFormula, Universal, Equivalence, new_predicate, formula_to_str
from wfomc.network import CardinalityConstraint
from wfomc.utils import Rational, Expr


class WFOMCProblem(object):
    """
    A weighted first-order model counting problem.
    """

    def __init__(self, sentence: SC2,
                 domain: set[Const],
                 weights: dict[Pred, tuple[Expr, Expr]],
                 cardinality_constraint: CardinalityConstraint = None,
                 unary_evidence: set[AtomicFormula] = None,
                 circle_len: int = None):
        self.domain: set[Const] = domain
        self.sentence: SC2 = sentence
        self.weights: dict[Pred, tuple[Expr, Expr]] = weights
        self.cardinality_constraint: CardinalityConstraint = cardinality_constraint
        self.unary_evidence = unary_evidence
        self.circle_len = circle_len if circle_len is not None else len(domain)
        if self.unary_evidence is not None:
            # check if the evidence is unary and consistent with the domain
            for atom in self.unary_evidence:
                if len(atom.args) != 1:
                    raise ValueError('Evidence must be unary.')
                if atom.args[0] not in self.domain:
                    raise ValueError(f'Evidence must be consistent with the domain: {atom.args[0]} not in {self.domain}.')
            for atom in self.unary_evidence:
                if ~atom in self.unary_evidence:
                    raise ValueError(f'Evidence must be consistent: {atom} and {~atom} both present.')
        # simplify expressions in weights
        for pred in self.weights:
            w_pos, w_neg = self.weights[pred]
            self.weights[pred] = (self._simplify_expr(w_pos), self._simplify_expr(w_neg))

    def _simplify_expr(self, expr):
        if isinstance(expr, Expr):
            return expr.simplify()
        elif isinstance(expr, int):
            return Rational(expr, 1)
        elif isinstance(expr, float):
            frac = Fraction(expr)
            return Rational(frac.numerator, frac.denominator)
        else:
            raise ValueError(f'Unsupported expression type: {type(expr)}')

    def contain_linear_order_axiom(self) -> bool:
        return Pred('LEQ', 2) in self.sentence.preds() or \
            self.contain_predecessor_axiom()

    def contain_predecessor_axiom(self) -> bool:
        preds = self.sentence.preds()
        return any(pred.name.startswith('PRED') for pred in preds) or \
            self.contain_circular_predecessor_axiom()

    def contain_circular_predecessor_axiom(self) -> bool:
        return Pred('CIRCULAR_PRED', 2) in self.sentence.preds()

    def contain_unary_evidence(self) -> bool:
        return self.unary_evidence is not None and len(self.unary_evidence) > 0

    def __str__(self) -> str:
        s = ''
        s += 'Domain: \n'
        s += '\t' + str(self.domain) + '\n'
        s += 'Sentence: \n'
        s += '\t' + str(self.sentence) + '\n'
        s += 'Weights: \n'
        s += '\t' + str(self.weights) + '\n'
        if self.cardinality_constraint is not None:
            s += 'Cardinality Constraint: \n'
            s += '\t' + str(self.cardinality_constraint) + '\n'
        if self.unary_evidence is not None:
            s += 'Unary Evidence: \n'
            s += '\t' + str(self.unary_evidence) + '\n'
        return s

    def __repr__(self) -> str:
        return str(self)

    def to_file(self, filename):
        domain = self.domain
        sentence = self.sentence
        uni_formula = sentence.uni_formula
        ext_formula = sentence.ext_formulas
        cardinality_constraint = self.cardinality_constraint
        unary_evidence = self.unary_evidence
        s = f"{formula_to_str(uni_formula)}\n"
        for f in ext_formula:
            s += f"& {formula_to_str(f)}\n"
        s += "\n"
        s += f"domain = {domain}\n\n"
        for pred, (weights_pos, weights_neg) in self.weights.items():
            if weights_pos == Rational(1, 1) and weights_neg == Rational(1, 1):
                continue
            s += f"{weights_pos} {weights_neg} {pred}\n"
        for expr, comp, param in cardinality_constraint.constraints:
            s += " ".join(f"{coef} |{pred.name}|" for pred, coef in expr.items())
            s += f" {comp} {param}\n"
        s += "\n"
        s += ", ".join(str(atom) for atom in unary_evidence)
        with open(filename, 'w') as f:
            f.write(s)


class MLNProblem(object):
    """
    A Markov Logic Network problem.
    """

    def __init__(self, rules: tuple[list[tuple[Rational, Rational]], list[Formula]],
                 domain: set[Const],
                 cardinality_constraint: CardinalityConstraint = None,
                 unary_evidence: set[AtomicFormula] = None,
                 circle_len: int = None):
        self.rules = rules
        # self.formulas: rules[1]
        # self.formula_weights: = dict(zip(rules[1], rules[0]))
        self.domain: set[Const] = domain
        self.cardinality_constraint: CardinalityConstraint = cardinality_constraint
        self.unary_evidence: set[AtomicFormula] = unary_evidence
        self.circle_len = circle_len if circle_len is not None else len(domain)


def MLN_to_WFOMC(mln: MLNProblem):
    sentence = top
    weightings: dict[Pred, tuple[Rational, Rational]] = dict()
    for weighting, formula in zip(*mln.rules):
        free_vars = formula.free_vars()
        if weighting != float('inf'):
            aux_pred = new_predicate(len(free_vars), AUXILIARY_PRED_NAME)
            formula = Equivalence(formula, aux_pred(*free_vars))
            weightings[aux_pred] = (Rational(Fraction(math.exp(weighting)).numerator,
                                             Fraction(math.exp(weighting)).denominator), Rational(1, 1))
        for free_var in free_vars:
            formula = QuantifiedFormula(Universal(free_var), formula)
        sentence = sentence & formula

    try:
        sentence = to_sc2(sentence)
    except:
        raise ValueError('Sentence must be a valid SC2 formula.')
    return WFOMCProblem(sentence, mln.domain, weightings,
                        mln.cardinality_constraint,
                        mln.unary_evidence,
                        mln.circle_len)
