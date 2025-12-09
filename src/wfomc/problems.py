from copy import deepcopy
from fractions import Fraction
import math

import sympy
from sympy.logic import boolalg

from wfomc.fol import SC2, to_sc2, AtomicFormula, Const, Pred, top, AUXILIARY_PRED_NAME, \
    Formula, QuantifiedFormula, Universal, Equivalence, new_predicate, QFFormula, Counting, \
    Implication, Conjunction, Disjunction, Negation, BinaryFormula, CompoundFormula
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

    def _formula_to_wfomcs_syntax(self, formula: Formula) -> str:
        """
        Convert a formula to .wfomcs syntax string.
        Handles proper formatting for operators like ->, <->, &, |, ~
        
        :param formula: The formula to convert
        :return: String representation in .wfomcs FOL syntax
        """
        if isinstance(formula, QuantifiedFormula):
            # Handle quantified formulas
            quant_str = str(formula.quantifier_scope)
            inner_str = self._formula_to_wfomcs_syntax(formula.quantified_formula)
            return f'{quant_str}: ({inner_str})'
        elif isinstance(formula, QFFormula):
            # Convert QFFormula - need to handle sympy boolean expressions
            # Parse the sympy expression and convert it properly
            return self._sympy_to_fol(formula.expr)
        elif isinstance(formula, Implication):
            left_str = self._formula_to_wfomcs_syntax(formula.left_formula)
            right_str = self._formula_to_wfomcs_syntax(formula.right_formula)
            return f'({left_str} -> {right_str})'
        elif isinstance(formula, Equivalence):
            left_str = self._formula_to_wfomcs_syntax(formula.left_formula)
            right_str = self._formula_to_wfomcs_syntax(formula.right_formula)
            return f'({left_str} <-> {right_str})'
        elif isinstance(formula, Conjunction):
            left_str = self._formula_to_wfomcs_syntax(formula.left_formula)
            right_str = self._formula_to_wfomcs_syntax(formula.right_formula)
            return f'({left_str} & {right_str})'
        elif isinstance(formula, Disjunction):
            left_str = self._formula_to_wfomcs_syntax(formula.left_formula)
            right_str = self._formula_to_wfomcs_syntax(formula.right_formula)
            return f'({left_str} | {right_str})'
        elif isinstance(formula, Negation):
            inner_str = self._formula_to_wfomcs_syntax(formula.sub_formula)
            return f'~{inner_str}'
        elif isinstance(formula, BinaryFormula):
            left_str = self._formula_to_wfomcs_syntax(formula.left_formula)
            right_str = self._formula_to_wfomcs_syntax(formula.right_formula)
            return f'({left_str} {formula.op_name} {right_str})'
        elif isinstance(formula, CompoundFormula):
            # Generic compound formula
            return str(formula)
        else:
            return str(formula)
    
    def _sympy_to_fol(self, expr: sympy.Basic) -> str:
        """
        Convert a sympy boolean expression to FOL syntax.
        
        :param expr: The sympy expression to convert
        :return: String representation in FOL syntax
        """
        if isinstance(expr, boolalg.Implies):
            left = self._sympy_to_fol(expr.args[0])
            right = self._sympy_to_fol(expr.args[1])
            return f'{left} -> {right}'
        elif isinstance(expr, boolalg.Equivalent):
            left = self._sympy_to_fol(expr.args[0])
            right = self._sympy_to_fol(expr.args[1])
            return f'{left} <-> {right}'
        elif isinstance(expr, boolalg.And):
            parts = [self._sympy_to_fol(arg) for arg in expr.args]
            return ' & '.join(f'({part})' if ' ' in part else part for part in parts)
        elif isinstance(expr, boolalg.Or):
            parts = [self._sympy_to_fol(arg) for arg in expr.args]
            return ' | '.join(f'({part})' if ' ' in part else part for part in parts)
        elif isinstance(expr, boolalg.Not):
            inner = self._sympy_to_fol(expr.args[0])
            if ' ' in inner:
                return f'~({inner})'
            else:
                return f'~{inner}'
        elif isinstance(expr, sympy.Symbol):
            # This is an atomic formula
            return str(expr)
        else:
            return str(expr)

    def _weight_to_string(self, weight) -> str:
        """
        Convert a weight value to string representation.
        
        :param weight: Weight value (Rational, Expr, or numeric)
        :return: String representation of the weight
        """
        if isinstance(weight, (Rational, Expr)):
            return str(weight)
        else:
            return str(float(weight))
    
    def export_wfomcs(self, filepath: str = None) -> str:
        """
        Export the WFOMC problem to a .wfomcs file format.
        
        :param filepath: Optional path to save the file. If not provided, returns the string.
        :return: String representation in .wfomcs format
        """
        lines = []
        
        # 1. Export the sentence (formula)
        # For SC2, we need to combine the universal and existential formulas
        if isinstance(self.sentence, SC2):
            # Combine all formulas with &
            formulas = []
            if self.sentence.uni_formula is not None:
                formulas.append(self._formula_to_wfomcs_syntax(self.sentence.uni_formula))
            for ext_formula in self.sentence.ext_formulas:
                formulas.append(self._formula_to_wfomcs_syntax(ext_formula))
            for cnt_formula in self.sentence.cnt_formulas:
                formulas.append(self._formula_to_wfomcs_syntax(cnt_formula))
            lines.append(' &\n'.join(formulas))
        else:
            lines.append(self._formula_to_wfomcs_syntax(self.sentence))
        lines.append('')
        
        # 2. Export the domain
        # Check if domain elements follow a pattern
        domain_list = sorted(self.domain, key=lambda c: c.name)
        
        # Try to determine if domain is numeric pattern
        domain_names = [c.name for c in domain_list]
        if len(domain_names) > 0:
            # Output as set notation
            lines.append('domain = {' + ', '.join(domain_names) + '}')
        
        # 3. Export weightings (if any non-default weights exist)
        if self.weights:
            for pred, (w_pos, w_neg) in sorted(self.weights.items(), key=lambda x: x[0].name):
                # Format weights - convert Rational to string
                w_pos_str = self._weight_to_string(w_pos)
                w_neg_str = self._weight_to_string(w_neg)
                lines.append(f'{w_pos_str} {w_neg_str} {pred.name}')
        
        # 4. Export cardinality constraints (if any)
        if self.cardinality_constraint is not None and not self.cardinality_constraint.empty():
            lines.append('')
            for expr, comp, param in self.cardinality_constraint.constraints:
                # Build the constraint expression
                terms = []
                for pred, coef in expr.items():
                    if coef == 1:
                        terms.append(f'|{pred.name}|')
                    else:
                        terms.append(f'{coef} |{pred.name}|')
                constraint_str = ' + '.join(terms) if len(terms) > 1 else terms[0]
                lines.append(f'{constraint_str} {comp} {param}')
        
        # 5. Export unary evidence (if any)
        if self.unary_evidence is not None and len(self.unary_evidence) > 0:
            lines.append('')
            evidence_strs = []
            for atom in sorted(self.unary_evidence, key=lambda a: (a.pred.name, a.args[0].name, not a.positive)):
                evidence_strs.append(str(atom))
            lines.append(', '.join(evidence_strs))
        
        # Join all lines
        result = '\n'.join(lines)
        
        # Save to file if filepath is provided
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(result)
        
        return result

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
