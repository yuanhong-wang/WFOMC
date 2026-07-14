"""Lark grammar for framework-native MLN problem files."""

from .cardinality import CARDINALITY_GRAMMAR
from .fol import FORMULA_GRAMMAR
from .wfomcs import DOMAIN_GRAMMAR


RULE_GRAMMAR = r"""
    rules: rule*
    rule: hard_rule | soft_rule
    hard_rule: ffl "."
    soft_rule: weighting ffl
    weighting: SIGNED_NUMBER
""" + FORMULA_GRAMMAR

MLN_GRAMMAR = r"""
    ?mln: rules domain cardinality_constraints unary_evidence
""" + RULE_GRAMMAR + DOMAIN_GRAMMAR + CARDINALITY_GRAMMAR

grammar = MLN_GRAMMAR
rule_grammar = RULE_GRAMMAR

__all__ = ["MLN_GRAMMAR", "RULE_GRAMMAR", "grammar", "rule_grammar"]
