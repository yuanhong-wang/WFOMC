"""Lark grammar for framework-native WFOMCS problem files."""

from .cardinality import CARDINALITY_GRAMMAR
from .fol import FORMULA_GRAMMAR


DOMAIN_GRAMMAR = r"""
    domain: domain_name "=" domain_spec
    domain_name: CNAME
    ?domain_spec: INT               -> int_domain
        | ("{" domain_elements "}") -> set_domain
    domain_elements: element ("," element)*
    element: CNAME
"""

WFOMCS_GRAMMAR = r"""
    ?wfomcs: ffl domain weightings cardinality_constraints unary_evidence
    weightings: weighting*
    weighting: weight weight predicate

    weight: SIGNED_FLOAT | SIGNED_INT
""" + DOMAIN_GRAMMAR + CARDINALITY_GRAMMAR + FORMULA_GRAMMAR

domain_grammar = DOMAIN_GRAMMAR
grammar = WFOMCS_GRAMMAR

__all__ = ["DOMAIN_GRAMMAR", "WFOMCS_GRAMMAR", "domain_grammar", "grammar"]
