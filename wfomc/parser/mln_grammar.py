from .fol_grammar import function_free_logic_grammar
from .cardinality_constraints_grammar import cc_grammar

domain_grammar = r"""
    domain: domain_name "=" domain_spec
    domain_name: CNAME
    ?domain_spec: INT               -> int_domain
        | ("{" domain_elements "}") -> set_domain
    domain_elements: element ("," element)*
    element: CNAME
"""

rule_grammar = r"""
    rules: rule*
    rule: hard_rule | soft_rule
    hard_rule: ffl "."
    soft_rule: weighting ffl
    weighting: SIGNED_NUMBER
""" + function_free_logic_grammar

grammar = r"""
    ?mln: rules domain cardinality_constraints
""" + rule_grammar + domain_grammar + cc_grammar
