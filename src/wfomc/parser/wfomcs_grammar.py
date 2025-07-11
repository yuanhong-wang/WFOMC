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


grammar = r"""
    ?wfomcs: ffl domain weightings cardinality_constraints
    weightings: weighting*
    weighting: weight weight predicate

    weight: FLOAT | INT
""" + domain_grammar + cc_grammar + function_free_logic_grammar
