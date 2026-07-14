"""Lark grammar for function-free first-order formulas."""

FORMULA_GRAMMAR = r"""
    ?ffl: atomic_ffl | compound_ffl | exactlyone
    atomic_ffl: predicate left_parenthesis [terms] right_parenthesis
        | predicate -> nullary_atomic
    exactlyone: "ExactlyOne" left_square_bracket predicates right_square_bracket
    predicates: predicate ("," predicate)*
    terms: term ("," term)*
    negation: not ffl
    conjunction: ffl and ffl
    disjunction: ffl or ffl
    implication: ffl implies ffl
    equivalence: ffl iff ffl
    ?compound_ffl: left_parenthesis ffl right_parenthesis -> parenthesis
       | quantifier_variable ":" quantifier_body -> quantification
       | equivalence
       | implication
       | disjunction
       | conjunction
       | negation
    ?quantifier_body: left_parenthesis ffl right_parenthesis -> parenthesized_body
       | atomic_ffl
    ?term: constant
        | variable

    unary_evidence: unary_literal ("," unary_literal)*
        |
    ?unary_literal: atomic_ffl | negation

    left_square_bracket: "["
    right_square_bracket: "]"
    left_parenthesis: "("
    right_parenthesis: ")"
    quantifier_variable: quantifier variable
    ?quantifier: universal_quantifier | existential_quantifier | counting_quantifier
    universal_quantifier: "\\forall"
    existential_quantifier: "\\exists"
    counting_quantifier: "\\exists" "_" counting_spec
    ?counting_spec: "{" counting_body "}"
        | counting_body
    ?counting_body: comparator count_parameter -> bounded_count
        | count_parameter "mod" count_parameter -> mod_count
    constant: LCASE_CNAME
    variable: UCASE_LETTER
    predicate: CNAME
    not: "~"
    and: "&"
    or: "|"
    implies: "->"
    iff: "<->"
    count_parameter: INT
    ?comparator: equality | le | ge | lt | gt | nequality
    equality: "="
    nequality: "!="
    le: "<="
    ge: ">="
    lt: "<"
    gt: ">"
    LCASE_CNAME: LCASE_LETTER ("_"|LCASE_LETTER|UCASE_LETTER|DIGIT)*

    %import common.LCASE_LETTER
    %import common.UCASE_LETTER
    %import common.CNAME
    %import common.DIGIT
    %import common.FLOAT
    %import common.INT
    %import common.SIGNED_FLOAT
    %import common.SIGNED_INT
    %import common.SIGNED_NUMBER
    %import common.NUMBER
    %import common.WS
    %import common.SH_COMMENT
    %ignore WS
    %ignore SH_COMMENT
"""

function_free_logic_grammar = FORMULA_GRAMMAR

__all__ = ["FORMULA_GRAMMAR", "function_free_logic_grammar"]
