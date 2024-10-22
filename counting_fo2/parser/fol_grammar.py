function_free_logic_grammar = r"""
    ?ffl: atomic_ffl | compound_ffl | exactlyone
    atomic_ffl: predicate [left_parenthesis terms right_parenthesis]
    exactlyone: "ExactlyOne" left_square_bracket predicates right_square_bracket
    predicates: predicate ("," predicate)*
    terms: term* ("," term)*
    negation: not ffl
    conjunction: ffl and ffl
    disjunction: ffl or ffl
    implication: ffl implies ffl
    equivalence: ffl iff ffl
    ?compound_ffl: left_parenthesis ffl right_parenthesis -> parenthesis
       | quantifier_variable ":" left_parenthesis ffl right_parenthesis -> quantification
       | equivalence
       | implication
       | disjunction
       | conjunction
       | negation
    ?term: constant
        | variable

    left_square_bracket: "["
    right_square_bracket: "]"
    left_parenthesis: "("
    right_parenthesis: ")"
    quantifier_variable: quantifier variable
    ?quantifier: universal_quantifier | existential_quantifier | counting_quantifier
    universal_quantifier: "\\forall"
    existential_quantifier: "\\exists"
    counting_quantifier: "\\exists_{" comparator count_parameter "}"
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
    LCASE_CNAME: LCASE_LETTER ("_"|LCASE_LETTER|DIGIT)*

    %import common.LCASE_LETTER
    %import common.UCASE_LETTER
    %import common.CNAME
    %import common.DIGIT
    %import common.FLOAT
    %import common.INT
    %import common.SIGNED_NUMBER
    %import common.NUMBER
    %import common.WS
    %import common.SH_COMMENT
    %ignore WS
    %ignore SH_COMMENT
"""
