
cc_grammar = r"""
    cardinality_constraints: cardinality_constraint*
    cardinality_constraint: cc_expr comparator NUMBER
    ?cc_expr: left_parenthesis cc_expr right_parenthesis -> parenthesis
        | cc_atom -> cc_atomic_expr
        | cc_expr "+" cc_atom -> cc_add
        | cc_expr "-" cc_atom -> cc_sub
    cc_atom: "|" predicate "|"
        | NUMBER "|" predicate "|"
"""
