from __future__ import annotations
from functools import reduce
from collections import defaultdict
from .syntax import *

PREDICATES = defaultdict(list)

PERMITTED_VAR_NAMES = range(ord('A'), ord('Z') + 1)


def new_var(exclude: frozenset[Var]) -> Var:
    for c in PERMITTED_VAR_NAMES:
        v = Var(chr(c))
        if v not in exclude:
            return v
    raise RuntimeError(
        "No more variables available"
    )


def new_predicate(arity: int, pred_name: str, used_pred_names: set[str] = None) -> Pred:
    """Creates a new predicate with a unique name."""
    global PREDICATES
    # If used_pred_names is provided, use a simple indexed naming scheme
    if used_pred_names is not None:
        i = 0
        name = f'{pred_name}{i}'
        while name in used_pred_names:
            i += 1
            name = f'{pred_name}{i}'
        return Pred(name, arity)

    # Otherwise, use the global PREDICATES dictionary for uniqueness
    name = f'{pred_name}{len(PREDICATES[pred_name])}'
    p = Pred(name, arity)
    PREDICATES[pred_name].append(p)
    return p


def get_predicates(name: str) -> list[Pred]:
    return PREDICATES[name]


def new_scott_predicate(arity: int) -> Pred:
    return new_predicate(arity, SCOTT_PREDICATE_PREFIX)


def pad_vars(vars: frozenset[Var], arity: int) -> frozenset[Var]:
    if arity > 3:
        raise RuntimeError(
            "Not support arity > 3"
        )
    ret_vars = set(vars)
    default_vars = [X, Y, Z]
    idx = 0
    while (len(ret_vars) < arity):
        ret_vars.add(default_vars[idx])
        idx += 1
    return frozenset(list(ret_vars)[:arity])


def exactly_one_qf(preds: list[Pred]) -> QFFormula:
    if len(preds) == 1:
        return top
    lits = [p(X) for p in preds]
    # p1(x) v p2(x) v ... v pm(x)
    formula = reduce(lambda x, y: x | y, lits) & \
        exclusive_qf(preds)
    return formula


def exactly_one(preds: list[Pred]) -> QuantifiedFormula:
    if len(preds) == 1:
        return top
    return QuantifiedFormula(Universal(X), exactly_one_qf(preds))


def exclusive_qf(preds: list[Pred]) -> QFFormula:
    if len(preds) == 1:
        return top
    lits = [p(X) for p in preds]
    formula = top
    for i, l1 in enumerate(lits):
        for j, l2 in enumerate(lits):
            if i < j:
                formula = formula & ~(l1 & l2)
    return formula


def exclusive(preds: list[Pred]) -> QuantifiedFormula:
    if len(preds) == 1:
        return top
    return QuantifiedFormula(Universal(X), exclusive_qf(preds))


def tseitin_transform(sentence):
    """Transform an SC2 sentence so every existential/counting quantified
    subformula has a single predicate atom as its innermost body.

    For each ext/cnt formula whose innermost quantifier-free body is not a
    single atom, introduce a fresh Tseitin predicate A(vars) and add
    the equivalence (A(vars) <-> body) to the universal part. The existential/
    counting formula is rewritten to use A(vars) directly.
    """
    # Import here to avoid a circular import (sc2.py imports from utils.py).
    from .sc2 import SC2

    canonical_order = [X, Y, Z]

    uni_body = sentence.uni_formula
    outer_quantifiers: list[Universal] = []
    while isinstance(uni_body, QuantifiedFormula):
        outer_quantifiers.append(uni_body.quantifier_scope)
        uni_body = uni_body.quantified_formula

    bound_vars: set[Var] = {q.quantified_var for q in outer_quantifiers}
    extra_equiv: QFFormula = top

    def transform_one(qformula: QuantifiedFormula) -> QuantifiedFormula:
        nonlocal extra_equiv
        scopes: list = []
        inner = qformula
        while isinstance(inner, QuantifiedFormula):
            scopes.append(inner.quantifier_scope)
            inner = inner.quantified_formula
        # inner is a QFFormula. Always introduce a fresh Tseitin predicate
        # (even if inner is already atomic) — WFOMC algorithms expect a
        # dedicated auxiliary predicate for each ext/cnt subformula. The
        # arity matches the quantifier-chain depth so that the pred has one
        # argument per quantifier in canonical order.
        chain_vars = {scope.quantified_var for scope in scopes}
        args = tuple(v for v in canonical_order if v in chain_vars)
        aux_pred = new_predicate(len(args), TSEITIN_PRED_NAME)
        aux_atom = aux_pred(*args)
        extra_equiv = extra_equiv & inner.equivalent(aux_atom)
        for v in args:
            bound_vars.add(v)
        rebuilt: Formula = aux_atom
        for scope in reversed(scopes):
            rebuilt = QuantifiedFormula(scope, rebuilt)
        return rebuilt

    new_ext_formulas = [transform_one(f) for f in sentence.ext_formulas]
    new_cnt_formulas = [transform_one(f) for f in sentence.cnt_formulas]

    new_uni_body = uni_body & extra_equiv
    # Ensure all variables referenced in the merged body are universally bound.
    existing_vars = {q.quantified_var for q in outer_quantifiers}
    final_quantifiers = list(outer_quantifiers)
    for v in canonical_order:
        if v in bound_vars and v not in existing_vars:
            final_quantifiers.append(Universal(v))
            existing_vars.add(v)

    new_uni_formula: Formula = new_uni_body
    for scope in reversed(final_quantifiers):
        new_uni_formula = QuantifiedFormula(scope, new_uni_formula)

    return SC2(
        uni_formula=new_uni_formula,
        ext_formulas=new_ext_formulas,
        cnt_formulas=new_cnt_formulas,
    )
