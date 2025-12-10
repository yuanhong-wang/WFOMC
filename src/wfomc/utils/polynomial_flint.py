from __future__ import annotations

from typing import TypeAlias

import sympy
from sympy import Rational, Expr, Poly
from flint import fmpq, fmpq_mpoly_ctx, fmpq_mpoly


# for efficiency, we directly use flint's multivariate polynomial for implementation
EPoly: TypeAlias = fmpq_mpoly
RingElement: TypeAlias = fmpq | fmpq_mpoly


def align_ctx(polys: list[EPoly]) -> list[EPoly]:
    """
    Align the contexts of the given polynomials.

    :param polys: The polynomials to align
    :return: A list of polynomials with aligned contexts
    """
    gens_name = set()
    for p in polys:
        gens_name.update(g for g in p.context().names())
    aligned_ctx = fmpq_mpoly_ctx.get(list(gens_name), 'lex')
    aligned_polys = list(
        p.project_to_context(aligned_ctx)
        for p in polys
    )
    return aligned_polys


def sympoly2fmpq_mpoly(p: Poly) -> EPoly:
    d = p.as_dict()
    gens = p.gens
    ctx = fmpq_mpoly_ctx.get([str(g) for g in gens], 'lex')
    fmpq_mpoly_terms = dict()
    for monomial_degrees, coeff in d.items():
        fmpq_coeff = fmpq(int(coeff.p), int(coeff.q))
        fmpq_mpoly_terms[monomial_degrees] = fmpq_coeff
    return ctx.from_dict(fmpq_mpoly_terms)


def to_ringelements(values: list[Expr]) -> list[RingElement]:
    ring_elements = list()
    for v in values:
        if isinstance(v, Rational):
            ring_elements.append(fmpq(int(v.p), int(v.q)))
        elif isinstance(v, Expr):
            p = Poly(v)
            ring_elements.append(sympoly2fmpq_mpoly(p))
        elif isinstance(v, Poly):
            ring_elements.append(sympoly2fmpq_mpoly(v))
        else:
            raise ValueError(f'Unsupported type: {type(v)}')
    polys = list()
    polys_index = list()
    for idx, re in enumerate(ring_elements):
        if isinstance(re, EPoly):
            polys.append(re)
            polys_index.append(idx)
    aligned_polys = align_ctx(polys)
    for aligned_poly, idx in zip(aligned_polys, polys_index):
        ring_elements[idx] = aligned_poly
    return ring_elements


def to_symexpr(e: RingElement) -> Expr:
    if isinstance(e, fmpq):
        return Rational(int(e.p), int(e.q))
    elif isinstance(e, fmpq_mpoly):
        sym_gens = [sympy.symbols(g) for g in e.context().names()]
        return Poly.from_dict(
            e.to_dict(),
            gens=sym_gens
        )
    else:
        raise ValueError(f'Unsupported type: {type(e)}')


def filter_poly(p: EPoly, corr_vars: list[Expr],
                filter_func: callable[[list[int]], bool]) -> EPoly:
    gens = p.context().names()
    indices = [gens.index(str(var)) for var in corr_vars]
    filtered = dict()
    for degrees, coeff in p.to_dict().items():
        selected_degrees = [degrees[i] for i in indices]
        if filter_func(selected_degrees):
            filtered[degrees] = coeff
    filtered_poly = p.context().from_dict(filtered)
    ret = filtered_poly.subs(
        {str(v): 1 for v in corr_vars}
    ).project_to_context(
        fmpq_mpoly_ctx.get(
            [g for g in gens if g not in {str(v) for v in corr_vars}],
            'lex'
        )
    )
    if ret.is_constant():
        ret = ret.leading_coefficient()
    return ret
