from __future__ import annotations

import random
from itertools import accumulate, repeat
import typing
from typing import Iterable, Generator, Union, TypeAlias
from functools import reduce
from collections import defaultdict
from math import lcm

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


def expand(polynomial: RingElement) -> RingElement:
    return polynomial


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


def coeff_dict(p: EPoly, gens: list[Expr]) -> Generator[tuple[tuple[int, ...], Rational], None, None]:
    p_gens = p.context().names()
    coeffs = defaultdict(lambda: Rational(0, 1))
    gens_map = {i: list(p_gens).index(str(g)) for i, g in enumerate(gens)}
    for monomial, coeff in p.terms():
        degrees = tuple(monomial[gens_map[i]] for i in range(len(gens)))
        coeffs[degrees] += Rational(int(coeff.p), int(coeff.q))
    for degrees, coeff in coeffs.items():
        yield degrees, coeff


def filter_poly(p: EPoly, corr_vars: list[Expr],
                filter_func: callable([[list[int]], bool])) -> EPoly:
    gens = p.context().names()
    indices = [list(gens).index(str(var)) for var in corr_vars]
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


def _choices_int_weights(population: Iterable, weights: Iterable[int], k=1) -> list:
    from bisect import bisect_left
    n = len(population)
    cum_weights = list(accumulate(weights))
    total = cum_weights[-1]
    hi = n - 1
    return [population[bisect_left(cum_weights, random.randint(1, total), 0, hi)]
            for _ in repeat(None, k)]


def choices(population: Iterable, weights: Iterable[Rational], k=1) -> list:
    """
    Return a k sized list of population elements chosen with replacement.

    Adapted from random.choices, but with rational weights.
    """
    lcm_val = reduce(lambda a, b: lcm(a, b), [w.q for w in weights])
    weights = [
        w.p * lcm_val // w.q for w in weights
    ]
    return _choices_int_weights(population, weights, k)


def bernoulli_trial(p: Rational) -> bool:
    """
    Perform a Bernoulli trial with success probability p.
    """
    denominator, numerator = p.q, p.p
    rand_int = random.randint(1, denominator)
    return rand_int <= numerator
