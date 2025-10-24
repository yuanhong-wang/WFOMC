from __future__ import annotations

import random
from itertools import accumulate, repeat
from typing import Iterable, Generator, TypeAlias
from functools import reduce
from collections import defaultdict

from flint import fmpq, fmpq_mpoly_ctx, fmpq_mpoly
from decimal import Decimal
from bisect import bisect_left


Rational: TypeAlias = fmpq
Poly: TypeAlias = fmpq_mpoly
RingElement: TypeAlias = Rational | Poly


def create_vars(var_name: str, count: int = 1) -> list[Poly]:
    """ Create polynomial variables.

    :param var_name: The base name for the variables
    :param count: The number of variables to create
    :return: A list of polynomial variables
    """
    ctx = fmpq_mpoly_ctx.get((var_name, count), 'lex')
    gens = ctx.gens()
    return gens if count > 1 else [gens[0]]


def expand(polynomial: Poly) -> Poly:
    return polynomial


def coeff_monomial(polynomial, monomial) -> Rational:
    for degrees, coeff in polynomial.terms():
        if degrees == monomial:
            return coeff


def round_rational(n: Rational) -> Decimal:
    n = Decimal(int(n.p)) / Decimal(int(n.q))
    return n


def _get_degrees(monomial: Poly):
    if monomial.is_Number:
        return ((None, 0), )
    if monomial.is_Symbol:
        return ((monomial, 1), )
    if monomial.is_Pow:
        return ((monomial.args[0], monomial.args[1]), )
    if monomial.is_Mul:
        return sum(
            (_get_degrees(arg) for arg in monomial.args),
            start=()
        )


def coeff_dict(p: Poly, gens: list[Poly]) -> Generator[tuple[int], Rational, None]:
    p_gens = p.context().gens()
    coeffs = defaultdict(lambda: Rational(0, 1))
    gens_map = {i: p_gens.index(g) for i, g in enumerate(gens)}
    for monomial, coeff in p.terms():
        degrees = tuple(monomial[gens_map[i]] for i in range(len(gens)))
        coeffs[degrees] += coeff
    for degrees, coeff in coeffs.items():
        yield degrees, coeff


def _choices_int_weights(population: Iterable, weights: Iterable[int], k=1):
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

# from gmpy2 import mpq
# from sympy import Poly, symbols
# Rational = mpq
# Poly = Poly
#
#
# def create_vars(conf):
#     return symbols(conf)
#
#
# def expand(polynomial):
#     return Poly(polynomial)
#
#
# def coeff_monomial(polynomial, monomial) -> Rational:
#     return polynomial.coeff_monomial(monomial)
