from __future__ import annotations

import random
from itertools import accumulate
from typing import Iterable, Generator
from functools import reduce
from itertools import accumulate, repeat

from symengine import Pow, var, Expr, Symbol
from symengine import Rational as sym_Rational
from symengine.lib.symengine_wrapper import lcm
from decimal import Decimal
from bisect import bisect_left

Rational = sym_Rational
Poly = Expr


def create_vars(conf: str) -> list[Symbol]:
    return var(conf)


def expand(polynomial: Poly) -> Poly:
    return polynomial.expand()


def coeff_monomial(polynomial, monomial) -> Rational:
    coeff = polynomial.as_coefficients_dict()[monomial]
    if isinstance(coeff, int):
        coeff = Rational(coeff, 1)
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


def coeff_dict(p: Poly, gens: list[Symbol]) -> Generator[tuple[int], Rational, None]:
    for monomial, coeff in p.as_coefficients_dict().items():
        degrees = dict(_get_degrees(monomial))
        for var, degree in degrees.items():
            if (var is not None and var not in gens) and degree != 0:
                coeff *= Pow(var, degree)
        yield tuple(degrees.get(sym, 0) for sym in gens), coeff


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
