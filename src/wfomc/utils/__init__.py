from collections import defaultdict
from typing import Generator
import numpy as np

import sympy
from sympy import Rational, Expr, Poly
from decimal import Decimal

from .multinomial import MultinomialCoefficients, multinomial, multinomial_less_than
from .polynomial_flint import RingElement, to_ringelements, to_symexpr, EPoly, filter_poly


def format_np_complex(num: np.ndarray) -> str:
    return '{num.real:+0.04f}+{num.imag:+0.04f}j'.format(num=num)


def create_vars(var_name: str, count: int = 1) -> list[Expr]:
    """ Create sympy symbolic variables.

    :param var_name: The base name of the variable(s)
    :param count: The number of variables to create
    :return: A list of sympy symbolic variables
    """
    sym_vars = list(sympy.symbols(f'{var_name}0:{count}'))
    return sym_vars


def round_rational(n: Rational) -> Decimal:
    n = Decimal(int(n.p)) / Decimal(int(n.q))
    return n


__all__ = [
    "MultinomialCoefficients",
    "multinomial",
    "multinomial_less_than",
    'Poly',
    'Expr',
    'Rational',
    'RingElement',
    'EPoly',
    'to_ringelements',
    'to_symexpr',
    'round_rational',
    'coeff_dict',
    'create_vars',
    "format_np_complex",
    "filter_poly",
]
