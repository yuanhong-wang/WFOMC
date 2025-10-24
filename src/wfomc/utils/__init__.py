import numpy as np

from .multinomial import MultinomialCoefficients, multinomial, multinomial_less_than
# from .polynomial import Rational, Poly, RingElement, expand, coeff_monomial, create_vars, coeff_dict, round_rational
from .polynomial_flint import Rational, Poly, RingElement, expand, coeff_monomial, create_vars, coeff_dict, round_rational

def format_np_complex(num: np.ndarray) -> str:
    return '{num.real:+0.04f}+{num.imag:+0.04f}j'.format(num=num)

__all__ = [
    "MultinomialCoefficients",
    "multinomial",
    "multinomial_less_than",
    "TreeSumContext",
    'Poly',
    'Rational',
    'RingElement',
    'round_rational',
    'expand',
    'coeff_monomial',
    'coeff_dict',
    'create_vars',
    "tree_sum",
    "format_np_complex",
]
