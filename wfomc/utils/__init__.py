import numpy as np

from .multinomial import MultinomialCoefficients, multinomial, multinomial_less_than
from .polynomial import Rational, expand, coeff_monomial, create_vars, coeff_dict, round_rational
from .third_typing import RingElement

def format_np_complex(num: np.ndarray) -> str:
    return '{num.real:+0.04f}+{num.imag:+0.04f}j'.format(num=num)

__all__ = [
    "MultinomialCoefficients",
    "multinomial",
    "multinomial_less_than",
    "TreeSumContext",
    'RingElement',
    'Rational',
    'round_rational',
    'expand',
    'coeff_monomial',
    'coeff_dict',
    'create_vars',
    "tree_sum",
    "format_np_complex"
]
