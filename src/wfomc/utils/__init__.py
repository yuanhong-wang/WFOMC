import numpy as np

from wfomc._compat import try_import_cython as _try_cy

_multinomial_mod = _try_cy("wfomc.utils.multinomial", "wfomc.utils.multinomial")
MultinomialCoefficients = _multinomial_mod.MultinomialCoefficients
multinomial = _multinomial_mod.multinomial
multinomial_less_than = _multinomial_mod.multinomial_less_than

# from .polynomial import Rational, Poly, RingElement, expand, coeff_monomial, create_vars, coeff_dict, round_rational
from .polynomial_flint import Rational, Poly, RingElement, expand, coeff_monomial, create_vars, coeff_dict, round_rational


def format_np_complex(num: np.ndarray) -> str:
    return '{num.real:+0.04f}+{num.imag:+0.04f}j'.format(num=num)


__all__ = [
    "MultinomialCoefficients",
    "multinomial",
    "multinomial_less_than",
    "Poly",
    "Rational",
    "RingElement",
    "round_rational",
    "expand",
    "coeff_monomial",
    "coeff_dict",
    "create_vars",
    "format_np_complex",
]
