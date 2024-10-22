from typing import TypeVar
from .polynomial import Rational, Poly


RingElement = TypeVar('RingElement', Poly, Rational)
