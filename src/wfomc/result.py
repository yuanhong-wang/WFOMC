"""Public inspection wrapper for exact and rounded WFOMC results."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Generator, Iterable

from flint import arb, arb_poly, fmpq, fmpq_mpoly, fmpq_poly


class WFOMCResult:
    """Public result wrapper for :func:`wfomc.solve`.

    The solver internally uses branch-selected exact or rounded values. This
    wrapper keeps those implementation details out of ordinary application
    code while exposing constants and polynomial coefficients.
    """

    def __init__(
        self,
        value: object,
        variable_names: tuple[str, ...] = (),
    ) -> None:
        self._value = value
        self._variable_names = variable_names

    @property
    def raw(self) -> object:
        """Return the internal FLINT value.

        Prefer the typed helpers on this class in application code.  This is
        mainly here for debugging and gradual migration of advanced callers.
        """

        return self._value

    def is_zero(self) -> bool:
        return self._value == 0

    def is_polynomial(self) -> bool:
        return isinstance(self._value, (fmpq_mpoly, fmpq_poly, arb_poly))

    def is_constant(self) -> bool:
        if isinstance(self._value, (int, float, fmpq, arb)):
            return True
        if isinstance(self._value, fmpq_mpoly):
            return self._value.is_constant()
        if isinstance(self._value, (fmpq_poly, arb_poly)):
            return self._value.degree() <= 0
        return False

    def constant_value(self) -> Fraction | float | None:
        """Return one exact rational or rounded scalar when constant."""

        if isinstance(self._value, (int, float)):
            return (
                float(self._value)
                if isinstance(self._value, float)
                else Fraction(self._value, 1)
            )
        if isinstance(self._value, fmpq):
            return Fraction(int(self._value.p), int(self._value.q))
        if isinstance(self._value, arb):
            return float(self._value)
        if isinstance(self._value, fmpq_mpoly) and self._value.is_constant():
            coeff = self._value.leading_coefficient()
            return Fraction(int(coeff.p), int(coeff.q))
        if isinstance(self._value, fmpq_poly) and self._value.degree() <= 0:
            coeff = self._value[0]
            return Fraction(int(coeff.p), int(coeff.q))
        if isinstance(self._value, arb_poly) and self._value.degree() <= 0:
            return float(self._value[0])
        return None

    def variable_names(self) -> tuple[str, ...]:
        if isinstance(self._value, fmpq_mpoly):
            return tuple(self._value.context().names())
        if isinstance(self._value, (fmpq_poly, arb_poly)):
            return self._variable_names or ("x",)
        return self._variable_names

    def terms(
        self,
        variables: Iterable[object] | None = None,
    ) -> Generator[tuple[tuple[int, ...], Fraction | float], None, None]:
        """Yield ``(degrees, coefficient)`` terms.

        If ``variables`` is provided, degrees are projected into that variable
        order and terms that only differ outside that projection are summed.
        Constant results yield one all-zero term.
        """

        variable_names = (
            tuple(str(v) for v in variables)
            if variables is not None
            else self.variable_names()
        )
        if isinstance(self._value, (int, float, fmpq, arb)):
            value = self.constant_value()
            assert value is not None
            yield (0,) * len(variable_names), value
            return

        if isinstance(self._value, (fmpq_poly, arb_poly)):
            polynomial_names = self.variable_names()
            missing = [name for name in variable_names if name not in polynomial_names]
            if missing:
                raise ValueError(f"Variables not present in result: {missing}")
            coefficients = tuple(self._value.coeffs())
            if variables is not None and not variable_names:
                total = sum(coefficients, self._value[0] * 0)
                yield (
                    (),
                    (
                        float(total)
                        if isinstance(self._value, arb_poly)
                        else Fraction(int(total.p), int(total.q))
                    ),
                )
                return
            for degree, coefficient in enumerate(coefficients):
                if coefficient == 0:
                    continue
                yield (
                    (degree,),
                    (
                        float(coefficient)
                        if isinstance(self._value, arb_poly)
                        else Fraction(int(coefficient.p), int(coefficient.q))
                    ),
                )
            return

        if not isinstance(self._value, fmpq_mpoly):
            raise TypeError(f"Unsupported result type: {type(self._value)}")

        polynomial_names = tuple(self._value.context().names())
        if variables is None:
            for degrees, coeff in self._value.terms():
                yield degrees, Fraction(int(coeff.p), int(coeff.q))
            return

        missing = [name for name in variable_names if name not in polynomial_names]
        if missing:
            raise ValueError(f"Variables not present in result: {missing}")

        indices = [polynomial_names.index(name) for name in variable_names]
        coeffs = defaultdict(lambda: Fraction(0, 1))
        for degrees, coeff in self._value.terms():
            projected = tuple(degrees[i] for i in indices)
            coeffs[projected] += Fraction(int(coeff.p), int(coeff.q))
        yield from coeffs.items()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WFOMCResult):
            other = other.raw
        if isinstance(other, Fraction):
            value = self.constant_value()
            return value == other
        return self._value == other

    def __int__(self) -> int:
        value = self.constant_value()
        if value is None:
            raise TypeError("Cannot convert a non-constant WFOMCResult to int")
        return int(value)

    def __float__(self) -> float:
        value = self.constant_value()
        if value is None:
            raise TypeError("Cannot convert a non-constant WFOMCResult to float")
        return float(value)

    def __truediv__(self, other: object) -> object:
        return self._value / other

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"WFOMCResult({self._value!r})"


__all__ = ["WFOMCResult"]
