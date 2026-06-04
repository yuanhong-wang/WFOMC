from __future__ import annotations

from collections import defaultdict
from typing import Generator, Iterable

from flint import fmpq, fmpq_mpoly
from sympy import Expr, Rational

from wfomc.utils.polynomial_flint import RingElement


class WFOMCResult:
    """Public result wrapper for :func:`wfomc.wfomc`.

    The solver internally uses FLINT values for exact arithmetic.  This wrapper
    keeps those implementation details out of the public API while still
    exposing the operations callers need to inspect constants and polynomial
    coefficients.
    """

    def __init__(self, value: RingElement) -> None:
        self._value = value

    @property
    def raw(self) -> RingElement:
        """Return the internal FLINT value.

        Prefer the typed helpers on this class in application code.  This is
        mainly here for debugging and gradual migration of advanced callers.
        """

        return self._value

    def is_zero(self) -> bool:
        return self._value == 0

    def is_polynomial(self) -> bool:
        return isinstance(self._value, fmpq_mpoly)

    def is_constant(self) -> bool:
        if isinstance(self._value, fmpq):
            return True
        if isinstance(self._value, fmpq_mpoly):
            return self._value.is_constant()
        return False

    def constant_value(self) -> Rational | None:
        """Return the result as a SymPy rational when it is constant."""

        if isinstance(self._value, fmpq):
            return Rational(int(self._value.p), int(self._value.q))
        if isinstance(self._value, fmpq_mpoly) and self._value.is_constant():
            coeff = self._value.leading_coefficient()
            return Rational(int(coeff.p), int(coeff.q))
        return None

    def variable_names(self) -> tuple[str, ...]:
        if isinstance(self._value, fmpq_mpoly):
            return tuple(self._value.context().names())
        return ()

    def terms(
        self,
        variables: Iterable[Expr | str] | None = None,
    ) -> Generator[tuple[tuple[int, ...], Rational], None, None]:
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
        if isinstance(self._value, fmpq):
            yield (0,) * len(variable_names), Rational(int(self._value.p), int(self._value.q))
            return

        if not isinstance(self._value, fmpq_mpoly):
            raise TypeError(f"Unsupported result type: {type(self._value)}")

        polynomial_names = tuple(self._value.context().names())
        if variables is None:
            for degrees, coeff in self._value.terms():
                yield degrees, Rational(int(coeff.p), int(coeff.q))
            return

        missing = [name for name in variable_names if name not in polynomial_names]
        if missing:
            raise ValueError(f"Variables not present in result: {missing}")

        indices = [polynomial_names.index(name) for name in variable_names]
        coeffs = defaultdict(lambda: Rational(0, 1))
        for degrees, coeff in self._value.terms():
            projected = tuple(degrees[i] for i in indices)
            coeffs[projected] += Rational(int(coeff.p), int(coeff.q))
        yield from coeffs.items()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WFOMCResult):
            other = other.raw
        return self._value == other

    def __int__(self) -> int:
        value = self.constant_value()
        if value is None:
            raise TypeError("Cannot convert a non-constant WFOMCResult to int")
        return int(value)

    def __truediv__(self, other: object) -> object:
        return self._value / other

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"WFOMCResult({self._value!r})"
