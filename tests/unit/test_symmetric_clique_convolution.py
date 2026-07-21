from __future__ import annotations

from math import comb

from wfomc.algo.symmetric_clique import TwistedBinomialConvolver
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext


def _direct(left, right, interaction, arithmetic):
    result = []
    for total in range(len(left)):
        value = arithmetic.zero()
        for selected in range(total + 1):
            factor = arithmetic.from_int(comb(total, selected))
            factor = arithmetic.multiply(factor, left[selected])
            factor = arithmetic.multiply(factor, right[total - selected])
            factor = arithmetic.multiply(
                factor,
                arithmetic.power(interaction, selected * (total - selected)),
            )
            value = arithmetic.add(value, factor)
        result.append(value)
    return tuple(result)


def test_twisted_convolution_matches_direct_coefficients() -> None:
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    left = tuple(arithmetic.from_int(value) for value in (1, 2, 3, 4, 5))
    right = tuple(arithmetic.from_int(value) for value in (1, 3, 5, 7, 9))
    interaction = arithmetic.from_int(2)
    convolver = TwistedBinomialConvolver(4, interaction, arithmetic)

    assert convolver.combine(left, right) == _direct(
        left, right, interaction, arithmetic
    )


def test_twisted_convolution_is_associative() -> None:
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    rows = [
        tuple(arithmetic.from_int(value) for value in values)
        for values in (
            (1, 2, 1, 3, 2),
            (1, 1, 4, 1, 5),
            (1, 3, 2, 2, 1),
        )
    ]
    convolver = TwistedBinomialConvolver(4, arithmetic.from_int(3), arithmetic)

    assert convolver.combine(convolver.combine(rows[0], rows[1]), rows[2]) == (
        convolver.combine(rows[0], convolver.combine(rows[1], rows[2]))
    )


class _CountingConvolver(TwistedBinomialConvolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combine_calls = 0

    def combine(self, left, right):
        self.combine_calls += 1
        return super().combine(left, right)


def test_power_uses_binary_exponentiation() -> None:
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    base = tuple(arithmetic.from_int(value) for value in (1, 2, 3, 4, 5, 6))
    interaction = arithmetic.from_int(2)
    expected_convolver = TwistedBinomialConvolver(5, interaction, arithmetic)
    expected = base
    for _ in range(7):
        expected = expected_convolver.combine(expected, base)

    convolver = _CountingConvolver(5, interaction, arithmetic)

    assert convolver.power(base, 8) == expected
    assert convolver.combine_calls == 3


def test_product_groups_equal_rows_before_powering() -> None:
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    first = tuple(arithmetic.from_int(value) for value in (1, 2, 3, 4, 5))
    second = tuple(arithmetic.from_int(value) for value in (1, 1, 2, 3, 5))
    rows = [first] * 8 + [second] * 2
    interaction = arithmetic.from_int(2)
    expected_convolver = TwistedBinomialConvolver(4, interaction, arithmetic)
    expected = rows[0]
    for row in rows[1:]:
        expected = expected_convolver.combine(expected, row)

    convolver = _CountingConvolver(4, interaction, arithmetic)

    assert convolver.product(rows) == expected
    assert convolver.combine_calls == 5


def test_symbolic_convolution_obeys_arithmetic_context() -> None:
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_POLY,
        symbolic_variables=("x",),
        degree_limits=(("x", 6),),
    )
    symbol = arithmetic.symbol("x")
    base = tuple(
        arithmetic.power(symbol, degree) for degree in range(5)
    )
    convolver = TwistedBinomialConvolver(4, symbol, arithmetic)

    assert convolver.power(base, 3) == convolver.combine(
        convolver.combine(base, base), base
    )
