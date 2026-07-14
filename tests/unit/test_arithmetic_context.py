"""Tests for :class:`wfomc.arithmetic.ArithmeticContext` (Task 2).

The context is the sole channel through which WFOMC mints numeric values.
These tests pin its backend-specific constructors and the early-failure
guarantee: a multivariate rounded-arb problem (for which ``ARB_MPOLY`` does
not exist) is rejected at planning time rather than silently falling back to
``fmpq_mpoly``.
"""

from __future__ import annotations

import pytest
from flint import arb, fmpq, fmpq_mpoly_ctx, fmpz

from wfomc.arithmetic import (
    ArithmeticBackend,
    ArithmeticContext,
    choose_arithmetic_backend,
)
from wfomc.errors import ArithmeticBackendError
from wfomc.weights import WeightOptions


def _ctx(
    backend: ArithmeticBackend, symbols: tuple[str, ...] = ()
) -> ArithmeticContext:
    return ArithmeticContext(backend, symbols)


def test_fmpq_context_creates_fmpq_zero_one_fraction():
    ctx = _ctx(ArithmeticBackend.FMPQ)

    zero = ctx.zero()
    one = ctx.one()
    half = ctx.from_fraction(3, 2)

    assert isinstance(zero, fmpq)
    assert isinstance(one, fmpq)
    assert isinstance(half, fmpq)
    assert zero == fmpq(0)
    assert one == fmpq(1)
    assert half == fmpq(3, 2)
    assert ctx.neg_one() == fmpq(-1)


def test_float_context_creates_float_zero_one_fraction():
    ctx = _ctx(ArithmeticBackend.FLOAT)

    zero = ctx.zero()
    one = ctx.one()
    half = ctx.from_fraction(3, 2)

    assert isinstance(zero, float)
    assert isinstance(one, float)
    assert isinstance(half, float)
    assert zero == 0.0
    assert one == 1.0
    assert half == 1.5


def test_arb_context_creates_arb_values():
    ctx = _ctx(ArithmeticBackend.ARB)

    zero = ctx.zero()
    one = ctx.one()
    half = ctx.from_fraction(3, 2)

    assert isinstance(zero, arb)
    assert isinstance(one, arb)
    assert isinstance(half, arb)
    assert zero.is_zero()
    assert one == arb(1)
    assert half == arb(3) / arb(2)


def test_arb_context_coerces_float_to_arb():
    ctx = _ctx(ArithmeticBackend.ARB)
    value = ctx.coerce(1.5)

    assert isinstance(value, arb)
    assert value == arb(1.5)


def test_fmpz_context_rejects_non_integer_fraction():
    ctx = _ctx(ArithmeticBackend.FMPZ)

    five = ctx.from_int(5)
    assert isinstance(five, fmpz)
    assert five == fmpz(5)

    # Integer fractions are fine.
    assert ctx.from_fraction(4, 1) == fmpz(4)

    # Non-unit denominations cannot be represented as exact integers.
    with pytest.raises(ValueError, match="(?i)fmpz"):
        ctx.from_fraction(3, 2)


def test_unsupported_arb_mpoly_fails_early():
    context = fmpq_mpoly_ctx.get(("x0", "x1"), "lex")
    assert context.nvars() == 2

    # ARB_MPOLY is intentionally absent from ArithmeticBackend: a multivariate
    # rounded-arb problem must fail at planning time rather than silently fall
    # back to fmpq_mpoly.
    with pytest.raises(ArithmeticBackendError, match="arb"):
        choose_arithmetic_backend(
            WeightOptions(precision="round", rounded_backend="arb"),
            symbolic_variables=("x0", "x1"),
        )

    # Sanity: the same problem under exact precision plans to FMPQ_MPOLY, so
    # the failure above is specific to the rounded-arb path.
    assert (
        choose_arithmetic_backend(
            WeightOptions(),
            symbolic_variables=("x0", "x1"),
        )
        is ArithmeticBackend.FMPQ_MPOLY
    )
