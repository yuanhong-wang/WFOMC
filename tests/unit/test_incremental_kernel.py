"""Focused tests for the ordered incremental DP hot path."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.algo.incremental.input import OrderedCellGraphComponent
from wfomc.algo.incremental.solve import _solve_component


@dataclass(frozen=True)
class _TrackedValue:
    value: int
    arithmetic: "_CountingArithmetic"

    def __mul__(self, other: object) -> "_TrackedValue":
        self.arithmetic.raw_multiply_calls += 1
        return _TrackedValue(self.value * _value(other), self.arithmetic)

    def __add__(self, other: object) -> "_TrackedValue":
        return _TrackedValue(self.value + _value(other), self.arithmetic)

    def __pow__(self, exponent: int) -> "_TrackedValue":
        self.arithmetic.raw_power_calls += 1
        return _TrackedValue(self.value**exponent, self.arithmetic)

    def __eq__(self, other: object) -> bool:
        return self.value == _value(other)


def _value(value: object) -> int:
    return value.value if isinstance(value, _TrackedValue) else int(value)


@dataclass
class _CountingArithmetic:
    raw_multiply_calls: int = 0
    raw_power_calls: int = 0

    def zero(self) -> _TrackedValue:
        return _TrackedValue(0, self)

    def one(self) -> _TrackedValue:
        return _TrackedValue(1, self)

    @staticmethod
    def is_zero(value: object) -> bool:
        return _value(value) == 0

    @staticmethod
    def is_one(value: object) -> bool:
        return _value(value) == 1

    def multiply(self, left: _TrackedValue, right: _TrackedValue) -> _TrackedValue:
        return _TrackedValue(left.value * right.value, self)

    def add(self, left: _TrackedValue, right: _TrackedValue) -> _TrackedValue:
        return _TrackedValue(left.value + right.value, self)

    def power(self, base: _TrackedValue, exponent: int) -> _TrackedValue:
        return _TrackedValue(base.value**exponent, self)


def test_incremental_skips_identity_pairs_and_zero_predecessor_edges():
    arithmetic = _CountingArithmetic()
    zero = arithmetic.zero()
    one = arithmetic.one()
    component = OrderedCellGraphComponent(
        cells=("left", "right"),
        cell_weights=(one, one),
        pair_weights=((one, one), (one, one)),
        graph_weight=one,
        predk_pair_tables={1: ((one, zero), (one, one))},
    )

    result = _solve_component(
        component,
        domain_size=2,
        predecessor_orders=(1,),
        predecessor_max_order=1,
        has_circular_predecessor=False,
        circle_len=2,
        arithmetic=arithmetic,
    )

    assert result == 3
    assert arithmetic.raw_multiply_calls == 0
    assert arithmetic.raw_power_calls == 0
