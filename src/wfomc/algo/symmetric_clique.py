"""Shared dense convolution for symmetric cell-graph cliques."""

from __future__ import annotations

from collections.abc import Sequence

from wfomc.arithmetic import ArithmeticValue


class _PowerRow:
    """Incrementally materialized ``1, base, base^2, ...`` values."""

    def __init__(self, base: ArithmeticValue, arithmetic):
        self.base = base
        self.arithmetic = arithmetic
        self.values = [arithmetic.one()]

    def get(self, exponent: int) -> ArithmeticValue:
        while len(self.values) <= exponent:
            self.values.append(
                self.arithmetic.multiply(self.values[-1], self.base)
            )
        return self.values[exponent]


class _ExponentCache:
    """Bounded cache for interaction exponents ``left * right``."""

    _SYMBOLIC_CACHE_LIMIT = 8192

    def __init__(self, base: ArithmeticValue, max_exponent: int, arithmetic):
        self.base = base
        self.arithmetic = arithmetic
        self.sequential = (
            not arithmetic.symbolic_variables or max_exponent <= 4096
        )
        self.row = _PowerRow(base, arithmetic) if self.sequential else None
        self.values: dict[int, ArithmeticValue] = {0: arithmetic.one()}

    def get(self, exponent: int) -> ArithmeticValue:
        if self.row is not None:
            return self.row.get(exponent)
        cached = self.values.get(exponent)
        if cached is not None:
            return cached
        value = self.arithmetic.power(self.base, exponent)
        if len(self.values) < self._SYMBOLIC_CACHE_LIMIT:
            self.values[exponent] = value
        return value


class _BinomialValues:
    """Pascal rows plus lazily coerced backend values."""

    def __init__(self, maximum: int, arithmetic):
        rows: list[tuple[int, ...]] = []
        current = [1]
        for _ in range(maximum + 1):
            rows.append(tuple(current))
            current = [1] + [
                current[index] + current[index + 1]
                for index in range(len(current) - 1)
            ] + [1]
        self.rows = tuple(rows)
        self.arithmetic = arithmetic
        self.values: dict[tuple[int, int], ArithmeticValue] = {}

    def value(self, total: int, selected: int) -> ArithmeticValue:
        key = (total, selected)
        cached = self.values.get(key)
        if cached is None:
            cached = self.arithmetic.from_int(self.rows[total][selected])
            self.values[key] = cached
        return cached


class TwistedBinomialConvolver:
    """Combine symmetric-clique cardinality rows up to one domain size.

    For interaction weight ``r``, the product is

    ``C[n] = sum_i choose(n, i) A[i] B[n-i] r^(i(n-i))``.

    This product is associative and commutative.  Repeated equal rows can
    therefore be combined with binary exponentiation.
    """

    def __init__(
        self,
        domain_size: int,
        interaction: ArithmeticValue,
        arithmetic,
    ):
        if domain_size < 0:
            raise ValueError("domain_size must be non-negative")
        self.domain_size = domain_size
        self.interaction = interaction
        self.arithmetic = arithmetic
        self.binomials = _BinomialValues(domain_size, arithmetic)
        max_cross = (domain_size // 2) * (domain_size - domain_size // 2)
        self.interaction_powers = _ExponentCache(
            interaction,
            max_cross,
            arithmetic,
        )

    def _row(
        self,
        values: Sequence[ArithmeticValue],
    ) -> tuple[ArithmeticValue, ...]:
        if len(values) != self.domain_size + 1:
            raise ValueError(
                "symmetric-clique rows must contain domain_size + 1 values"
            )
        return tuple(values)

    def identity(self) -> tuple[ArithmeticValue, ...]:
        return (self.arithmetic.one(),) + (
            self.arithmetic.zero(),
        ) * self.domain_size

    def combine(
        self,
        left: Sequence[ArithmeticValue],
        right: Sequence[ArithmeticValue],
    ) -> tuple[ArithmeticValue, ...]:
        """Return the truncated twisted binomial convolution of two rows."""

        left_row = self._row(left)
        right_row = self._row(right)
        output = [self.arithmetic.zero() for _ in range(self.domain_size + 1)]
        for total in range(self.domain_size + 1):
            accumulator = self.arithmetic.zero()
            for left_count in range(total + 1):
                left_value = left_row[left_count]
                right_value = right_row[total - left_count]
                if self.arithmetic.is_zero(left_value) or self.arithmetic.is_zero(
                    right_value
                ):
                    continue
                factor = self.binomials.value(total, left_count)
                cross_exponent = left_count * (total - left_count)
                if cross_exponent:
                    factor = self.arithmetic.multiply(
                        factor,
                        self.interaction_powers.get(cross_exponent),
                    )
                factor = self.arithmetic.multiply(factor, left_value)
                accumulator = self.arithmetic.add_product(
                    accumulator,
                    factor,
                    right_value,
                )
            output[total] = accumulator
        return tuple(output)

    def power(
        self,
        base: Sequence[ArithmeticValue],
        multiplicity: int,
    ) -> tuple[ArithmeticValue, ...]:
        """Raise one row under convolution using binary exponentiation."""

        if multiplicity < 0:
            raise ValueError("multiplicity must be non-negative")
        current = self._row(base)
        result: tuple[ArithmeticValue, ...] | None = None
        remaining = multiplicity
        while remaining:
            if remaining & 1:
                result = current if result is None else self.combine(result, current)
            remaining >>= 1
            if remaining:
                current = self.combine(current, current)
        return self.identity() if result is None else result

    def product(
        self,
        rows: Sequence[Sequence[ArithmeticValue]],
    ) -> tuple[ArithmeticValue, ...]:
        """Group equal rows, power each group, and combine the groups."""

        groups: list[tuple[tuple[ArithmeticValue, ...], int]] = []
        for values in rows:
            row = self._row(values)
            for index, (existing, count) in enumerate(groups):
                if existing == row:
                    groups[index] = (existing, count + 1)
                    break
            else:
                groups.append((row, 1))

        result: tuple[ArithmeticValue, ...] | None = None
        for row, multiplicity in groups:
            powered = self.power(row, multiplicity)
            result = powered if result is None else self.combine(result, powered)
        return self.identity() if result is None else result


__all__ = ["TwistedBinomialConvolver"]
