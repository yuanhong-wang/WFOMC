"""Shared multinomial enumeration and coefficient helpers."""

from __future__ import annotations

import math
from typing import Generator


__all__ = [
    "multinomial",
    "multinomial_coefficient",
    "multinomial_less_than",
]


def multinomial(length: int, total_sum: int) -> Generator[tuple[int], None, None]:
    """
    Generate a list of numbers, whose size is `length` and sum is `total_sum`

    :param length int: length of the generated list
    :param total_sum int: the summation over the list
    :rtype tuple[int]:
    """
    if length == 1:
        yield (total_sum, )
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial(length - 1, total_sum - value):
                yield (value, ) + permutation


def multinomial_less_than(length: int, total_sum: int) -> Generator[tuple[int], None, None]:
    """
    Generate a list of numbers, whose size is `length` and sum is less than `total_sum`

    :param length int: length of the generated list
    :param total_sum int: the summation over the list
    :rtype tuple[int]:
    """
    if length == 0:
        yield ()
        return
    if length == 1:
        for i in range(total_sum + 1):
            yield (i, )
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial_less_than(length - 1, total_sum - value):
                yield (value, ) + permutation


def multinomial_coefficient(parts: tuple[int, ...]) -> int:
    """Return the exact multinomial coefficient for non-negative parts."""

    if any(part < 0 for part in parts):
        raise ValueError("multinomial parts must be non-negative")
    coefficient = 1
    remaining = sum(parts)
    for part in reversed(parts[1:]):
        coefficient *= math.comb(remaining, part)
        remaining -= part
    return coefficient
