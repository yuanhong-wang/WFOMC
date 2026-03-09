# cython: language_level=3
# cython: boundscheck=False, wraparound=True, nonecheck=False, cdivision=True
from __future__ import annotations

import functools


def multinomial(int length, int total_sum):
    """
    Generate tuples of `length` non-negative integers summing to `total_sum`.
    """
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial(length - 1, total_sum - value):
                yield (value,) + permutation


def multinomial_less_than(int length, int total_sum):
    """
    Generate tuples of `length` non-negative integers summing to at most `total_sum`.
    """
    if length == 0:
        yield ()
        return
    if length == 1:
        for i in range(total_sum + 1):
            yield (i,)
    else:
        for value in range(total_sum + 1):
            for permutation in multinomial_less_than(length - 1, total_sum - value):
                yield (value,) + permutation


class MultinomialCoefficients(object):
    """
    Multinomial coefficients backed by a precomputed Pascal triangle.
    """
    pt: list = None
    n: int = 0

    @staticmethod
    def setup(int n):
        cdef int i
        if n <= MultinomialCoefficients.n:
            return
        pt = []
        lst = [1]
        for i in range(n + 1):
            pt.append(lst)
            newlist = []
            newlist.append(lst[0])
            for j in range(len(lst) - 1):
                newlist.append(lst[j] + lst[j + 1])
            newlist.append(lst[-1])
            lst = newlist
        MultinomialCoefficients.pt = pt
        MultinomialCoefficients.n = n

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def coef(tuple lst) -> int:
        cdef int ret = 1
        cdef tuple tmplist
        if MultinomialCoefficients.pt is None:
            raise RuntimeError(
                'Please initialize MultinomialCoefficients first by '
                '`MultinomialCoefficients.setup(n)`'
            )
        if sum(lst) > MultinomialCoefficients.n:
            raise RuntimeError(
                f'The sum {sum(lst)} of input is larger than precomputed '
                f'maximal sum {MultinomialCoefficients.n}, '
                'please re-initialized MultinomialCoefficients using bigger n'
            )
        tmplist = lst
        while len(tmplist) > 1:
            ret *= MultinomialCoefficients.comb(sum(tmplist), tmplist[-1])
            tmplist = tmplist[:-1]
        return ret

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def comb(int a, int b) -> int:
        if a < b:
            return 0
        elif b == 0:
            return 1
        else:
            return MultinomialCoefficients.pt[a][b]
