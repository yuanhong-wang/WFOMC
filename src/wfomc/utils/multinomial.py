from __future__ import annotations

import functools


def multinomial(length: int, total_sum: int) -> tuple[int]:
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


def multinomial_less_than(length: int, total_sum: int) -> tuple[int]:
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


class MultinomialCoefficients(object):
    """
    Multinomial coefficients

    Usage:
    ```
    MultinomialCoefficients.precompute_pascal(n)
    ...
    MultinomialCoefficients.coef(list)
    ```


    """
    pt: list[list[int]] = None
    n: int = 0

    @staticmethod
    # @jit
    def setup(n: int):
        """
        Pre-compute the pascal triangle.

        :param n int: the maximal total sum
        """
        if n <= MultinomialCoefficients.n:
            return
        pt: list[list[int]] = []
        lst: list[int] = [1]
        for i in range(n + 1):
            pt.append(lst)
            newlist = []
            newlist.append(lst[0])
            for i in range(len(lst) - 1):
                newlist.append(lst[i] + lst[i + 1])
            newlist.append(lst[-1])
            lst = newlist
        MultinomialCoefficients.pt = pt
        MultinomialCoefficients.n = n

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def coef(lst: tuple[int]) -> int:
        """
        Compute the multinomial coefficient of `lst`
        """
        if MultinomialCoefficients.pt is None:
            raise RuntimeError(
                'Please initialize MultinomialCoefficients first by `MultinomialCoefficients.setup(n)`'
            )
        if sum(lst) > MultinomialCoefficients.n:
            raise RuntimeError(
                'The sum %d of input is larger than precomputed maximal sum %d, '
                'please re-initialized MultinomialCoefficients using bigger n',
                sum(lst), MultinomialCoefficients.n
            )
        ret = 1
        tmplist = lst
        while len(tmplist) > 1:
            ret *= MultinomialCoefficients.comb(sum(tmplist), tmplist[-1])
            tmplist = tmplist[:-1]
        return ret

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def comb(a, b):
        if a < b:
            return 0
        elif b == 0:
            return 1
        else:
            return MultinomialCoefficients.pt[a][b]
