"""Fast/fastv2 optimized weight recursion.

Single implementation of the symmetric-clique weight DP used by both clique
discovery and the
table-only ``MaterializedOptimizedOperations`` snapshot consumed by the fast
algorithm. Each function takes an ``ops`` object exposing the structure indices
(``cliques`` / ``nonind`` / ``nonind_map`` / ``i1_ind`` / ``i2_ind`` /
``domain_size`` / ``modified_cell_symmetry``), the weight accessors
(``get_cell_weight`` / ``get_two_table_weight``), and per-object memo dicts
(``term_cache`` / ``j_term_cache`` / ``d_term_cache``).
"""

from __future__ import annotations

from wfomc.arithmetic import ArithmeticValue
from wfomc.multinomial import MultinomialCoefficients


def optimized_term(
    ops, iv: int, bign: int, partition: tuple[int, ...]
) -> ArithmeticValue:
    key = (iv, bign)
    if key in ops.term_cache:
        return ops.term_cache[key]

    if iv == 0:
        accum = ops.arithmetic.zero()
        for j in ops.i1_ind:
            tmp = ops.get_cell_weight(ops.cliques[j][0])
            for i in ops.nonind:
                tmp = (
                    tmp
                    * ops.get_two_table_weight((ops.cliques[i][0], ops.cliques[j][0]))
                    ** partition[ops.nonind_map[i]]
                )
            accum = accum + tmp
        accum = accum ** (ops.domain_size - sum(partition) - bign)
        ops.term_cache[key] = accum
        return accum

    sumtoadd = ops.arithmetic.zero()
    s = ops.i2_ind[len(ops.i2_ind) - iv]
    for nval in range(ops.domain_size - sum(partition) - bign + 1):
        smul = ops.arithmetic.from_int(
            MultinomialCoefficients.comb(
                ops.domain_size - sum(partition) - bign,
                nval,
            )
        )
        smul = smul * optimized_J_term(ops, s, nval)
        if not ops.modified_cell_symmetry:
            smul = smul * ops.get_cell_weight(ops.cliques[s][0]) ** nval
        for i in ops.nonind:
            smul = smul * ops.get_two_table_weight(
                (ops.cliques[i][0], ops.cliques[s][0])
            ) ** (partition[ops.nonind_map[i]] * nval)
        smul = smul * optimized_term(ops, iv - 1, bign + nval, partition)
        sumtoadd = sumtoadd + smul
    ops.term_cache[key] = sumtoadd
    return sumtoadd


def optimized_J_term(ops, clique_idx: int, nhat: int) -> ArithmeticValue:
    key = (clique_idx, nhat)
    if key in ops.j_term_cache:
        return ops.j_term_cache[key]

    if len(ops.cliques[clique_idx]) == 1:
        thesum = ops.get_two_table_weight(
            (ops.cliques[clique_idx][0], ops.cliques[clique_idx][0])
        ) ** (int(nhat * (nhat - 1) / 2))
        if ops.modified_cell_symmetry:
            thesum = thesum * ops.get_cell_weight(ops.cliques[clique_idx][0]) ** nhat
    else:
        thesum = optimized_d_term(ops, clique_idx, nhat)
    ops.j_term_cache[key] = thesum
    return thesum


def optimized_d_term(ops, clique_idx: int, n: int, cur: int = 0) -> ArithmeticValue:
    key = (clique_idx, n, cur)
    if key in ops.d_term_cache:
        return ops.d_term_cache[key]

    clique_size = len(ops.cliques[clique_idx])
    r = ops.get_two_table_weight(
        (ops.cliques[clique_idx][0], ops.cliques[clique_idx][1])
    )
    s = ops.get_two_table_weight(
        (ops.cliques[clique_idx][0], ops.cliques[clique_idx][0])
    )
    if cur == clique_size - 1:
        if ops.modified_cell_symmetry:
            w = ops.get_cell_weight(ops.cliques[clique_idx][cur]) ** n
            s = ops.get_two_table_weight(
                (
                    ops.cliques[clique_idx][cur],
                    ops.cliques[clique_idx][cur],
                )
            )
            ret = w * s ** MultinomialCoefficients.comb(n, 2)
        else:
            ret = s ** MultinomialCoefficients.comb(n, 2)
    else:
        ret = ops.arithmetic.zero()
        for ni in range(n + 1):
            mult = ops.arithmetic.from_int(MultinomialCoefficients.comb(n, ni))
            if ops.modified_cell_symmetry:
                w = ops.get_cell_weight(ops.cliques[clique_idx][cur]) ** ni
                s = ops.get_two_table_weight(
                    (
                        ops.cliques[clique_idx][cur],
                        ops.cliques[clique_idx][cur],
                    )
                )
                mult = mult * w
            mult = mult * (s ** MultinomialCoefficients.comb(ni, 2))
            mult = mult * r ** (ni * (n - ni))
            mult = mult * optimized_d_term(ops, clique_idx, n - ni, cur + 1)
            ret = ret + mult
    ops.d_term_cache[key] = ret
    return ret


__all__ = ["optimized_term", "optimized_J_term", "optimized_d_term"]
