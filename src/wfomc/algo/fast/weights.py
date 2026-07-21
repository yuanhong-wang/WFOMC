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

import math

from wfomc.algo.symmetric_clique import TwistedBinomialConvolver
from wfomc.arithmetic import ArithmeticValue


def symmetric_clique_row(
    domain_size: int,
    weight: ArithmeticValue,
    self_interaction: ArithmeticValue,
    arithmetic,
) -> tuple[ArithmeticValue, ...]:
    """Return ``weight^n * self_interaction^choose(n, 2)`` for all ``n``."""

    row = [arithmetic.one()]
    value = arithmetic.one()
    interaction_power = arithmetic.one()
    for _count in range(1, domain_size + 1):
        value = arithmetic.multiply(value, weight)
        value = arithmetic.multiply(value, interaction_power)
        row.append(value)
        interaction_power = arithmetic.multiply(
            interaction_power,
            self_interaction,
        )
    return tuple(row)


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
                tmp = ops.arithmetic.multiply(
                    tmp,
                    ops.arithmetic.power(
                        ops.get_two_table_weight(
                            (ops.cliques[i][0], ops.cliques[j][0])
                        ),
                        partition[ops.nonind_map[i]],
                    ),
                )
            accum = ops.arithmetic.add(accum, tmp)
        accum = ops.arithmetic.power(
            accum,
            ops.domain_size - sum(partition) - bign,
        )
        ops.term_cache[key] = accum
        return accum

    sumtoadd = ops.arithmetic.zero()
    s = ops.i2_ind[len(ops.i2_ind) - iv]
    for nval in range(ops.domain_size - sum(partition) - bign + 1):
        smul = ops.arithmetic.from_int(
            math.comb(
                ops.domain_size - sum(partition) - bign,
                nval,
            )
        )
        smul = ops.arithmetic.multiply(smul, optimized_J_term(ops, s, nval))
        if not ops.modified_cell_symmetry:
            smul = ops.arithmetic.multiply(
                smul,
                ops.arithmetic.power(
                    ops.get_cell_weight(ops.cliques[s][0]),
                    nval,
                ),
            )
        for i in ops.nonind:
            smul = ops.arithmetic.multiply(
                smul,
                ops.arithmetic.power(
                    ops.get_two_table_weight(
                        (ops.cliques[i][0], ops.cliques[s][0])
                    ),
                    partition[ops.nonind_map[i]] * nval,
                ),
            )
        smul = ops.arithmetic.multiply(
            smul,
            optimized_term(ops, iv - 1, bign + nval, partition),
        )
        sumtoadd = ops.arithmetic.add(sumtoadd, smul)
    ops.term_cache[key] = sumtoadd
    return sumtoadd


def optimized_J_term(ops, clique_idx: int, nhat: int) -> ArithmeticValue:
    key = (clique_idx, nhat)
    if key in ops.j_term_cache:
        return ops.j_term_cache[key]

    if (
        getattr(ops, "domain_size", None) is not None
        and hasattr(ops, "symmetric_message_cache")
    ):
        thesum = optimized_J_message(ops, clique_idx)[nhat]
    elif len(ops.cliques[clique_idx]) == 1:
        thesum = ops.arithmetic.power(
            ops.get_two_table_weight(
                (ops.cliques[clique_idx][0], ops.cliques[clique_idx][0])
            ),
            int(nhat * (nhat - 1) / 2),
        )
        if ops.modified_cell_symmetry:
            thesum = ops.arithmetic.multiply(
                thesum,
                ops.arithmetic.power(
                    ops.get_cell_weight(ops.cliques[clique_idx][0]),
                    nhat,
                ),
            )
    else:
        thesum = optimized_d_term(ops, clique_idx, nhat)
    ops.j_term_cache[key] = thesum
    return thesum


def optimized_J_message(ops, clique_idx: int) -> tuple[ArithmeticValue, ...]:
    """Build the full cardinality message for one symmetric clique once."""

    cached = ops.symmetric_message_cache.get(clique_idx)
    if cached is not None:
        return cached
    if ops.domain_size is None:
        raise RuntimeError("optimized operations are not bound to a domain")

    clique = ops.cliques[clique_idx]
    if not clique:
        raise ValueError("symmetric cliques must contain at least one cell")
    rows = []
    for cell in clique:
        weight = (
            ops.get_cell_weight(cell)
            if ops.modified_cell_symmetry
            else ops.arithmetic.one()
        )
        rows.append(
            symmetric_clique_row(
                ops.domain_size,
                weight,
                ops.get_two_table_weight((cell, cell)),
                ops.arithmetic,
            )
        )

    interaction = (
        ops.get_two_table_weight((clique[0], clique[1]))
        if len(clique) > 1
        else ops.arithmetic.one()
    )
    message = TwistedBinomialConvolver(
        ops.domain_size,
        interaction,
        ops.arithmetic,
    ).product(rows)
    ops.symmetric_message_cache[clique_idx] = message
    return message


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
            w = ops.arithmetic.power(
                ops.get_cell_weight(ops.cliques[clique_idx][cur]),
                n,
            )
            s = ops.get_two_table_weight(
                (
                    ops.cliques[clique_idx][cur],
                    ops.cliques[clique_idx][cur],
                )
            )
            ret = ops.arithmetic.multiply(
                w,
                ops.arithmetic.power(
                    s,
                    math.comb(n, 2),
                ),
            )
        else:
            ret = ops.arithmetic.power(
                s,
                math.comb(n, 2),
            )
    else:
        ret = ops.arithmetic.zero()
        for ni in range(n + 1):
            mult = ops.arithmetic.from_int(math.comb(n, ni))
            if ops.modified_cell_symmetry:
                w = ops.arithmetic.power(
                    ops.get_cell_weight(ops.cliques[clique_idx][cur]),
                    ni,
                )
                s = ops.get_two_table_weight(
                    (
                        ops.cliques[clique_idx][cur],
                        ops.cliques[clique_idx][cur],
                    )
                )
                mult = ops.arithmetic.multiply(mult, w)
            mult = ops.arithmetic.multiply(
                mult,
                ops.arithmetic.power(
                    s,
                    math.comb(ni, 2),
                ),
            )
            mult = ops.arithmetic.multiply(
                mult,
                ops.arithmetic.power(r, ni * (n - ni)),
            )
            mult = ops.arithmetic.multiply(
                mult,
                optimized_d_term(ops, clique_idx, n - ni, cur + 1),
            )
            ret = ops.arithmetic.add(ret, mult)
    ops.d_term_cache[key] = ret
    return ret


__all__ = [
    "optimized_term",
    "optimized_J_term",
    "optimized_J_message",
    "optimized_d_term",
    "symmetric_clique_row",
]
