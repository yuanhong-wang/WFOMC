from collections import defaultdict
import functools
import pandas as pd
from typing import Callable
from functools import reduce
# from rcviz import callgraph, viz

from wfomc.cell_graph import build_cell_graphs
from wfomc.context.wfomc_context import WFOMCContext
from wfomc.utils import RingElement, Rational
from wfomc.fol.syntax import Const, Pred, QFFormula
from wfomc.utils import multinomial
from wfomc.utils.multinomial import MultinomialCoefficients
from wfomc.utils.polynomial import coeff_dict, expand


def incremental_wfomc(context: WFOMCContext) -> RingElement:
    new_ccs = True

    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    res = Rational(0, 1)
    domain_size = len(domain)


    # prepare for cardinality constraint
    ccs: list[int] = list()
    gen_vars: list[RingElement] = list()
    if context.contain_cardinality_constraint() and new_ccs:
        pred2var = dict(
            (pred, var) for var, pred in
            context.cardinality_constraint.var2pred.items()
        )
        constraints = context.cardinality_constraint.constraints
        for constraint in constraints:
            coeffs, comp, param = constraint
            assert len(coeffs) == 1 and comp == '='
            param = int(param)
            pred, coef = next(iter(coeffs.items()))
            assert coef == 1
            ccs.append(param)
            gen_vars.append(pred2var[pred])

    def helper(poly, gen_vars, ccs):
        for degrees, coeff in coeff_dict(poly, gen_vars):
            if all(
                    degree <= cc for degree, cc in zip(degrees, ccs)
            ):
                yield degrees, coeff

    cell_graph, _ = next(
        build_cell_graphs(
            formula, get_weight, leq_pred=leq_pred
        )
    )
    cells = cell_graph.get_cells()
    print(cells)

    n_decomposed = dict(
        (n, list()) for n in range(domain_size + 1)
    )

    # def dp(old_ivec: tuple[int], old_ccs: tuple[int],
    #        old_coeff: int = Rational(1, 1),
    #        save_decomposed: bool = True) -> RingElement:
    #     decomposed = n_decomposed[sum(old_ivec)]
    #     if all(i == 0 for i in old_ivec):
    #         if all(i == 0 for i in old_ccs):
    #             return Rational(1, 1)
    #         else:
    #             return Rational(0, 1)
    #     ret = Rational(0, 1)
    #     for i, cell in enumerate(cells):
    #         if old_ivec[i] == 0:
    #             continue
    #         new_ivec = list(old_ivec)
    #         new_ivec[i] -= 1
    #         new_ivec = tuple(new_ivec)
    #         w = cell_graph.get_cell_weight(cell)
    #         mul = w * reduce(
    #             lambda x, y: x * y,
    #             (
    #                 cell_graph.get_two_table_weight(
    #                     (cell, cells[k])
    #                 )
    #                 ** int(new_ivec[k]) for k in range(len(cells))
    #             ),
    #             Rational(1, 1)
    #         )
    #         mul = expand(mul)
    #         for deg, coeff in helper(mul, gen_vars, old_ccs):
    #             new_ccs = tuple(
    #                 n - k for n, k in zip(old_ccs, deg)
    #             )
    #             new_coeff = coeff * old_coeff
    #             sub = dp(
    #                 new_ivec, new_ccs, new_coeff,
    #                 sum(new_ccs) != sum(new_ivec)
    #             )
    #             if save_decomposed:
    #                 decomposed.append(
    #                     (new_coeff, new_ivec, new_ccs)
    #                 )
    #             ret = ret + sub * coeff
    #     return ret

    @functools.lru_cache(maxsize=None)
    def dp(old_ivec: tuple[int], old_ccs: tuple[int]) -> RingElement:
        if all(i == 0 for i in old_ivec):
            if all(i == 0 for i in old_ccs):
                return Rational(1, 1)
            else:
                return Rational(0, 1)
        ret = Rational(0, 1)
        for i, cell in enumerate(cells):
            if old_ivec[i] == 0:
                continue
            new_ivec = list(old_ivec)
            new_ivec[i] -= 1
            new_ivec = tuple(new_ivec)
            w = cell_graph.get_cell_weight(cell)
            mul = w * reduce(
                lambda x, y: x * y,
                (
                    cell_graph.get_two_table_weight(
                        (cell, cells[k])
                    )
                    ** int(new_ivec[k]) for k in range(len(cells))
                ),
                Rational(1, 1)
            )
            mul = expand(mul)
            for deg, coeff in helper(mul, gen_vars, old_ccs):
                new_ccs = tuple(
                    n - k for n, k in zip(old_ccs, deg)
                )
                sub = dp(
                    new_ivec, new_ccs
                )
                ret = ret + sub * coeff
        return ret

    for ivec in multinomial(len(cells), domain_size):
        sub = dp(ivec, tuple(ccs))
        res = res + sub
    print(res)

    with open('./tmp.pkl', 'wb') as f:
        import pickle
        pickle.dump(n_decomposed, f)
    decomposed = n_decomposed[domain_size - 1]
    all_decomposed = dict()
    for item in decomposed:
        if item[-1] not in all_decomposed:
            tmp = dict()
        else:
            tmp = all_decomposed[item[-1]]
        if item[1] not in tmp:
            tmp[item[1]] = item[0]
        else:
            tmp[item[1]] = tmp[item[1]] + item[0]
        all_decomposed[item[-1]] = tmp
    df = []
    for k, v in all_decomposed.items():
        for kk, vv in v.items():
            if vv != 0:
                df.append([k, kk, vv])
    df = pd.DataFrame(df, columns=["ccs", "ivec", "coeff"])
    print(df.to_latex())
    # callgraph.render("incremental_wfomc.svg")
    return res
