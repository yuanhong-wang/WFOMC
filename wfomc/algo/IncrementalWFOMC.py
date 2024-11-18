from typing import Callable
from functools import reduce

from wfomc.cell_graph import build_cell_graphs
from wfomc.context.wfomc_context import WFOMCContext
from wfomc.network.constraint import UnaryEvidenceEncoding
from wfomc.utils import RingElement, Rational
from wfomc.fol.syntax import Const, Pred, QFFormula
from wfomc.utils.multinomial import MultinomialCoefficients


def incremental_wfomc(context: WFOMCContext) -> RingElement:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    predecessor_pred = context.predecessor_pred
    circular_predecessor_pred = context.circular_predecessor_pred
    res = Rational(0, 1)
    domain_size = len(domain)
    for cell_graph, weight in build_cell_graphs(
        formula, get_weight,
        leq_pred=leq_pred,
        predecessor_pred=predecessor_pred
    ):
        # cell_graph.show()
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        domain_size = len(domain)

        def helper(cell, pc_pred, pc_ccs):
            for i, p in enumerate(pc_pred):
                if cell.is_positive(p) and pc_ccs[i] > 0:
                    return i
            return None

        if context.unary_evidence_encoding == \
                UnaryEvidenceEncoding.PC:
            pc_pred, pc_ccs = zip(*context.partition_constraint.partition)
            table = dict()
            for i, cell in enumerate(cells):
                j = helper(cell, pc_pred, pc_ccs)
                if j is None:
                    continue
                if predecessor_pred is None:
                    table[tuple(int(k == i) for k in range(n_cells))]  = (
                        cell_graph.get_cell_weight(cell),
                        tuple(cc - 1 if k == j else cc
                              for k, cc in enumerate(pc_ccs))
                    )
                else:
                    if circular_predecessor_pred is None:
                        table[(tuple(int(k == i) for k in range(n_cells)), cell)] = (
                            cell_graph.get_cell_weight(cell),
                            tuple(cc - 1 if k == j else cc
                                  for k, cc in enumerate(pc_ccs))
                        )
                    else:
                        table[(tuple(int(k == i) for k in range(n_cells)), cell, cell)] = (
                            cell_graph.get_cell_weight(cell),
                            tuple(cc - 1 if k == j else cc
                                  for k, cc in enumerate(pc_ccs))
                        )
        else:
            if predecessor_pred is None:
                table = dict(
                    (
                        tuple(int(k == i) for k in range(n_cells)),
                        (
                            cell_graph.get_cell_weight(cell),
                            None
                        )
                    )
                    for i, cell in enumerate(cells)
                )
            else:
                if circular_predecessor_pred is not None:
                    table = dict(
                        (
                            (
                                tuple(int(k == i) for k in range(n_cells)),
                                cell,
                                cell
                            ),
                            (
                                cell_graph.get_cell_weight(cell),
                                None
                            )
                        )
                        for i, cell in enumerate(cells)
                    )
                else:
                    table = dict(
                        (
                            (
                                tuple(int(k == i) for k in range(n_cells)),
                                cell
                            ),
                            (
                                cell_graph.get_cell_weight(cell),
                                None
                            )
                        )
                        for i, cell in enumerate(cells)
                    )

        for _ in range(domain_size - 1):
            old_table = table
            table = dict()
            for j, cell in enumerate(cells):
                w = cell_graph.get_cell_weight(cell)
                for key, (w_old, old_ccs) in old_table.items():
                    if old_ccs is not None:
                        idx = helper(cell, pc_pred, old_ccs)
                        if idx is None:
                            continue
                        new_ccs = tuple(
                            cc - 1 if k == idx else cc
                            for k, cc in enumerate(old_ccs)
                        )
                    else:
                        new_ccs = None

                    if predecessor_pred is None:
                        ivec = key
                        w_new = w_old * w * reduce(
                            lambda x, y: x * y,
                            (
                                cell_graph.get_two_table_weight((cell, cells[k]))
                                ** int(ivec[k]) for k in range(n_cells)
                            ),
                            Rational(1, 1)
                        )
                    else:
                        if circular_predecessor_pred is not None:
                            ivec, first_cell, last_cell = key
                        else:
                            ivec, last_cell = key
                        # ivec, last_cell = key
                        w_new = w_old * w
                        for k, other_cell in enumerate(cells):
                            if other_cell == last_cell:
                                w_new = (
                                    w_new * cell_graph.get_two_table_with_pred_weight((cell, other_cell))
                                    * cell_graph.get_two_table_weight((cell, other_cell)) ** max(ivec[k] - 1, 0)
                                )
                            else:
                                w_new = w_new * cell_graph.get_two_table_weight((cell, other_cell)) ** ivec[k]
                    ivec = list(ivec)
                    ivec[j] += 1
                    ivec = tuple(ivec)
                    if predecessor_pred is None:
                        w_new = w_new + table.get(ivec, (Rational(0, 1), ()))[0]
                        table[ivec] = (
                            w_new, new_ccs
                        )
                    else:
                        if circular_predecessor_pred is not None:
                            w_new = w_new + table.get((ivec, first_cell, cell), (Rational(0, 1), ()))[0]
                            table[(tuple(ivec), first_cell, cell)] = (
                            # table[(tuple(ivec), cell)] = (
                                w_new, new_ccs
                            )
                        else:
                            w_new = w_new + table.get((ivec, cell), (Rational(0, 1), ()))[0]
                            table[(tuple(ivec), cell)] = (
                            # table[(tuple(ivec), cell)] = (
                                w_new, new_ccs
                            )
        if circular_predecessor_pred is not None:
            w_sum = Rational(0, 1)
            for key, (w, _) in table.items():
                if w == 0:
                    continue
                _, first_cell, last_cell = key
                w = w / cell_graph.get_two_table_weight((last_cell, first_cell)) * \
                cell_graph.get_two_table_with_pred_weight((first_cell, last_cell))
                w_sum = w_sum + w
            res = res + weight * w_sum
        else:
            res = res + weight * sum(w for w, _ in table.values())

    if context.unary_evidence_encoding == \
            UnaryEvidenceEncoding.PC:
        res = res / MultinomialCoefficients.coef(
            tuple(
                i for _, i in context.partition_constraint.partition
            )
        )
    # if leq_pred is not None:
    #     res *= Rational(math.factorial(domain_size), 1)
    return res
