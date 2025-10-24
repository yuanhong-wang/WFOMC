from functools import reduce

from wfomc.cell_graph import build_cell_graphs
from wfomc.context import WFOMCContext
from wfomc.network import UnaryEvidenceEncoding
from wfomc.utils import RingElement, Rational, MultinomialCoefficients


def incremental_wfomc(context: WFOMCContext) -> RingElement:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    res = Rational(0, 1)
    domain_size = len(domain)
    for cell_graph, weight in build_cell_graphs(
            formula, get_weight, leq_pred=leq_pred
    ):
        # cell_graph.show()
        cells = cell_graph.get_cells()
        n_cells = len(cells)

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
                table[
                    tuple(int(k == i) for k in range(n_cells))
                ] = (
                    cell_graph.get_cell_weight(cell),
                    tuple(cc - 1 if k == j else cc
                          for k, cc in enumerate(pc_ccs))
                )
        else:
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

        for _ in range(domain_size - 1):
            old_table = table
            table = dict()
            for j, cell in enumerate(cells):
                w = cell_graph.get_cell_weight(cell)
                for ivec, (w_old, old_ccs) in old_table.items():
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

                    w_new = w_old * w * reduce(
                        lambda x, y: x * y,
                        (
                            cell_graph.get_two_table_weight((cell, cells[k]))
                            ** int(ivec[k]) for k in range(n_cells)
                        ),
                        Rational(1, 1)
                    )
                    ivec = list(ivec)
                    ivec[j] += 1
                    ivec = tuple(ivec)
                    w_new = w_new + table.get(ivec, (Rational(0, 1), ()))[0]
                    table[tuple(ivec)] = (
                        w_new, new_ccs
                    )
        res = res + weight * sum(w for w, _ in table.values())

    if context.unary_evidence_encoding == \
            UnaryEvidenceEncoding.PC:
        res = res / MultinomialCoefficients.coef(
            tuple(
                i for _, i in context.partition_constraint.partition
            )
        )
    return res
