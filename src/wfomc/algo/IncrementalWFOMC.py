from functools import reduce

from wfomc.cell_graph import build_cell_graphs
from wfomc.context import WFOMCContext
from wfomc.network import UnaryEvidenceEncoding
from wfomc.utils import RingElement, Rational, MultinomialCoefficients


def incremental_wfomc(context: WFOMCContext,
                      circle_len: int = None) -> RingElement:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    res = Rational(0, 1)
    predecessor_preds = context.predecessor_preds
    pred_orders = None
    pred_max_order = 0
    if predecessor_preds is not None:
        pred_orders = list(predecessor_preds.keys())
        pred_max_order = max(pred_orders)
    circular_predecessor_pred = context.circular_predecessor_pred
    domain_size = len(domain)
    if circle_len is None:
        circle_len = domain_size
    for cell_graph, weight in build_cell_graphs(
        formula, get_weight,
        leq_pred=leq_pred,
        predecessor_preds=predecessor_preds
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
                    (
                        tuple(int(k == i) for k in range(n_cells)),
                        None if pred_orders is None else tuple(
                            cell for _ in range(pred_max_order)
                        ),
                        None if circular_predecessor_pred is None else cell
                    )
                ] = (
                    cell_graph.get_cell_weight(cell),
                    tuple(cc - 1 if k == j else cc
                          for k, cc in enumerate(pc_ccs))
                )
        else:
            table = dict(
                (
                    (
                        tuple(int(k == i) for k in range(n_cells)),
                        None if pred_orders is None else tuple(
                            cell for _ in range(pred_max_order)
                        ),
                        None if circular_predecessor_pred is None else cell
                    ),
                    (
                        cell_graph.get_cell_weight(cell),
                        None
                    )
                )
                for i, cell in enumerate(cells)
            )

        for cur_idx in range(domain_size - 1):
            old_table = table
            table = dict()
            for j, cell in enumerate(cells):
                w = cell_graph.get_cell_weight(cell)
                for (ivec, last_cells, first_cell), (w_old, old_ccs) in old_table.items():
                    old_ivec = list(ivec)
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

                    w_new = w_old * w
                    # for cycular predecessor
                    if cur_idx == circle_len - 2 and first_cell is not None:
                        w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                            (first_cell, cell), pred_idx
                        )
                        old_ivec[cells.index(first_cell)] -= 1
                    # for predecessors
                    if last_cells is not None:
                        for pred_idx in pred_orders:
                            if cur_idx >= pred_idx - 1:
                                pred_cell = last_cells[-pred_idx]
                                w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                                    (cell, pred_cell), pred_idx
                                )
                                old_ivec[cells.index(pred_cell)] -= 1
                        new_last_cells = last_cells[1:] + (cell,)
                    else:
                        new_last_cells = None
                    w_new = w_new * reduce(
                        lambda x, y: x * y,
                        (
                            cell_graph.get_two_table_weight((cell, other_cell)) ** old_ivec[k]
                            for k, other_cell in enumerate(cells)
                        )
                    )
                    ivec = tuple((num if k != j else num + 1) for k, num in enumerate(ivec))
                    new_last_cells = (
                        tuple(new_last_cells)
                        if new_last_cells is not None else None
                    )
                    w_new = w_new + table.get(
                        (ivec, new_last_cells, first_cell),
                        (Rational(0, 1), ())
                    )[0]
                    table[(tuple(ivec), new_last_cells, first_cell)] = (
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
