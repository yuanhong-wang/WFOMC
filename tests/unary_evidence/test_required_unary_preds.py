from wfomc.cell_graph import CellGraph
from wfomc.fol import Pred, X
from wfomc.utils import Rational


def test_required_unary_predicate_is_assigned_and_weighted_once():
    t = Pred("RequiredT", 1)
    evidence_only = Pred("RequiredEvidenceOnly", 1)

    def get_weight(pred):
        if pred == evidence_only:
            return Rational(2, 1), Rational(3, 1)
        return Rational(1, 1), Rational(1, 1)

    graph = CellGraph(
        t(X),
        get_weight,
        required_unary_preds=frozenset({evidence_only}),
    )

    assert len(graph.get_cells()) == 2
    assert {
        graph.get_cell_weight(cell)
        for cell in graph.get_cells()
    } == {Rational(2, 1), Rational(3, 1)}
    assert all(
        graph.get_two_table_weight((left, right)) == Rational(1, 1)
        for left in graph.get_cells()
        for right in graph.get_cells()
    )
