"""Fast evidence-aware cell-graph layout regressions."""

from types import MethodType
from typing import Any, cast

import networkx as nx

from wfomc.algo.fast.graph import (
    CellWithEvidenceProfile,
    _EvidenceOptimizedAnalysis,
)
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext


def _profile_biased_analysis() -> _EvidenceOptimizedAnalysis:
    """Build a graph where one small-profile node blocks three large-profile nodes."""

    analysis = object.__new__(_EvidenceOptimizedAnalysis)
    analysis.arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    analysis.evidence_profile_sizes = (20, 1)
    analysis.cells = [
        CellWithEvidenceProfile(cast(Any, node), profile)
        for node, profile in ((0, 0), (1, 0), (2, 0), (3, 1))
    ]
    blocking_edges = {
        frozenset((0, 3)),
        frozenset((1, 3)),
        frozenset((2, 3)),
    }

    def get_two_table_weight(
        self: _EvidenceOptimizedAnalysis,
        cells: tuple[CellWithEvidenceProfile, CellWithEvidenceProfile],
    ) -> object:
        left, right = (cell.base_cell for cell in cells)
        if left == right:
            return self.arithmetic.one()
        return self.arithmetic.from_int(
            2 if frozenset((left, right)) in blocking_edges else 1
        )

    analysis.get_two_table_weight = MethodType(get_two_table_weight, analysis)
    return analysis


def test_evidence_independent_set_prioritizes_largest_profile(monkeypatch) -> None:
    """A small evidence profile must not crowd the dominant profile out of I1."""

    def choose_small_profile_first(graph, nodes=None, seed=None):
        del graph, seed
        return [3] if nodes is None else list(nodes)

    monkeypatch.setattr(nx, "maximal_independent_set", choose_small_profile_first)
    analysis = _profile_biased_analysis()

    i1, i2, nonindependent = analysis._find_independent_sets()

    assert i1 == [0, 1, 2]
    assert i2 == []
    assert nonindependent == [3]
