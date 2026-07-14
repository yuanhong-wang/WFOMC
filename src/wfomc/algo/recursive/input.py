"""Input preparation for the recursive algorithm."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, AlgoOptions
from wfomc.cell_graph import CellGraphComponent, CellGraphData, build_cell_graphs
from wfomc.problem import CompiledProblem
from wfomc.engine.features import FeatureSet


@dataclass(frozen=True)
class RecursiveInput(AlgoInput):
    components: tuple[CellGraphComponent, ...] = ()
    domain_size: int = 0


def build_input(
    reduced: CompiledProblem,
    *,
    options: AlgoOptions,
    features: FeatureSet,
) -> RecursiveInput:
    leq = features.leq_predicate
    components = tuple(
        _component(data, graph_weight)
        for data, graph_weight in build_cell_graphs(
            reduced.sentence,
            reduced.weights,
            reduced.arithmetic,
            leq_pred=leq,
        )
    )
    return RecursiveInput(
        algo=None,
        options=options,
        arithmetic=reduced.arithmetic,
        components=components,
        domain_size=len(reduced.domain),
    )


def _component(
    data: CellGraphData,
    graph_weight: ArithmeticValue,
) -> CellGraphComponent:
    return CellGraphComponent(
        cells=data.cells,
        cell_weights=data.cell_weights,
        pair_weights=data.pair_weights(),
        graph_weight=graph_weight,
    )


__all__ = ["RecursiveInput", "build_input"]
