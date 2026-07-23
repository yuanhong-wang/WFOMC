"""Fast-specific clique analysis over immutable base cell-graph data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from .weights import optimized_J_term, optimized_d_term
from wfomc.arithmetic import ArithmeticValue
from wfomc.cell_graph import (
    Cell,
    CellEvidenceAllocation,
    CellGraphData,
    build_cell_graphs,
)
from wfomc.fol import Formula, Predicate

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext
    from wfomc.evidence.profile import ProfileCapacityConstraint


class _GraphView:
    """Indexed weight view used only while preparing a fast input."""

    def __init__(self, data: CellGraphData):
        self.data = data
        self.arithmetic = data.arithmetic
        self.cells = list(data.cells)
        self._cell_index = {cell: idx for idx, cell in enumerate(data.cells)}

    def get_cells(self) -> list[Cell]:
        return self.cells

    def get_cell_weight(self, cell: Cell) -> ArithmeticValue:
        return self.data.cell_weights[self._cell_index[cell]]

    def get_two_table_weight(
        self,
        cells: tuple[Cell, Cell],
    ) -> ArithmeticValue:
        left, right = (self._cell_index[cell] for cell in cells)
        return self.data.pair_factors[left][right].total_weight

    def get_all_weights(
        self,
    ) -> tuple[list[ArithmeticValue], list[list[ArithmeticValue]]]:
        return (
            list(self.data.cell_weights),
            [list(row) for row in self.data.pair_weights()],
        )


class _OptimizedAnalysis(_GraphView):
    def __init__(
        self,
        data: CellGraphData,
        modified_cell_symmetry: bool,
    ):
        super().__init__(data)
        self.modified_cell_symmetry = modified_cell_symmetry
        self.i1_evidence_profile_partition = None
        self.evidence_profile_sizes = None
        self.evidence_profile_cliques = None
        self.term_cache: dict = {}
        self.j_term_cache: dict = {}
        self.d_term_cache: dict = {}
        if modified_cell_symmetry:
            i1, i2, nonind = self._find_independent_sets()
            self.cliques, groups = self._build_symmetric_cliques_in_groups(
                (i1, i2, nonind)
            )
            self.i1_ind, self.i2_ind, self.nonind = groups
        else:
            self.cliques = self._build_symmetric_cliques()
            self.i1_ind, self.i2_ind, _ind, self.nonind = (
                self._find_independent_cliques()
            )
        self.nonind_map = {
            clique_idx: index for index, clique_idx in enumerate(self.nonind)
        }

    def _build_symmetric_cliques(self) -> list[list[Cell]]:
        cliques: list[list[Cell]] = []
        remaining = list(self.cells)
        while remaining:
            cell = remaining.pop()
            clique = [cell]
            for other in remaining:
                if self._matches(clique, other):
                    clique.append(other)
            for other in clique[1:]:
                remaining.remove(other)
            cliques.append(clique)
        cliques.sort(key=len)
        return cliques

    def _build_symmetric_cliques_in_groups(
        self,
        cell_index_groups: tuple[list[int], list[int], list[int]],
    ) -> tuple[list[list[Cell]], list[list[int]]]:
        i1_indices = set(cell_index_groups[0])
        cliques: list[list[Cell]] = []
        clique_groups: list[list[int]] = []
        for source_group in cell_index_groups:
            remaining = list(source_group)
            group = []
            while remaining:
                cell_idx = remaining.pop()
                clique = [self.cells[cell_idx]]
                if cell_idx not in i1_indices:
                    for other_idx in remaining:
                        other = self.cells[other_idx]
                        if self._matches(clique, other):
                            clique.append(other)
                    for other in clique[1:]:
                        remaining.remove(self.cells.index(other))
                cliques.append(clique)
                group.append(len(cliques) - 1)
            clique_groups.append(group)
        return cliques, clique_groups

    def _find_independent_sets(self) -> tuple[list[int], list[int], list[int]]:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.cells)))
        if not graph:
            return [], [], []
        for left in range(len(self.cells)):
            for right in range(left + 1, len(self.cells)):
                if not self.arithmetic.equal(
                    self.get_two_table_weight((self.cells[left], self.cells[right])),
                    self.arithmetic.one(),
                ):
                    graph.add_edge(left, right)

        self_loops = {
            index
            for index, cell in enumerate(self.cells)
            if not self.arithmetic.equal(
                self.get_two_table_weight((cell, cell)),
                self.arithmetic.one(),
            )
        }
        without_self_loops = set(graph.nodes) - self_loops
        i1 = (
            set()
            if not without_self_loops
            else set(nx.maximal_independent_set(graph.subgraph(without_self_loops)))
        )
        independent = set(nx.maximal_independent_set(graph, nodes=i1))
        i2 = independent - i1
        nonind = set(graph.nodes) - i1 - i2
        return list(i1), list(i2), list(nonind)

    def _find_independent_cliques(
        self,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.cliques)))
        for left in range(len(self.cliques)):
            for right in range(left + 1, len(self.cliques)):
                if not self.arithmetic.equal(
                    self.get_two_table_weight(
                        (self.cliques[left][0], self.cliques[right][0])
                    ),
                    self.arithmetic.one(),
                ):
                    graph.add_edge(left, right)

        # Clique independence is structural. A multi-cell clique necessarily
        # has an internal choice interaction; a singleton is self-interacting
        # exactly when its diagonal pair factor is non-unit. Do not probe
        # J-terms up to the current domain size here: the layout is reusable
        # across every n.
        self_loops = {
            clique_idx
            for clique_idx, clique in enumerate(self.cliques)
            if len(clique) > 1
            or not self.arithmetic.equal(
                self.get_two_table_weight((clique[0], clique[0])),
                self.arithmetic.one(),
            )
        }
        without_self_loops = set(graph.nodes) - self_loops
        independent = (
            set()
            if not without_self_loops
            else set(nx.maximal_independent_set(graph.subgraph(without_self_loops)))
        )
        i2 = independent & self_loops
        i1 = independent - i2
        nonind = set(graph.nodes) - i1 - i2
        return list(i1), list(i2), list(independent), list(nonind)

    def _matches(self, clique: list[Cell], other: Cell) -> bool:
        cell = clique[0]
        if not self.modified_cell_symmetry and (
            not self.arithmetic.equal(
                self.get_cell_weight(cell),
                self.get_cell_weight(other),
            )
            or not self.arithmetic.equal(
                self.get_two_table_weight((cell, cell)),
                self.get_two_table_weight((other, other)),
            )
        ):
            return False
        if len(clique) > 1:
            relation = self.get_two_table_weight((cell, clique[1]))
            if any(
                not self.arithmetic.equal(
                    relation,
                    self.get_two_table_weight((other, third)),
                )
                for third in clique
            ):
                return False
        return all(
            self.arithmetic.equal(
                self.get_two_table_weight((cell, third)),
                self.get_two_table_weight((other, third)),
            )
            for third in self.cells
            if other != third and third not in clique
        )

    def get_J_term(self, clique_idx: int, count: int) -> ArithmeticValue:
        return optimized_J_term(self, clique_idx, count)

    def get_d_term(
        self,
        clique_idx: int,
        count: int,
        current: int = 0,
    ) -> ArithmeticValue:
        return optimized_d_term(self, clique_idx, count, current)


@dataclass(frozen=True)
class CellWithEvidenceProfile:
    base_cell: Cell
    evidence_profile_index: int

    def is_positive(self, predicate: Predicate) -> bool:
        return self.base_cell.is_positive(predicate)

    def __str__(self) -> str:
        return f"{self.base_cell} [evidence_profile={self.evidence_profile_index}]"


class _EvidenceOptimizedAnalysis:
    def __init__(
        self,
        data: CellGraphData,
        constraint: "ProfileCapacityConstraint",
    ):
        self.data = data
        self.arithmetic = data.arithmetic
        self.modified_cell_symmetry = False
        self._base_index = {cell: idx for idx, cell in enumerate(data.cells)}
        allocation = CellEvidenceAllocation.from_constraint(constraint, data.cells)
        self.evidence_profile_sizes = allocation.evidence_profile_sizes
        self.cells = [
            CellWithEvidenceProfile(data.cells[cell_idx], profile_idx)
            for cell_idx, profile_indices in enumerate(
                allocation.compatible_evidence_profiles_by_cell
            )
            for profile_idx in profile_indices
        ]

        if not self.cells:
            self.i1_ind = []
            self.i2_ind = []
            self.cliques = []
            self.nonind = []
            self.nonind_map = {}
            self.clique_evidence_profile_partitions = {}
            self.evidence_profile_cliques = {}
            self.i1_evidence_profile_partition = [
                [] for _ in self.evidence_profile_sizes
            ]
            return

        i1, i2, nonind = self._find_independent_sets()
        self.i1_ind = i1
        self.i2_ind = i2
        self.cliques, self.nonind = self._build_symmetric_cliques(i2 + nonind)
        self.nonind_map = {
            clique_idx: index for index, clique_idx in enumerate(self.nonind)
        }
        self.clique_evidence_profile_partitions = {}
        evidence_profile_cliques: dict[int, list[int]] = defaultdict(list)
        for clique_idx, clique in enumerate(self.cliques):
            partitions = []
            for profile_idx in range(len(self.evidence_profile_sizes)):
                cell_indices = [
                    cell_idx
                    for cell_idx, cell in enumerate(clique)
                    if cell.evidence_profile_index == profile_idx
                ]
                if cell_indices:
                    partitions.append(cell_indices)
                    evidence_profile_cliques[profile_idx].append(clique_idx)
            self.clique_evidence_profile_partitions[clique_idx] = partitions
        self.evidence_profile_cliques = dict(evidence_profile_cliques)
        self.i1_evidence_profile_partition = [
            [
                cell_idx
                for cell_idx in self.i1_ind
                if self.cells[cell_idx].evidence_profile_index == profile_idx
            ]
            for profile_idx in range(len(self.evidence_profile_sizes))
        ]

    def get_cells(self) -> list[CellWithEvidenceProfile]:
        return self.cells

    def get_cell_weight(self, cell: CellWithEvidenceProfile) -> ArithmeticValue:
        return self.data.cell_weights[self._base_index[cell.base_cell]]

    def get_two_table_weight(
        self,
        cells: tuple[CellWithEvidenceProfile, CellWithEvidenceProfile],
    ) -> ArithmeticValue:
        left = self._base_index[cells[0].base_cell]
        right = self._base_index[cells[1].base_cell]
        return self.data.pair_factors[left][right].total_weight

    def get_all_weights(
        self,
    ) -> tuple[list[ArithmeticValue], list[list[ArithmeticValue]]]:
        return (
            [self.get_cell_weight(cell) for cell in self.cells],
            [
                [self.get_two_table_weight((left, right)) for right in self.cells]
                for left in self.cells
            ],
        )

    def _find_independent_sets(self) -> tuple[list[int], list[int], list[int]]:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.cells)))
        for left in range(len(self.cells)):
            for right in range(left + 1, len(self.cells)):
                if not self.arithmetic.equal(
                    self.get_two_table_weight((self.cells[left], self.cells[right])),
                    self.arithmetic.one(),
                ):
                    graph.add_edge(left, right)
        self_loops = {
            idx
            for idx, cell in enumerate(self.cells)
            if not self.arithmetic.equal(
                self.get_two_table_weight((cell, cell)),
                self.arithmetic.one(),
            )
        }
        candidates = set(graph.nodes) - self_loops
        i1 = (
            set()
            if not candidates
            else set(nx.maximal_independent_set(graph.subgraph(candidates)))
        )
        independent = set(nx.maximal_independent_set(graph, nodes=i1))
        return list(i1), list(independent - i1), list(set(graph.nodes) - independent)

    def _matches(
        self,
        clique: list[CellWithEvidenceProfile],
        other: CellWithEvidenceProfile,
    ) -> bool:
        cell = clique[0]
        if len(clique) > 1:
            relation = self.get_two_table_weight((cell, clique[1]))
            if any(
                not self.arithmetic.equal(
                    relation,
                    self.get_two_table_weight((other, third)),
                )
                for third in clique
            ):
                return False
        return all(
            self.arithmetic.equal(
                self.get_two_table_weight((cell, third)),
                self.get_two_table_weight((other, third)),
            )
            for third in self.cells
            if other != third and third not in clique
        )

    def _build_symmetric_cliques(
        self,
        cell_indices: list[int],
    ) -> tuple[list[list[CellWithEvidenceProfile]], list[int]]:
        remaining = list(cell_indices)
        cliques = []
        clique_indices = []
        while remaining:
            cell_idx = remaining.pop()
            clique = [self.cells[cell_idx]]
            for other_idx in remaining:
                other = self.cells[other_idx]
                if self._matches(clique, other):
                    clique.append(other)
            for other in clique[1:]:
                remaining.remove(self.cells.index(other))
            cliques.append(clique)
            clique_indices.append(len(cliques) - 1)
        return cliques, clique_indices


def build_optimized_cell_graphs(
    formula: Formula,
    weights: Mapping[Predicate, tuple[ArithmeticValue, ArithmeticValue]],
    arithmetic: "ArithmeticContext",
    *,
    modified_cell_symmetry: bool,
    required_unary_preds: frozenset[Predicate] = frozenset(),
    profile_capacity_constraint: "ProfileCapacityConstraint | None" = None,
    cell_formulas: tuple[Formula, ...] | None = None,
):
    """Build base branches, then perform fast-only clique analysis."""

    for data, graph_weight in build_cell_graphs(
        formula,
        weights,
        arithmetic,
        required_unary_preds=required_unary_preds,
        cell_formulas=cell_formulas,
    ):
        analysis = (
            _OptimizedAnalysis(data, modified_cell_symmetry)
            if profile_capacity_constraint is None
            else _EvidenceOptimizedAnalysis(
                data,
                profile_capacity_constraint,
            )
        )
        yield analysis, graph_weight


__all__ = ["CellWithEvidenceProfile", "build_optimized_cell_graphs"]
