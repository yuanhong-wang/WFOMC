"""Table-only optimized weight engines consumed by fast solvers."""

from __future__ import annotations

import functools

from wfomc.arithmetic import ArithmeticValue
from wfomc.multinomial import MultinomialCoefficients
from .weights import (
    optimized_J_term,
    optimized_d_term,
    optimized_term,
)


class MaterializedOptimizedOperations:
    """Weight recursion for a non-evidence optimized component."""

    def __init__(
        self,
        *,
        cliques,
        nonind,
        nonind_map,
        i1_ind,
        i2_ind,
        domain_size,
        modified_cell_symmetry,
        arithmetic,
        cell_weight,
        two_table,
    ):
        self.cliques = cliques
        self.nonind = nonind
        self.nonind_map = nonind_map
        self.i1_ind = i1_ind
        self.i2_ind = i2_ind
        self.domain_size = domain_size
        self.modified_cell_symmetry = modified_cell_symmetry
        self.arithmetic = arithmetic
        self._cell_weight = cell_weight
        self._two_table = two_table
        self.term_cache: dict = {}
        self.j_term_cache: dict = {}
        self.d_term_cache: dict = {}

    def get_cell_weight(self, cell) -> ArithmeticValue:
        return self._cell_weight[cell]

    def get_two_table_weight(self, cells) -> ArithmeticValue:
        return self._two_table[cells]

    def setup_term_cache(self):
        self.term_cache = dict()

    def get_term(
        self,
        iv: int,
        bign: int,
        partition: tuple[int, ...],
    ) -> ArithmeticValue:
        return optimized_term(self, iv, bign, partition)

    def get_J_term(self, clique_idx: int, nhat: int) -> ArithmeticValue:
        return optimized_J_term(self, clique_idx, nhat)

    def get_d_term(self, clique_idx: int, n: int, cur: int = 0) -> ArithmeticValue:
        return optimized_d_term(self, clique_idx, n, cur)


class MaterializedOptimizedEvidenceOperations:
    """Weight recursion for an evidence-aware optimized component."""

    def __init__(
        self,
        *,
        cells,
        cliques,
        nonind,
        nonind_map,
        i1_evidence_profile_partition,
        clique_evidence_profile_partitions,
        cell_weight,
        two_table,
        arithmetic,
    ):
        self.cells = cells
        self.cliques = cliques
        self.nonind = nonind
        self.nonind_map = nonind_map
        self.i1_evidence_profile_partition = i1_evidence_profile_partition
        self.clique_evidence_profile_partitions = clique_evidence_profile_partitions
        self._cell_weight = cell_weight
        self._two_table = two_table
        self.arithmetic = arithmetic

    def get_cell_weight(self, cell) -> ArithmeticValue:
        return self._cell_weight[cell]

    def get_two_table_weight(self, cells) -> ArithmeticValue:
        return self._two_table[cells]

    def get_i1_weight(
        self, i1_config: tuple[int, ...], config: tuple[int, ...]
    ) -> ArithmeticValue:
        result = self.arithmetic.one()
        for i1_indices, count in zip(self.i1_evidence_profile_partition, i1_config):
            if not i1_indices and count > 0:
                return self.arithmetic.zero()
            accum = self.arithmetic.zero()
            for cell_idx in i1_indices:
                term = self.get_cell_weight(self.cells[cell_idx])
                for clique_idx in self.nonind:
                    term = self.arithmetic.multiply(
                        term,
                        self.arithmetic.power(
                            self.get_two_table_weight(
                                (
                                    self.cliques[clique_idx][0],
                                    self.cells[cell_idx],
                                )
                            ),
                            config[self.nonind_map[clique_idx]],
                        ),
                    )
                accum = self.arithmetic.add(accum, term)
            result = self.arithmetic.multiply(
                result,
                self.arithmetic.power(accum, count),
            )
        return result

    @functools.lru_cache(maxsize=None)
    def get_J_term(
        self,
        clique_idx: int,
        clique_config: tuple[int, ...],
    ) -> ArithmeticValue:
        result = self.arithmetic.one()
        clique = self.cliques[clique_idx]
        evidence_profile_groups = self.clique_evidence_profile_partitions[clique_idx]
        if len(evidence_profile_groups) == 1:
            return self.get_partitioned_J_term(clique_idx, 0, clique_config[0])

        relation = self.get_two_table_weight((clique[0], clique[1]))
        cross_pairs = sum(
            left_count * right_count
            for left_idx, left_count in enumerate(clique_config)
            for right_idx, right_count in enumerate(clique_config)
            if left_idx < right_idx
        )
        result = self.arithmetic.multiply(
            result,
            self.arithmetic.power(relation, cross_pairs),
        )
        for partition_idx in range(len(evidence_profile_groups)):
            result = self.arithmetic.multiply(
                result,
                self.get_partitioned_J_term(
                    clique_idx,
                    partition_idx,
                    clique_config[partition_idx],
                ),
            )
        return result

    @functools.lru_cache(maxsize=None)
    def get_partitioned_J_term(
        self,
        clique_idx: int,
        partition_idx: int,
        count: int,
    ) -> ArithmeticValue:
        cell_indices = self.clique_evidence_profile_partitions[clique_idx][
            partition_idx
        ]
        clique = self.cliques[clique_idx]
        if len(cell_indices) == 1:
            cell = clique[cell_indices[0]]
            return self.arithmetic.multiply(
                self.arithmetic.power(
                    self.get_two_table_weight((cell, cell)),
                    MultinomialCoefficients.comb(count, 2),
                ),
                self.arithmetic.power(self.get_cell_weight(cell), count),
            )
        return self.get_d_term(clique_idx, count, partition_idx)

    @functools.lru_cache(maxsize=None)
    def get_d_term(
        self,
        clique_idx: int,
        count: int,
        partition_idx: int,
        cur: int = 0,
    ) -> ArithmeticValue:
        cell_indices = self.clique_evidence_profile_partitions[clique_idx][
            partition_idx
        ]
        cell_index = cell_indices[cur]
        clique = self.cliques[clique_idx]
        cell = clique[cell_index]
        relation = self.get_two_table_weight((clique[0], clique[1]))
        self_relation = self.get_two_table_weight((cell, cell))
        weight = self.get_cell_weight(cell)

        if cur == len(cell_indices) - 1:
            return self.arithmetic.multiply(
                self.arithmetic.power(weight, count),
                self.arithmetic.power(
                    self_relation,
                    MultinomialCoefficients.comb(count, 2),
                ),
            )

        result = self.arithmetic.zero()
        for cell_count in range(count + 1):
            term = self.arithmetic.from_int(
                MultinomialCoefficients.comb(count, cell_count)
            )
            term = self.arithmetic.multiply(
                term,
                self.arithmetic.power(weight, cell_count),
            )
            term = self.arithmetic.multiply(
                term,
                self.arithmetic.power(
                    self_relation,
                    MultinomialCoefficients.comb(cell_count, 2),
                ),
            )
            term = self.arithmetic.multiply(
                term,
                self.arithmetic.power(
                    relation,
                    cell_count * (count - cell_count),
                ),
            )
            term = self.arithmetic.multiply(
                term,
                self.get_d_term(
                    clique_idx,
                    count - cell_count,
                    partition_idx,
                    cur + 1,
                ),
            )
            result = self.arithmetic.add(result, term)
        return result


def _weight_tables(graph) -> tuple[dict, dict]:
    cells = list(graph.get_cells())
    cell_weight = {cell: graph.get_cell_weight(cell) for cell in cells}
    two_table = {
        (a, b): graph.get_two_table_weight((a, b)) for a in cells for b in cells
    }
    return cell_weight, two_table


OptimizedOperations = (
    MaterializedOptimizedOperations | MaterializedOptimizedEvidenceOperations
)


def materialize_optimized_operations(graph) -> OptimizedOperations:
    """Copy temporary clique analysis into the solver weight engine."""

    cell_weight, two_table = _weight_tables(graph)
    if graph.i1_evidence_profile_partition is not None:
        return MaterializedOptimizedEvidenceOperations(
            cells=list(graph.cells),
            cliques=list(graph.cliques),
            nonind=list(graph.nonind),
            nonind_map=dict(graph.nonind_map),
            i1_evidence_profile_partition=list(graph.i1_evidence_profile_partition),
            clique_evidence_profile_partitions=dict(
                graph.clique_evidence_profile_partitions
            ),
            cell_weight=cell_weight,
            two_table=two_table,
            arithmetic=graph.arithmetic,
        )
    return MaterializedOptimizedOperations(
        cliques=list(graph.cliques),
        nonind=list(graph.nonind),
        nonind_map=dict(graph.nonind_map),
        i1_ind=list(graph.i1_ind),
        i2_ind=list(graph.i2_ind),
        domain_size=graph.domain_size,
        modified_cell_symmetry=graph.modified_cell_symmetry,
        cell_weight=cell_weight,
        two_table=two_table,
        arithmetic=graph.arithmetic,
    )


def instantiate_optimized_operations(
    operations: OptimizedOperations,
    *,
    arithmetic,
    domain_size: int,
) -> OptimizedOperations:
    """Rebind static operation tables and allocate fresh per-domain caches."""

    cell_weight = {
        cell: arithmetic.coerce(weight)
        for cell, weight in operations._cell_weight.items()
    }
    two_table = {
        cells: arithmetic.coerce(weight)
        for cells, weight in operations._two_table.items()
    }
    if isinstance(operations, MaterializedOptimizedEvidenceOperations):
        return MaterializedOptimizedEvidenceOperations(
            cells=list(operations.cells),
            cliques=list(operations.cliques),
            nonind=list(operations.nonind),
            nonind_map=dict(operations.nonind_map),
            i1_evidence_profile_partition=list(
                operations.i1_evidence_profile_partition
            ),
            clique_evidence_profile_partitions=dict(
                operations.clique_evidence_profile_partitions
            ),
            cell_weight=cell_weight,
            two_table=two_table,
            arithmetic=arithmetic,
        )
    return MaterializedOptimizedOperations(
        cliques=list(operations.cliques),
        nonind=list(operations.nonind),
        nonind_map=dict(operations.nonind_map),
        i1_ind=list(operations.i1_ind),
        i2_ind=list(operations.i2_ind),
        domain_size=domain_size,
        modified_cell_symmetry=operations.modified_cell_symmetry,
        cell_weight=cell_weight,
        two_table=two_table,
        arithmetic=arithmetic,
    )


__all__ = [
    "MaterializedOptimizedOperations",
    "MaterializedOptimizedEvidenceOperations",
    "materialize_optimized_operations",
    "instantiate_optimized_operations",
    "OptimizedOperations",
]
