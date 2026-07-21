from __future__ import annotations

import math

from wfomc.algo.fast.operations import (
    MaterializedOptimizedEvidenceOperations,
    MaterializedOptimizedOperations,
    instantiate_optimized_operations,
)
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _direct_clique_weight(
    count,
    *,
    weights,
    self_relations,
    interaction,
    include_weights,
    arithmetic,
):
    result = arithmetic.zero()
    for allocation in _compositions(count, len(weights)):
        coefficient = math.factorial(count)
        for cell_count in allocation:
            coefficient //= math.factorial(cell_count)
        term = arithmetic.from_int(coefficient)
        cross_pairs = 0
        for cell_index, cell_count in enumerate(allocation):
            if include_weights:
                term = arithmetic.multiply(
                    term,
                    arithmetic.power(weights[cell_index], cell_count),
                )
            term = arithmetic.multiply(
                term,
                arithmetic.power(
                    self_relations[cell_index],
                    math.comb(cell_count, 2),
                ),
            )
            cross_pairs += cell_count * sum(allocation[cell_index + 1 :])
        term = arithmetic.multiply(
            term,
            arithmetic.power(interaction, cross_pairs),
        )
        result = arithmetic.add(result, term)
    return result


def _ordinary_operations(
    *,
    domain_size,
    weights,
    self_relations,
    interaction,
    modified_cell_symmetry,
):
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    cells = tuple(f"cell-{index}" for index in range(len(weights)))
    two_table = {
        (left, right): (
            self_relations[left_index]
            if left_index == right_index
            else interaction
        )
        for left_index, left in enumerate(cells)
        for right_index, right in enumerate(cells)
    }
    return MaterializedOptimizedOperations(
        cliques=[list(cells)],
        nonind=[0],
        nonind_map={0: 0},
        i1_ind=[],
        i2_ind=[],
        domain_size=domain_size,
        modified_cell_symmetry=modified_cell_symmetry,
        arithmetic=arithmetic,
        cell_weight=dict(zip(cells, weights)),
        two_table=two_table,
    )


def test_fast_homogeneous_clique_uses_one_cached_convolution_message() -> None:
    domain_size = 6
    weights = (7,) * 8
    self_relations = (2,) * 8
    interaction = 3
    operations = _ordinary_operations(
        domain_size=domain_size,
        weights=weights,
        self_relations=self_relations,
        interaction=interaction,
        modified_cell_symmetry=False,
    )

    actual = tuple(
        operations.get_J_term(0, count) for count in range(domain_size + 1)
    )
    expected = tuple(
        _direct_clique_weight(
            count,
            weights=weights,
            self_relations=self_relations,
            interaction=interaction,
            include_weights=False,
            arithmetic=operations.arithmetic,
        )
        for count in range(domain_size + 1)
    )

    assert actual == expected
    assert list(operations.symmetric_message_cache) == [0]
    assert operations.d_term_cache == {}


def test_fastv2_groups_equal_local_rows_in_one_convolution_message() -> None:
    domain_size = 5
    weights = (2, 2, 5, 5, 11)
    self_relations = (3, 3, 7, 7, 13)
    interaction = 17
    operations = _ordinary_operations(
        domain_size=domain_size,
        weights=weights,
        self_relations=self_relations,
        interaction=interaction,
        modified_cell_symmetry=True,
    )

    actual = tuple(
        operations.get_J_term(0, count) for count in range(domain_size + 1)
    )
    expected = tuple(
        _direct_clique_weight(
            count,
            weights=weights,
            self_relations=self_relations,
            interaction=interaction,
            include_weights=True,
            arithmetic=operations.arithmetic,
        )
        for count in range(domain_size + 1)
    )

    assert actual == expected
    assert list(operations.symmetric_message_cache) == [0]
    assert operations.d_term_cache == {}


def test_evidence_partition_uses_convolution_message() -> None:
    domain_size = 5
    weights = (2, 3, 5, 7)
    self_relations = (11, 13, 17, 19)
    interaction = 23
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    cells = tuple(f"cell-{index}" for index in range(len(weights)))
    operations = MaterializedOptimizedEvidenceOperations(
        cells=list(cells),
        cliques=[list(cells)],
        nonind=[0],
        nonind_map={0: 0},
        i1_evidence_profile_partition=[],
        clique_evidence_profile_partitions={0: [list(range(len(cells)))]},
        domain_size=domain_size,
        arithmetic=arithmetic,
        cell_weight=dict(zip(cells, weights)),
        two_table={
            (left, right): (
                self_relations[left_index]
                if left_index == right_index
                else interaction
            )
            for left_index, left in enumerate(cells)
            for right_index, right in enumerate(cells)
        },
    )

    actual = tuple(
        operations.get_partitioned_J_term(0, 0, count)
        for count in range(domain_size + 1)
    )
    expected = tuple(
        _direct_clique_weight(
            count,
            weights=weights,
            self_relations=self_relations,
            interaction=interaction,
            include_weights=True,
            arithmetic=arithmetic,
        )
        for count in range(domain_size + 1)
    )

    assert actual == expected
    assert list(operations.partitioned_message_cache) == [(0, 0)]
    assert operations._d_term_cache == {}


def test_instantiation_keeps_symmetric_messages_domain_local() -> None:
    template = _ordinary_operations(
        domain_size=None,
        weights=(2, 2, 2),
        self_relations=(3, 3, 3),
        interaction=5,
        modified_cell_symmetry=False,
    )
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)

    small = instantiate_optimized_operations(
        template,
        arithmetic=arithmetic,
        domain_size=3,
    )
    large = instantiate_optimized_operations(
        template,
        arithmetic=arithmetic,
        domain_size=6,
    )
    small.get_J_term(0, 3)
    large.get_J_term(0, 6)

    assert len(small.symmetric_message_cache[0]) == 4
    assert len(large.symmetric_message_cache[0]) == 7
    assert small.symmetric_message_cache is not large.symmetric_message_cache
