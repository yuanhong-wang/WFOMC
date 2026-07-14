"""Direct fast WFOMC algorithms over optimized cell-graph inputs."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING

from .input import (
    OptimizedCellGraphComponent,
    OptimizedCellGraphInput,
)
from wfomc.multinomial import MultinomialCoefficients
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


def solve(
    algo_input: OptimizedCellGraphInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    if not isinstance(algo_input, OptimizedCellGraphInput):
        raise TypeError("fast algorithm expects an OptimizedCellGraphInput")

    MultinomialCoefficients.setup(algo_input.domain_size)
    arithmetic = algo_input.arithmetic
    result = arithmetic.zero()
    for component in algo_input.components:
        if component.evidence_profile_sizes is not None:
            subtotal = _solve_component_with_evidence(component, arithmetic)
        else:
            subtotal = _solve_component_without_evidence(
                component,
                algo_input.domain_size,
                arithmetic,
            )
        result = arithmetic.add(
            result,
            arithmetic.multiply(component.graph_weight, subtotal),
        )

    return WFOMCResult(result)


def _solve_component_without_evidence(
    component: OptimizedCellGraphComponent,
    domain_size: int,
    arithmetic,
) -> object:
    from wfomc.multinomial import MultinomialCoefficients, multinomial_less_than

    operations = _operations(component)
    cliques = component.cliques
    nonind = component.non_independent
    i2_ind = component.i2_independent
    nonind_map = component.non_independent_index

    subtotal = arithmetic.zero()
    for partition in multinomial_less_than(len(nonind), domain_size):
        if sum(partition) < domain_size:
            mu = tuple(partition) + (domain_size - sum(partition),)
        else:
            mu = tuple(partition)
        coefficient = arithmetic.from_int(MultinomialCoefficients.coef(mu))
        body = arithmetic.one()

        for i, clique1 in enumerate(cliques):
            for j, clique2 in enumerate(cliques):
                if i in nonind and j in nonind and i < j:
                    body = arithmetic.multiply(
                        body,
                        arithmetic.power(
                            operations.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ),
                            partition[nonind_map[i]] * partition[nonind_map[j]],
                        ),
                    )

        for clique_idx in nonind:
            body = arithmetic.multiply(
                body,
                operations.get_J_term(
                    clique_idx,
                    partition[nonind_map[clique_idx]],
                ),
            )
            if not component.modified_cell_symmetry:
                body = arithmetic.multiply(
                    body,
                    arithmetic.power(
                        operations.get_cell_weight(cliques[clique_idx][0]),
                        partition[nonind_map[clique_idx]],
                    ),
                )

        operations.setup_term_cache()
        multiplier = operations.get_term(len(i2_ind), 0, partition)
        subtotal = arithmetic.add(
            subtotal,
            arithmetic.multiply(
                arithmetic.multiply(coefficient, multiplier),
                body,
            ),
        )
    return subtotal


def _solve_component_with_evidence(
    component: OptimizedCellGraphComponent,
    arithmetic,
) -> object:
    from wfomc.multinomial import MultinomialCoefficients, multinomial_less_than

    operations = _operations(component)
    cliques = component.cliques
    nonind = component.non_independent
    nonind_map = component.non_independent_index
    evidence_profile_sizes = component.evidence_profile_sizes or ()
    evidence_profile_cliques = component.evidence_profile_cliques or {}

    subtotal = arithmetic.zero()
    for configs in product(
        *(
            list(
                multinomial_less_than(
                    len(evidence_profile_cliques.get(evidence_profile_idx, ())),
                    constrained_num,
                )
            )
            for evidence_profile_idx, constrained_num in enumerate(
                evidence_profile_sizes
            )
        )
    ):
        coefficient = arithmetic.one()
        remainings = []
        overall_config = [0 for _ in range(len(cliques))]
        clique_configs = defaultdict(list)
        for evidence_profile_idx, (constrained_num, config) in enumerate(
            zip(evidence_profile_sizes, configs)
        ):
            remainings.append(constrained_num - sum(config))
            mu = tuple(config) + (constrained_num - sum(config),)
            coefficient = arithmetic.multiply(
                coefficient,
                arithmetic.from_int(MultinomialCoefficients.coef(mu)),
            )
            for count, clique_idx in zip(
                config,
                evidence_profile_cliques.get(evidence_profile_idx, ()),
            ):
                overall_config[clique_idx] += count
                clique_configs[clique_idx].append(count)

        body = operations.get_i1_weight(tuple(remainings), tuple(overall_config))

        for i, clique1 in enumerate(cliques):
            for j, clique2 in enumerate(cliques):
                if i in nonind and j in nonind and i < j:
                    body = arithmetic.multiply(
                        body,
                        arithmetic.power(
                            operations.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ),
                            overall_config[nonind_map[i]]
                            * overall_config[nonind_map[j]],
                        ),
                    )

        for clique_idx in nonind:
            body = arithmetic.multiply(
                body,
                operations.get_J_term(
                    clique_idx,
                    tuple(clique_configs[nonind_map[clique_idx]]),
                ),
            )
        subtotal = arithmetic.add(
            subtotal,
            arithmetic.multiply(coefficient, body),
        )
    return subtotal


def _operations(component: OptimizedCellGraphComponent) -> object:
    if component.weight_operations is None:
        raise RuntimeError("optimized cell-graph operations are not materialized")
    return component.weight_operations


__all__ = ["solve"]
