"""Shared staging helpers for cell-graph algorithm inputs."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace
from typing import TYPE_CHECKING

from wfomc.cell_graph.evidence import (
    materialize_cell_evidence,
    profile_cell_formulas,
)
from wfomc.problem import Domain
from wfomc.stages import (
    CompiledBranchInstance,
    CompiledReducedBranch,
)

if TYPE_CHECKING:
    from wfomc.cell_graph.data import CellGraphComponent, PairWeightMatrix
    from wfomc.evidence.profile import ProfileCapacityConstraint
    from wfomc.fol.syntax import Formula


CellGraphInputVariant = tuple[int, ...] | bool | None


def select_input_variant(
    compiled: CompiledReducedBranch,
    domain: Domain,
) -> CellGraphInputVariant:
    """Select the profile-dependent cell-graph shape for one domain."""

    reduced = compiled.reduced_problem
    profile = reduced.profile_constraint
    if profile is not None:
        return profile.active_profile_indices(domain.size)
    unmarked_size = reduced.ccs_unmarked_size
    if unmarked_size is None:
        return None
    concrete_size = unmarked_size.evaluate(domain.size)
    if concrete_size.denominator != 1 or concrete_size < 0:
        raise ValueError("Unary CCS unmarked size must be a non-negative integer")
    return concrete_size == 0


def select_cell_graph_structure(
    compiled: CompiledReducedBranch,
    input_variant: Hashable,
) -> tuple["ProfileCapacityConstraint | None", tuple["Formula", ...] | None]:
    """Select only the profile structure needed to enumerate cells."""

    reduced = compiled.reduced_problem
    profile = reduced.profile_constraint
    structural_profile = None
    cell_formulas = None
    if profile is not None:
        if not isinstance(input_variant, tuple) or not all(
            isinstance(index, int) for index in input_variant
        ):
            raise TypeError("Profile input variant must be a tuple of indices")
        structural_profile = profile.structural_constraint(input_variant)
        cell_formulas = profile_cell_formulas(structural_profile)
    elif reduced.ccs_unmarked_size is not None:
        if not isinstance(input_variant, bool):
            raise TypeError("CCS input variant must select open or closed")
        if input_variant:
            from wfomc.fol import X, disjunction

            cell_formulas = (
                disjunction(
                    *(predicate(X) for predicate in reduced.ccs_profile_markers)
                ),
            )
    elif input_variant is not None:
        raise TypeError("Input variant is invalid without unary profiles")

    return structural_profile, cell_formulas


def rebind_matrix(
    matrix: "PairWeightMatrix",
    concrete: CompiledBranchInstance,
) -> "PairWeightMatrix":
    """Copy one weight matrix into a concrete branch arithmetic context."""

    return tuple(
        tuple(concrete.arithmetic.coerce(value) for value in row) for row in matrix
    )


def rebind_component(
    component: "CellGraphComponent",
    concrete: CompiledBranchInstance,
    *,
    include_evidence: bool,
) -> "CellGraphComponent":
    """Rebind shared graph values and optional profile allocation."""

    return replace(
        component,
        cell_weights=tuple(
            concrete.arithmetic.coerce(value) for value in component.cell_weights
        ),
        pair_weights=rebind_matrix(component.pair_weights, concrete),
        graph_weight=concrete.arithmetic.coerce(component.graph_weight),
        cell_evidence_allocation=(
            materialize_cell_evidence(
                concrete.profile_capacity_constraint,
                component.cells,
            )
            if include_evidence
            else None
        ),
    )


__all__ = [
    "CellGraphInputVariant",
    "rebind_component",
    "rebind_matrix",
    "select_cell_graph_structure",
    "select_input_variant",
]
