"""Boundary-Profile algorithm specification."""

from __future__ import annotations

from collections.abc import Hashable

from wfomc.algo.core import (
    AlgoBranch,
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    option_resolver,
)
from wfomc.cell_graph.staging import select_input_variant
from wfomc.problem import Domain
from wfomc.stages import CompiledReducedBranch

from .input import BoundaryProfileInputTemplate, build_input_template
from .solve import solve


def input_template_key(
    branch: AlgoBranch,
    domain: Domain,
):
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Boundary-Profile requires a compiled reduced branch")
    return select_input_variant(branch, domain)


def build_boundary_profile_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    options: AlgoOptions,
) -> BoundaryProfileInputTemplate:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Boundary-Profile requires a compiled reduced branch")
    return build_input_template(
        branch,
        input_variant=input_key,
        options=options.boundary_profile_options,
    )


SPEC = AlgoSpec(
    name=AlgoName.BOUNDARY_PROFILE,
    resolve_options=option_resolver(
        algo=AlgoName.BOUNDARY_PROFILE,
        default_unary_evidence=EvidenceStrategy.CCS,
        supported_unary_evidence=(EvidenceStrategy.CCS,),
    ),
    solve=solve,
    build_input_template=build_boundary_profile_input_template,
    input_template_key=input_template_key,
    maturity=AlgoMaturity.BETA,
)


__all__ = ["SPEC"]
