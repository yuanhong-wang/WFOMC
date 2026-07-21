"""Fast staged algorithm spec."""

from __future__ import annotations

from collections.abc import Hashable

from wfomc.algo.core import (
    AlgoBranch,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    option_resolver,
)
from wfomc.cell_graph.staging import select_input_variant
from wfomc.problem import Domain
from wfomc.stages import CompiledReducedBranch

from .input import FastInputTemplate, build_input_template
from .solve import solve


def input_template_key(
    branch: AlgoBranch,
    domain: Domain,
) -> tuple[int, ...] | bool | None:
    """Select the Fast cell-graph shape needed by one concrete domain."""

    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Fast requires a compiled reduced branch")
    return select_input_variant(branch, domain)


def build_fast_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    _options: AlgoOptions,
    *,
    modified_cell_symmetry: bool = False,
) -> FastInputTemplate:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Fast requires a compiled reduced branch")
    return build_input_template(
        branch,
        input_variant=input_key,
        modified_cell_symmetry=modified_cell_symmetry,
    )


SPEC = AlgoSpec(
    name=AlgoName.FAST,
    resolve_options=option_resolver(
        algo=AlgoName.FAST,
        default_unary_evidence=EvidenceStrategy.CCS,
        supported_unary_evidence=(
            EvidenceStrategy.CCS,
            EvidenceStrategy.LIFTED_PROFILES,
        ),
    ),
    solve=solve,
    build_input_template=build_fast_input_template,
    input_template_key=input_template_key,
)


__all__ = ["SPEC"]
