"""Standard algorithm spec."""

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
from wfomc.cell_graph.staging import CellGraphInputVariant, select_input_variant
from wfomc.problem import Domain
from wfomc.stages import CompiledReducedBranch

from .input import StandardInputTemplate, build_input_template
from .solve import solve


def input_template_key(
    branch: AlgoBranch,
    domain: Domain,
) -> CellGraphInputVariant:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Standard requires a compiled reduced branch")
    return select_input_variant(branch, domain)


def build_standard_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    _options: AlgoOptions,
) -> StandardInputTemplate:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Standard requires a compiled reduced branch")
    return build_input_template(
        branch,
        input_variant=input_key,
    )


SPEC = AlgoSpec(
    name=AlgoName.STANDARD,
    resolve_options=option_resolver(
        algo=AlgoName.STANDARD,
        default_unary_evidence=EvidenceStrategy.LIFTED_PROFILES,
        supported_unary_evidence=(
            EvidenceStrategy.CCS,
            EvidenceStrategy.LIFTED_PROFILES,
        ),
    ),
    solve=solve,
    build_input_template=build_standard_input_template,
    input_template_key=input_template_key,
)


__all__ = ["SPEC"]
