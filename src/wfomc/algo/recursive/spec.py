"""Recursive algorithm spec."""

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

from .input import RecursiveInputTemplate, build_input_template
from .solve import solve


def input_template_key(
    branch: AlgoBranch,
    domain: Domain,
) -> CellGraphInputVariant:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Recursive requires a compiled reduced branch")
    return select_input_variant(branch, domain)


def build_recursive_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    _options: AlgoOptions,
) -> RecursiveInputTemplate:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Recursive requires a compiled reduced branch")
    return build_input_template(
        branch,
        input_variant=input_key,
    )


SPEC = AlgoSpec(
    name=AlgoName.RECURSIVE,
    resolve_options=option_resolver(
        algo=AlgoName.RECURSIVE,
        default_unary_evidence=EvidenceStrategy.CCS,
        supported_unary_evidence=(EvidenceStrategy.CCS,),
        supports_linear_order=True,
    ),
    solve=solve,
    build_input_template=build_recursive_input_template,
    input_template_key=input_template_key,
)


__all__ = ["SPEC"]
