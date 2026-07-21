"""FastV2 staged algorithm spec."""

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
from wfomc.algo.fast.input import FastInputTemplate
from wfomc.algo.fast.solve import solve
from wfomc.algo.fast.spec import (
    build_fast_input_template,
    input_template_key,
)

def build_fastv2_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    options: AlgoOptions,
) -> FastInputTemplate:
    return build_fast_input_template(
        branch,
        input_key,
        options,
        modified_cell_symmetry=True,
    )


SPEC = AlgoSpec(
    name=AlgoName.FASTV2,
    resolve_options=option_resolver(
        algo=AlgoName.FASTV2,
        default_unary_evidence=EvidenceStrategy.LIFTED_PROFILES,
        supported_unary_evidence=(
            EvidenceStrategy.CCS,
            EvidenceStrategy.LIFTED_PROFILES,
        ),
    ),
    solve=solve,
    build_input_template=build_fastv2_input_template,
    input_template_key=input_template_key,
)


__all__ = ["SPEC"]
