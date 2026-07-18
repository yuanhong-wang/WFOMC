"""FastV2 staged algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    option_resolver,
)
from wfomc.algo.fast.solve import solve
from wfomc.algo.fast.spec import (
    branch_applies,
    build_fast_input_template,
    compile_fast_branches,
    input_template_variant,
    instantiate_branch,
)
from wfomc.problem import Problem


def compile_branches(
    problem: Problem,
    options: AlgoOptions,
) -> tuple[object, ...]:
    return compile_fast_branches(
        problem,
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
    compile_branches=compile_branches,
    branch_applies=branch_applies,
    input_template_variant=input_template_variant,
    build_input_template=build_fast_input_template,
    instantiate_branch=instantiate_branch,
)


__all__ = ["SPEC"]
