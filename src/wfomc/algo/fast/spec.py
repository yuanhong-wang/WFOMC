"""Fast staged algorithm spec."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.algo.core import (
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    PreparedBranch,
    option_resolver,
)
from wfomc.engine.compilation import (
    CompiledReducedProblem,
    compile_reduced_problem,
    instantiate_reduced_problem,
)
from wfomc.problem import Domain, Problem
from wfomc.reduction import reduce_problem
from wfomc.cell_graph.staging import select_input_variant

from .input import (
    FastInputTemplate,
    build_input_template,
    instantiate_input_template,
)
from .solve import solve


@dataclass(frozen=True)
class CompiledFastBranch:
    """Reusable logical/numeric branch and static Fast variant."""

    numeric: CompiledReducedProblem
    modified_cell_symmetry: bool


def compile_fast_branches(
    problem: Problem,
    options: AlgoOptions,
    *,
    modified_cell_symmetry: bool,
) -> tuple[object, ...]:
    branches = []
    for reduced in reduce_problem(problem, options):
        numeric = compile_reduced_problem(reduced, options)
        branches.append(
            CompiledFastBranch(
                numeric=numeric,
                modified_cell_symmetry=modified_cell_symmetry,
            )
        )
    return tuple(branches)


def compile_branches(
    problem: Problem,
    options: AlgoOptions,
) -> tuple[object, ...]:
    return compile_fast_branches(
        problem,
        options,
        modified_cell_symmetry=False,
    )


def instantiate_branch(
    branch: object,
    input_template: object,
    domain: Domain,
    options: AlgoOptions,
) -> PreparedBranch | None:
    if not isinstance(branch, CompiledFastBranch):
        raise TypeError("Fast compiled branch has an invalid type")
    instantiated = instantiate_reduced_problem(branch.numeric, domain)
    if instantiated is None:
        return None
    concrete, decoder = instantiated
    if not isinstance(input_template, FastInputTemplate):
        raise TypeError("Fast input template has an invalid type")
    algo_input = instantiate_input_template(
        input_template,
        concrete,
        options=options,
    )
    return PreparedBranch(
        branch.numeric.reduced_problem,
        algo_input,
        decoder,
    )


def branch_applies(branch: object, domain: Domain) -> bool:
    if not isinstance(branch, CompiledFastBranch):
        raise TypeError("Fast compiled branch has an invalid type")
    return branch.numeric.reduced_problem.applies(domain)


def input_template_variant(
    branch: object,
    domain: Domain,
) -> tuple[int, ...] | bool | None:
    """Select the Fast cell-graph shape needed by one concrete domain."""

    if not isinstance(branch, CompiledFastBranch):
        raise TypeError("Fast compiled branch has an invalid type")
    return select_input_variant(branch.numeric, domain)


def build_fast_input_template(
    branch: object,
    input_variant: object,
    options: AlgoOptions,
) -> FastInputTemplate:
    if not isinstance(branch, CompiledFastBranch):
        raise TypeError("Fast compiled branch has an invalid type")
    return build_input_template(
        branch.numeric,
        input_variant=input_variant,
        modified_cell_symmetry=branch.modified_cell_symmetry,
        options=options,
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
    compile_branches=compile_branches,
    branch_applies=branch_applies,
    input_template_variant=input_template_variant,
    build_input_template=build_fast_input_template,
    instantiate_branch=instantiate_branch,
)


__all__ = ["SPEC"]
