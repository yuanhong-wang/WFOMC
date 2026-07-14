"""FastV2 algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    PreparedBranch,
    compile_reduced_problem,
    option_resolver,
    reduce_unary_evidence_for_options,
)
from wfomc.algo.fast.input import build_input
from wfomc.algo.fast.solve import solve
from wfomc.problem import Problem
from wfomc.reduction import (
    reduce_cardinality_constraints,
    reduce_counting_quantifiers,
    reduce_existential_quantifiers,
    apply_reductions,
)


REDUCTIONS = (
    reduce_unary_evidence_for_options,
    reduce_counting_quantifiers,
    reduce_existential_quantifiers,
    reduce_cardinality_constraints,
)


def prepare(problem: Problem, options: AlgoOptions) -> tuple[PreparedBranch, ...]:
    reduced = apply_reductions(problem, REDUCTIONS, options)
    prepared = []
    for branch in reduced.problems:
        compiled, _features = compile_reduced_problem(branch.problem, options)
        algo_input = build_input(
            compiled,
            domain_size=len(compiled.domain),
            modified_cell_symmetry=True,
            options=options,
        )
        prepared.append(PreparedBranch(branch.problem, algo_input, branch.decoder))
    return tuple(prepared)


SPEC = AlgoSpec(
    name=AlgoName.FASTV2,
    resolve_options=option_resolver(
        algo=AlgoName.FASTV2,
        default_unary_evidence=EvidenceStrategy.CCS,
        supported_unary_evidence=(EvidenceStrategy.CCS,),
    ),
    prepare=prepare,
    solve=solve,
)


__all__ = ["SPEC"]
