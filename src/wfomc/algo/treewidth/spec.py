"""Bounded-treewidth algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    PreparedBranch,
    option_resolver,
)
from .solve import solve
from wfomc.errors import UnsupportedFeatureError
from wfomc.problem import Problem


def _reduce(
    _problem: Problem,
    *,
    options: AlgoOptions,
) -> None:
    raise UnsupportedFeatureError("bounded-treewidth reduction is not yet implemented")


def prepare(problem: Problem, options: AlgoOptions) -> tuple[PreparedBranch, ...]:
    _reduce(problem, options=options)
    return ()


SPEC = AlgoSpec(
    name=AlgoName.BOUNDED_TREEWIDTH,
    resolve_options=option_resolver(
        algo=AlgoName.BOUNDED_TREEWIDTH,
        default_unary_evidence=EvidenceStrategy.TREE_DECOMPOSITION_FACTORS,
        supported_unary_evidence=(EvidenceStrategy.TREE_DECOMPOSITION_FACTORS,),
    ),
    prepare=prepare,
    solve=solve,
    maturity=AlgoMaturity.UNAVAILABLE,
    external_requirements=("bounded-treewidth decomposition solver",),
)


__all__ = ["SPEC"]
