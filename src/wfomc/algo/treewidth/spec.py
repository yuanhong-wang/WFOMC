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
from wfomc.problem import Domain, Problem


def _reduce(
    _problem: Problem,
    *,
    options: AlgoOptions,
) -> None:
    raise UnsupportedFeatureError("bounded-treewidth reduction is not yet implemented")


def compile_branches(
    problem: Problem,
    _options: AlgoOptions,
) -> tuple[object, ...]:
    return (problem,)


def build_input_template(
    branch: object,
    _input_variant: object,
    options: AlgoOptions,
) -> object:
    if not isinstance(branch, Problem):
        raise TypeError("Bounded-treewidth compiled branch has an invalid type")
    _reduce(branch, options=options)
    raise AssertionError("unreachable")


def instantiate_branch(
    _branch: object,
    _input_template: object,
    _domain: Domain,
    _options: AlgoOptions,
) -> PreparedBranch:
    raise RuntimeError("bounded-treewidth input construction did not fail")


SPEC = AlgoSpec(
    name=AlgoName.BOUNDED_TREEWIDTH,
    resolve_options=option_resolver(
        algo=AlgoName.BOUNDED_TREEWIDTH,
        default_unary_evidence=EvidenceStrategy.TREE_DECOMPOSITION_FACTORS,
        supported_unary_evidence=(EvidenceStrategy.TREE_DECOMPOSITION_FACTORS,),
    ),
    solve=solve,
    compile_branches=compile_branches,
    build_input_template=build_input_template,
    instantiate_branch=instantiate_branch,
    maturity=AlgoMaturity.UNAVAILABLE,
    external_requirements=("bounded-treewidth decomposition solver",),
)


__all__ = ["SPEC"]
