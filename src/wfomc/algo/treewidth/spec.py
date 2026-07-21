"""Bounded-treewidth algorithm spec."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Never

from wfomc.algo.core import (
    AlgoBranch,
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    option_resolver,
)
from wfomc.errors import UnsupportedFeatureError

from .solve import solve


def build_input_template(
    _branch: AlgoBranch,
    _input_key: Hashable,
    _options: AlgoOptions,
) -> Never:
    raise UnsupportedFeatureError(
        "bounded-treewidth reduction is not yet implemented"
    )


SPEC = AlgoSpec(
    name=AlgoName.BOUNDED_TREEWIDTH,
    resolve_options=option_resolver(
        algo=AlgoName.BOUNDED_TREEWIDTH,
        default_unary_evidence=EvidenceStrategy.TREE_DECOMPOSITION_FACTORS,
        supported_unary_evidence=(EvidenceStrategy.TREE_DECOMPOSITION_FACTORS,),
    ),
    solve=solve,
    build_input_template=build_input_template,
    uses_reduction=False,
    maturity=AlgoMaturity.UNAVAILABLE,
)


__all__ = ["SPEC"]
