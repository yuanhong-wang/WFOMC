"""Incremental3 algorithm spec."""

from __future__ import annotations

from collections.abc import Hashable

from wfomc.algo.core import (
    AlgoBranch,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    ExistentialStrategy,
    option_resolver,
)
from wfomc.cell_graph.staging import select_input_variant
from wfomc.problem import Domain
from wfomc.stages import CompiledReducedBranch

from .input import (
    Incremental3InputTemplate,
    Incremental3InputVariant,
    build_input_template,
)
from .solve import solve


def input_template_key(
    branch: AlgoBranch,
    domain: Domain,
) -> Incremental3InputVariant:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Incremental3 requires a compiled reduced branch")
    from .counting_state import build_counting_state_for_normal_form

    counting_state, _unary_masks = build_counting_state_for_normal_form(
        branch.reduced_problem.normal_form,
        domain_size=domain.size,
    )
    return Incremental3InputVariant(
        evidence=select_input_variant(branch, domain),
        counting_state=counting_state,
    )


def build_incremental3_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    _options: AlgoOptions,
) -> Incremental3InputTemplate:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Incremental3 requires a compiled reduced branch")
    if not isinstance(input_key, Incremental3InputVariant):
        raise TypeError("Incremental3 input key has an invalid type")
    return build_input_template(
        branch,
        input_variant=input_key,
    )


SPEC = AlgoSpec(
    name=AlgoName.INCREMENTAL3,
    resolve_options=option_resolver(
        algo=AlgoName.INCREMENTAL3,
        default_unary_evidence=EvidenceStrategy.LIFTED_PROFILES,
        supported_unary_evidence=(
            EvidenceStrategy.CCS,
            EvidenceStrategy.LIFTED_PROFILES,
        ),
        default_existential_strategy=ExistentialStrategy.COUNTING,
        supported_existential_strategies=(
            ExistentialStrategy.COUNTING,
            ExistentialStrategy.SKOLEM,
        ),
        supports_linear_order=True,
        supports_mod_counting=True,
    ),
    solve=solve,
    build_input_template=build_incremental3_input_template,
    input_template_key=input_template_key,
    reduce_counting_quantifiers=False,
)


__all__ = ["SPEC"]
