"""Propositional algorithm over explicit ground-CNF input."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .input import GroundCNFInput
from wfomc.arithmetic import ArithmeticBackend
from wfomc.errors import ArithmeticBackendError
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


def solve(
    algo_input: GroundCNFInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    from .counting import propositional_ground_wfomc

    if not isinstance(algo_input, GroundCNFInput):
        raise TypeError("propositional algorithm expects a GroundCNFInput")
    if algo_input.arithmetic.backend not in {
        ArithmeticBackend.FMPQ,
        ArithmeticBackend.FMPQ_POLY,
        ArithmeticBackend.FMPQ_MPOLY,
    }:
        raise ArithmeticBackendError(
            "propositional Ganak execution currently supports only exact "
            "fmpq/fmpq_poly/fmpq_mpoly arithmetic"
        )

    ganak_path = (
        runtime.options.propositional_ganak_path if runtime is not None else None
    )
    result = propositional_ground_wfomc(
        algo_input.cnf,
        algo_input.literal_weights,
        ganak_path=ganak_path,
        linear_order_encoding=algo_input.linear_order_encoding,
        leq_present=algo_input.leq_present,
        arithmetic=algo_input.arithmetic,
    )
    return WFOMCResult(result)


__all__ = ["solve"]
