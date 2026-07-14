"""Tail-signature algorithm adapter shell."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from .input import TailSignatureInput
from wfomc.errors import ExternalToolError, UnsupportedFeatureError
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


def solve(
    algo_input: TailSignatureInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    engine_factory = _engine_factory(runtime)
    if engine_factory is None:
        raise ExternalToolError(
            "tail-signature algorithm adapter exists, but the engine is not integrated yet"
        )
    if algo_input.unary_profile_capacities is not None:
        raise UnsupportedFeatureError(
            "tail-signature algorithm does not support unary evidence profiles yet"
        )
    if algo_input.polynomial_context is not None:
        raise UnsupportedFeatureError(
            "tail-signature algorithm does not support cardinality polynomial constraints yet"
        )

    value_converter = _value_converter(runtime, engine_factory)
    result = algo_input.arithmetic.zero()
    for component in algo_input.components:
        engine = engine_factory(
            _as_lists(component.w_tables, value_converter),
            _as_lists(component.r_matrix, value_converter),
        )
        component_result = engine.compute(
            algo_input.domain_size,
            **(algo_input.engine_options or {}),
        )
        result = algo_input.arithmetic.add(
            result,
            algo_input.arithmetic.multiply(
                component.graph_weight,
                _coerce_scalar_result(
                    component_result,
                    algo_input.arithmetic,
                ),
            ),
        )

    return WFOMCResult(result)


def _engine_factory(runtime: "RuntimeContext | None") -> Callable[..., object] | None:
    if runtime is None:
        return None
    return runtime.options.tail_signature_engine_factory


def _value_converter(
    runtime: "RuntimeContext | None",
    engine_factory: Callable[..., object],
) -> Callable[[object], object] | None:
    configured = (
        runtime.options.tail_signature_value_converter if runtime is not None else None
    )
    if configured is not None:
        return configured
    module_name = getattr(engine_factory, "__module__", None)
    module = sys.modules.get(module_name) if module_name else None
    mpq = getattr(module, "mpq", None)
    if not callable(mpq):
        return None

    def convert(value: object) -> object:
        try:
            return mpq(value)
        except (TypeError, ValueError):
            return mpq(str(value))

    return convert


def _as_lists(
    table: object,
    value_converter: Callable[[object], object] | None = None,
) -> list[list[object]]:
    if value_converter is None:
        return [list(row) for row in table]
    return [[value_converter(value) for value in row] for row in table]


def _coerce_scalar_result(value: object, arithmetic) -> object:
    from flint import fmpq

    try:
        return arithmetic.coerce(value)
    except (TypeError, ValueError):
        try:
            return arithmetic.coerce(fmpq(str(value)))
        except (TypeError, ValueError):
            raise TypeError(
                f"tail-signature engine returned unsupported numeric value "
                f"{type(value).__name__}"
            ) from None


__all__ = ["solve"]
