"""Weight configuration, symbol discovery, and branch weight compilation.

This module does not own arithmetic domains. Callers supply an
``ArithmeticContext`` selected by the preparation pipeline.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Mapping, TypeAlias

from flint import (
    arb,
    arb_poly,
    fmpq,
    fmpq_mpoly,
    fmpq_mpoly_ctx,
    fmpq_poly,
    fmpq_series,
)

from wfomc.arithmetic import (
    ArithmeticBackend as _ArithmeticBackend,
    ArithmeticContext as _ArithmeticContext,
    ArithmeticValue as _ArithmeticValue,
)
from wfomc.options import WeightOptions

# ---------------------------------------------------------------------------
# Type aliases for weight values
# ---------------------------------------------------------------------------

RawWeightValue: TypeAlias = int | float | Fraction | _ArithmeticValue
"""A raw weight value before compilation: numeric or backend-native."""

CompiledWeightValue: TypeAlias = _ArithmeticValue
"""A compiled weight value after passing through an ArithmeticContext."""

RawWeightMapping: TypeAlias = Mapping[object, tuple[RawWeightValue, RawWeightValue]]
"""A raw predicate-to-weight-pair mapping before compilation."""

CompiledWeightMapping: TypeAlias = dict[
    object, tuple[CompiledWeightValue, CompiledWeightValue]
]
"""A compiled predicate-to-weight-pair mapping after compilation."""


def collect_symbolic_weight_variables(weights: Mapping[object, object]) -> tuple[str, ...]:
    """Discover all solver symbols already present in a raw weight mapping."""

    return tuple(sorted(_explicit_symbolic_weight_variables(weights)))


def collect_output_weight_variables(
    weights: Mapping[object, object],
    internal_symbols: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return only user-visible symbols occurring in declared weight values."""

    return tuple(sorted(_explicit_symbolic_weight_variables(weights) - set(internal_symbols)))


def _explicit_symbolic_weight_variables(
    weights: Mapping[object, object],
) -> set[str]:
    values = weights.values()
    variables = set()
    for pair in values:
        for value in pair:
            if isinstance(value, fmpq_mpoly) and not value.is_constant():
                variables.update(value.context().names())
    return variables


def compile_weight_mapping(
    weights: RawWeightMapping,
    arithmetic: _ArithmeticContext,
) -> CompiledWeightMapping:
    """Compile a raw weight mapping through an :class:`ArithmeticContext`.

    Each value is minted via the context's backend-specific coercion so the
    user-selected :class:`WeightOptions` controls every compiled weight. The
    context is built once by the reduced-problem compiler and
    reused for algorithm-input threading, so weight compilation and the
    algorithm's own temporaries share one arithmetic backend.

    Args:
        weights: Raw ``predicate -> (positive, negative)`` weight mapping.
        arithmetic: Resolved :class:`ArithmeticContext` for the branch.

    Returns:
        Compiled ``predicate -> (positive, negative)`` ring elements.
    """
    return {
        predicate: (
            _compile_weight_value(positive, arithmetic),
            _compile_weight_value(negative, arithmetic),
        )
        for predicate, (positive, negative) in weights.items()
    }


def _compile_weight_value(value: object, context: _ArithmeticContext) -> object:
    backend = context.backend
    if backend is _ArithmeticBackend.FMPQ_SERIES:
        compiled = _to_fmpq_series(value, context)
        return context.truncate(compiled)
    if backend is _ArithmeticBackend.FMPQ_POLY:
        compiled = _to_fmpq_poly(value, context)
        return context.truncate(compiled)
    if backend is _ArithmeticBackend.ARB_POLY:
        compiled = _to_arb_poly(value, context)
        return context.truncate(compiled)
    if backend is _ArithmeticBackend.FMPQ_MPOLY:
        compiled = _to_fmpq_mpoly(value, context)
        return context.truncate(compiled)
    return context.coerce(value)


def _to_fmpq_series(value: object, context: _ArithmeticContext) -> fmpq_series:
    return context.coerce(value)


def _to_fmpq_poly(value: object, context: _ArithmeticContext) -> fmpq_poly:
    if isinstance(value, fmpq_poly):
        return value
    if isinstance(value, fmpq_mpoly):
        return _mpoly_to_fmpq_poly(value, context)
    return fmpq_poly([_to_fmpq_scalar(value)])


def _to_arb_poly(value: object, context: _ArithmeticContext) -> arb_poly:
    if isinstance(value, arb_poly):
        return value
    if isinstance(value, fmpq_mpoly):
        return arb_poly([arb(coeff) for coeff in _mpoly_to_fmpq_poly(value, context)])
    if isinstance(value, arb):
        return arb_poly([value])
    if isinstance(value, float):
        return arb_poly([arb(value)])
    return arb_poly([arb(_to_fmpq_scalar(value))])


def _to_fmpq_mpoly(value: object, context: _ArithmeticContext) -> fmpq_mpoly:
    target_ctx = fmpq_mpoly_ctx.get(context.symbolic_variables, "lex")
    if isinstance(value, fmpq_mpoly):
        return value.project_to_context(target_ctx)
    if isinstance(value, fmpq_poly):
        if len(context.symbolic_variables) != 1:
            if value.degree() <= 0:
                return target_ctx.constant(value[0])
            raise ValueError(
                "Cannot infer which symbolic variable a non-constant "
                "fmpq_poly should use in a multivariate context"
            )
        return target_ctx.from_dict(
            {
                (degree,): coefficient
                for degree, coefficient in enumerate(value.coeffs())
                if coefficient
            }
        )
    return target_ctx.constant(_to_fmpq_scalar(value))


def _mpoly_to_fmpq_poly(value: fmpq_mpoly, context: _ArithmeticContext) -> fmpq_poly:
    if len(context.symbolic_variables) != 1:
        raise ValueError(
            f"{context.backend!s} backend requires exactly one symbolic variable"
        )
    variable = context.symbolic_variables[0]
    names = tuple(value.context().names())
    try:
        variable_index = names.index(variable)
    except ValueError as exc:
        if value.is_constant():
            return fmpq_poly([value.leading_coefficient()])
        raise ValueError(
            f"Weight variable context {names!r} does not contain {variable!r}"
        ) from exc

    terms = value.to_dict()
    max_degree = max((monomial[variable_index] for monomial in terms), default=0)
    coeffs = [fmpq(0) for _ in range(max_degree + 1)]
    for monomial, coeff in terms.items():
        other_degree = sum(
            degree for idx, degree in enumerate(monomial) if idx != variable_index
        )
        if other_degree:
            raise ValueError(
                f"{context.backend!s} backend cannot compile multivariate "
                f"weight {value!s}"
            )
        coeffs[monomial[variable_index]] += coeff
    return fmpq_poly(coeffs)


def _to_fmpq_scalar(value: object) -> fmpq:
    if isinstance(value, bool):
        return fmpq(int(value))
    if isinstance(value, int):
        return fmpq(value)
    if isinstance(value, fmpq):
        return value
    if isinstance(value, Fraction):
        return fmpq(int(value.numerator), int(value.denominator))
    if isinstance(value, fmpq_mpoly) and value.is_constant():
        return value.leading_coefficient()
    raise ValueError(f"Unsupported exact weight type: {type(value).__name__!s}")


__all__ = [
    "CompiledWeightMapping",
    "CompiledWeightValue",
    "RawWeightMapping",
    "RawWeightValue",
    "WeightOptions",
    "collect_output_weight_variables",
    "collect_symbolic_weight_variables",
    "compile_weight_mapping",
]
