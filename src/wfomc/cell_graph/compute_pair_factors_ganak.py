"""Exact projected pair factors from one Ganak polynomial WMC call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flint import fmpq, fmpq_mpoly, fmpq_mpoly_ctx, fmpq_poly

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext, ArithmeticValue
from wfomc.errors import GanakError
from wfomc.fol.cnf import TseitinCNF
from wfomc.ganak import ganak_count


Factor = dict[int, ArithmeticValue]
WeightPair = tuple[ArithmeticValue, ArithmeticValue]
_EXACT_GANAK_BACKENDS = frozenset(
    (
        ArithmeticBackend.FMPQ,
        ArithmeticBackend.FMPQ_POLY,
        ArithmeticBackend.FMPQ_MPOLY,
    )
)


def compute_factor_with_ganak(
    cnf: TseitinCNF,
    literal_weights: Sequence[WeightPair],
    projection: Mapping[int, int],
    arithmetic: ArithmeticContext,
    *,
    timeout: float,
) -> Factor | None:
    """Return one exact projected factor, or decline an unsupported backend.

    Each projected positive literal receives a private polynomial marker. A
    single Ganak call then returns a polynomial whose marker monomials are the
    projected truth masks and whose coefficients are their exact weights.
    """

    if arithmetic.backend not in _EXACT_GANAK_BACKENDS:
        return None
    _validate_inputs(cnf, literal_weights, projection)

    marker_count = max(projection.values(), default=-1) + 1
    base_names = arithmetic.symbolic_variables
    if not base_names and marker_count == 0:
        weights = {
            variable: (
                _as_rational(literal_weights[variable][0]),
                _as_rational(literal_weights[variable][1]),
            )
            for variable in range(1, cnf.n_vars + 1)
        }
        count = ganak_count(
            cnf.n_vars,
            cnf.clauses,
            weights,
            symbolic=False,
            timeout=timeout,
        )
        value = arithmetic.coerce(count)
        return {} if arithmetic.is_zero(value) else {0: value}

    marker_names = _fresh_marker_names(base_names, marker_count)
    polynomial_context = fmpq_mpoly_ctx.get(base_names + marker_names)
    base_count = len(base_names)
    ganak_weights = {}
    for variable in range(1, cnf.n_vars + 1):
        positive, negative = literal_weights[variable]
        positive_poly = _embed_weight(
            positive,
            polynomial_context,
            base_count,
            marker_count,
        )
        negative_poly = _embed_weight(
            negative,
            polynomial_context,
            base_count,
            marker_count,
        )
        marker_bit = projection.get(variable)
        if marker_bit is not None:
            positive_poly *= polynomial_context.gen(base_count + marker_bit)
        ganak_weights[variable] = (positive_poly, negative_poly)

    count = ganak_count(
        cnf.n_vars,
        cnf.clauses,
        ganak_weights,
        symbolic=True,
        npolyvars=polynomial_context.nvars(),
        poly_ctx=polynomial_context,
        timeout=timeout,
    )
    polynomial = _as_polynomial(
        count,
        polynomial_context,
    )
    return _extract_factor(
        polynomial,
        base_names,
        marker_count,
        arithmetic,
    )


def _validate_inputs(
    cnf: TseitinCNF,
    literal_weights: Sequence[WeightPair],
    projection: Mapping[int, int],
) -> None:
    if len(literal_weights) <= cnf.n_vars:
        raise ValueError("literal_weights must contain an entry for every CNF variable")
    bits = tuple(projection.values())
    if any(variable < 1 or variable > cnf.n_vars for variable in projection):
        raise ValueError("projection contains a variable outside the CNF")
    if any(bit < 0 for bit in bits) or len(set(bits)) != len(bits):
        raise ValueError("projection bits must be distinct non-negative integers")


def _as_rational(value: ArithmeticValue) -> fmpq:
    if isinstance(value, fmpq):
        return value
    raise GanakError(
        f"Ganak rational mode received unsupported weight {type(value).__name__}"
    )


def _fresh_marker_names(
    base_names: tuple[str, ...],
    marker_count: int,
) -> tuple[str, ...]:
    used = set(base_names)
    result = []
    for index in range(marker_count):
        name = f"__wfomc_pair_marker_{index}"
        while name in used:
            name += "_"
        used.add(name)
        result.append(name)
    return tuple(result)


def _embed_weight(
    value: ArithmeticValue,
    target_context: fmpq_mpoly_ctx,
    base_count: int,
    marker_count: int,
) -> fmpq_mpoly:
    if isinstance(value, fmpq):
        return target_context.constant(value)
    if isinstance(value, fmpq_poly):
        if base_count != 1:
            raise GanakError("univariate weight requires exactly one arithmetic symbol")
        monomials = {
            (degree,) + (0,) * marker_count: coefficient
            for degree, coefficient in enumerate(value.coeffs())
            if coefficient
        }
        return target_context.from_dict(monomials)
    if isinstance(value, fmpq_mpoly):
        monomials = {
            tuple(monomial) + (0,) * marker_count: coefficient
            for monomial, coefficient in zip(value.monoms(), value.coeffs())
            if coefficient
        }
        if any(len(monomial) != base_count for monomial in value.monoms()):
            raise GanakError("multivariate weight does not match arithmetic symbols")
        return target_context.from_dict(monomials)
    raise GanakError(
        f"Ganak polynomial mode received unsupported weight {type(value).__name__}"
    )


def _as_polynomial(
    value,
    context: fmpq_mpoly_ctx,
) -> fmpq_mpoly:
    if isinstance(value, fmpq_mpoly):
        return value.project_to_context(context)
    if isinstance(value, fmpq):
        return context.constant(value)
    raise GanakError(
        "Ganak symbolic mode returned an unsupported value: "
        f"{type(value).__name__}; expected fmpq_mpoly"
    )


def _extract_factor(
    polynomial: fmpq_mpoly,
    base_names: tuple[str, ...],
    marker_count: int,
    arithmetic: ArithmeticContext,
) -> Factor:
    base_count = len(base_names)
    result: Factor = {}
    for monomial, coefficient in zip(polynomial.monoms(), polynomial.coeffs()):
        base_exponents = tuple(monomial[:base_count])
        marker_exponents = tuple(monomial[base_count:])
        if len(marker_exponents) != marker_count:
            raise GanakError("Ganak polynomial has an unexpected marker dimension")
        if any(exponent not in (0, 1) for exponent in marker_exponents):
            raise GanakError("Ganak polynomial contains a non-Boolean marker exponent")

        mask = sum(
            1 << bit
            for bit, exponent in enumerate(marker_exponents)
            if exponent
        )
        value = arithmetic.from_fraction(int(coefficient.p), int(coefficient.q))
        for name, exponent in zip(base_names, base_exponents):
            if exponent:
                value = arithmetic.multiply(
                    value,
                    arithmetic.power(arithmetic.symbol(name), exponent),
                )
        result[mask] = arithmetic.add(
            result.get(mask, arithmetic.zero()),
            value,
        )

    return {
        mask: value
        for mask, value in result.items()
        if not arithmetic.is_zero(value)
    }


__all__ = ["compute_factor_with_ganak"]
