"""Ground-CNF weighted model counting for the propositional algorithm."""

from __future__ import annotations

import logging
from typing import Optional, Union

from flint import fmpq_mpoly, fmpq_mpoly_ctx, fmpq_poly, fmpq_series

from wfomc.arithmetic import ArithmeticContext, ArithmeticValue
from wfomc.fol.grounding import (
    LinearOrderEncoding,
    resolve_linear_order_encoding,
)


logger = logging.getLogger(__name__)


def _max_var_id(
    clauses: list[frozenset[int]] | tuple[frozenset[int], ...],
    weights: dict[int, tuple[ArithmeticValue, ArithmeticValue]],
) -> int:
    max_clause_var = max(
        (abs(lit) for clause in clauses for lit in clause),
        default=0,
    )
    return max(max_clause_var, max(weights, default=0))


def _align_ganak_weights(
    weights: dict[int, tuple[ArithmeticValue, ArithmeticValue]],
    arithmetic: ArithmeticContext,
) -> tuple[
    dict[int, tuple[ArithmeticValue, ArithmeticValue]],
    bool,
    int,
    object | None,
]:
    """Normalize literal weights for ganak.

    Rational weights are passed through unchanged. Polynomial weights are
    projected into one shared FLINT context because both ganak serialization
    and FLINT arithmetic require a common context.
    """
    poly_weights = [
        w
        for pos, neg in weights.values()
        for w in (pos, neg)
        if isinstance(w, (fmpq_poly, fmpq_series, fmpq_mpoly))
    ]
    if not poly_weights:
        return weights, False, 0, None

    names = sorted(
        {
            name
            for polynomial in poly_weights
            if isinstance(polynomial, fmpq_mpoly)
            for name in polynomial.context().names()
        }
        | {
            name
            for polynomial in poly_weights
            if isinstance(polynomial, (fmpq_poly, fmpq_series))
            for name in arithmetic.symbolic_variables
        }
    )
    aligned_context = fmpq_mpoly_ctx.get(names, "lex")

    def align(value: ArithmeticValue) -> ArithmeticValue:
        if isinstance(value, fmpq_mpoly):
            return value.project_to_context(aligned_context)
        if isinstance(value, (fmpq_poly, fmpq_series)):
            if aligned_context.nvars() != 1:
                raise ValueError(
                    "univariate Ganak weights require one symbolic variable"
                )
            return aligned_context.from_dict(
                {
                    (degree,): coefficient
                    for degree, coefficient in enumerate(value.coeffs())
                    if coefficient
                }
            )
        return value

    poly_ctx = aligned_context
    aligned_weights = {
        vid: (align(pos), align(neg))
        for vid, (pos, neg) in weights.items()
    }
    return aligned_weights, True, poly_ctx.nvars(), poly_ctx


def propositional_ground_wfomc(
    clauses: list[frozenset[int]] | tuple[frozenset[int], ...],
    weights: dict[int, tuple[ArithmeticValue, ArithmeticValue]],
    *,
    ganak_path: str = None,
    linear_order_encoding: Optional[Union[LinearOrderEncoding, str]] = None,
    leq_present: Optional[bool] = None,
    arithmetic,
) -> ArithmeticValue:
    """Count an already-ground weighted CNF and return the raw count."""
    encoding = resolve_linear_order_encoding(linear_order_encoding)
    leq = bool(leq_present)
    clauses = tuple(frozenset(clause) for clause in clauses)

    if any(len(clause) == 0 for clause in clauses):
        logger.debug("Ground CNF is unsatisfiable; propositional WFOMC is zero")
        raw = arithmetic.zero()
    else:
        weights, symbolic, npolyvars, poly_ctx = _align_ganak_weights(
            dict(weights),
            arithmetic,
        )
        n_vars = _max_var_id(clauses, weights)
        logger.info(
            "Propositional ground WFOMC: vars=%d clauses=%d weights=%s encoding=%s",
            n_vars,
            len(clauses),
            "polynomial" if symbolic else "rational",
            f"{encoding}" + (" (active)" if leq else " (no-op)"),
        )
        from wfomc.ganak import ganak_count

        raw = ganak_count(
            n_vars,
            clauses,
            weights,
            symbolic=symbolic,
            npolyvars=npolyvars,
            poly_ctx=poly_ctx,
            ganak_path=ganak_path,
        )

    return arithmetic.coerce(raw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


__all__ = ["propositional_ground_wfomc"]
