"""Arithmetic context: the sole channel for WFOMC to create numeric values.

Every weight, algorithm temporary, multinomial coefficient, default weight,
and decode/cardinality factor is produced through an
:class:`ArithmeticContext` whose backend is fixed after branch reduction.
Algorithms never import FLINT types to
mint constants directly; they ask the context for ``zero`` / ``one`` /
``from_int`` / ``from_fraction`` / ``coerce``. This guarantees the
user-selected :class:`wfomc.weights.WeightOptions` controls the entire
arithmetic type stack end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, Iterable, TypeAlias

from flint import (
    arb,
    arb_poly,
    fmpq,
    fmpq_mpoly,
    fmpq_mpoly_ctx,
    fmpq_poly,
    fmpz,
    fmpz_mpoly,
    fmpz_mpoly_ctx,
    fmpz_poly,
)

from wfomc.errors import ArithmeticBackendError

if TYPE_CHECKING:
    from wfomc.weights import WeightOptions


ArithmeticValue: TypeAlias = (
    int
    | float
    | fmpz
    | fmpq
    | arb
    | fmpz_poly
    | fmpq_poly
    | arb_poly
    | fmpz_mpoly
    | fmpq_mpoly
)


class ArithmeticBackend(Enum):
    """Numeric domain used by one prepared branch."""

    # Exact integer scalars.
    FMPZ = "fmpz"
    # Exact rational scalars.
    FMPQ = "fmpq"
    # Native double-precision floating-point scalars.
    FLOAT = "float"
    # Arbitrary-precision real-ball scalars.
    ARB = "arb"
    # Exact univariate polynomials with integer coefficients.
    FMPZ_POLY = "fmpz_poly"
    # Exact univariate polynomials with rational coefficients.
    FMPQ_POLY = "fmpq_poly"
    # Univariate polynomials with arbitrary-precision ball coefficients.
    ARB_POLY = "arb_poly"
    # Exact multivariate polynomials with integer coefficients.
    FMPZ_MPOLY = "fmpz_mpoly"
    # Exact multivariate polynomials with rational coefficients.
    FMPQ_MPOLY = "fmpq_mpoly"

    def __str__(self) -> str:
        return self.value


def choose_arithmetic_backend(
    options: "WeightOptions",
    *,
    symbolic_variables: tuple[str, ...],
) -> ArithmeticBackend:
    """Select the numeric domain from precision and solver symbols."""

    symbol_count = len(symbolic_variables)
    if options.precision == "exact":
        if symbol_count == 0:
            return ArithmeticBackend.FMPQ
        if options.exact_symbolic_backend == "fmpq_poly":
            if symbol_count != 1:
                raise ArithmeticBackendError(
                    "Exact 'fmpq_poly' arithmetic supports exactly one "
                    "symbolic variable"
                )
            return ArithmeticBackend.FMPQ_POLY
        return ArithmeticBackend.FMPQ_MPOLY
    if options.rounded_backend == "float":
        if symbol_count == 0:
            return ArithmeticBackend.FLOAT
        raise ArithmeticBackendError(
            "Rounded 'float' arithmetic does not support symbolic weights"
        )
    if symbol_count == 0:
        return ArithmeticBackend.ARB
    if symbol_count == 1:
        return ArithmeticBackend.ARB_POLY
    raise ArithmeticBackendError(
        "Rounded 'arb' arithmetic does not support multiple symbols; "
        "python-flint has no arb_mpoly"
    )


@dataclass(frozen=True)
class ArithmeticContext:
    """Backend-bound factory for the numeric values used by WFOMC.

    One context is built per reduced-problem branch after backend and symbol
    selection, then threaded into every algorithm input. All numeric
    construction goes through it so that ``WeightOptions`` governs every
    value rather than a patchwork of hard-coded ``fmpq(1)`` / ``1`` /
    ``Fraction(...)`` literals scattered through the solvers.
    """

    backend: ArithmeticBackend
    symbolic_variables: tuple[str, ...] = ()
    output_symbols: tuple[str, ...] = ()
    degree_limits: tuple[tuple[str, int], ...] = ()
    _direct_fmpq: bool = field(init=False, repr=False, compare=False)
    _cached_zero: ArithmeticValue = field(init=False, repr=False, compare=False)
    _cached_one: ArithmeticValue = field(init=False, repr=False, compare=False)
    _cached_mpoly_context: object | None = field(
        init=False, repr=False, compare=False
    )
    _symbol_indices: dict[str, int] = field(init=False, repr=False, compare=False)
    _univariate_degree_limit: int | None = field(
        init=False, repr=False, compare=False
    )
    _indexed_degree_limits: tuple[tuple[int, int], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        normalized = tuple(sorted(self.degree_limits))
        if len({name for name, _limit in normalized}) != len(normalized):
            raise ValueError("ArithmeticContext degree limits must be unique")
        unknown = set(name for name, _limit in normalized) - set(
            self.symbolic_variables
        )
        if unknown:
            raise ValueError(
                f"ArithmeticContext degree limits reference unknown symbols: "
                f"{sorted(unknown)!r}"
            )
        if any(limit < 0 for _name, limit in normalized):
            raise ValueError("ArithmeticContext degree limits must be non-negative")
        object.__setattr__(self, "degree_limits", normalized)
        symbol_indices = {
            name: index for index, name in enumerate(self.symbolic_variables)
        }
        object.__setattr__(self, "_symbol_indices", symbol_indices)
        object.__setattr__(
            self,
            "_indexed_degree_limits",
            tuple((symbol_indices[name], limit) for name, limit in normalized),
        )
        object.__setattr__(
            self,
            "_univariate_degree_limit",
            (
                dict(normalized).get(self.symbolic_variables[0])
                if len(self.symbolic_variables) == 1
                else None
            ),
        )

        names = self.symbolic_variables or ("_w",)
        mpoly_context = None
        if self.backend is ArithmeticBackend.FMPQ_MPOLY:
            mpoly_context = fmpq_mpoly_ctx.get(list(names), "lex")
        elif self.backend is ArithmeticBackend.FMPZ_MPOLY:
            mpoly_context = fmpz_mpoly_ctx.get(list(names), "lex")
        object.__setattr__(self, "_cached_mpoly_context", mpoly_context)

        direct_fmpq = self.backend is ArithmeticBackend.FMPQ and not normalized
        object.__setattr__(self, "_direct_fmpq", direct_fmpq)
        object.__setattr__(self, "_cached_zero", self._from_fraction(0, 1))
        object.__setattr__(self, "_cached_one", self._from_fraction(1, 1))

    # -- public numeric factory API --------------------------------------

    def zero(self):
        """Additive identity for this backend."""
        return self._cached_zero

    def one(self):
        """Multiplicative identity for this backend."""
        return self._cached_one

    def neg_one(self):
        """Negative one for this backend."""
        return self.from_int(-1)

    def from_int(self, value: int):
        """Create a backend value from a Python int."""
        return self._from_fraction(int(value), 1)

    def from_fraction(self, numerator: int, denominator: int = 1):
        """Create a backend value from an explicit numerator/denominator."""
        numerator = int(numerator)
        denominator = int(denominator)
        if denominator == 0:
            raise ZeroDivisionError("ArithmeticContext.from_fraction: zero denominator")
        return self._from_fraction(numerator, denominator)

    def coerce(self, value):
        """Coerce a Python/FLINT numeric value into this backend's type.

        Accepts ``bool``/``int``/``Fraction``/``float`` and the
        FLINT scalar types (``fmpz``/``fmpq``/``arb``). Already-compatible
        FLINT polynomial/mpoly values are passed through unchanged so that
        symbolic weight expressions keep their structure.
        """
        if isinstance(value, bool):
            return self.from_int(int(value))
        if isinstance(value, int):
            return self.from_int(value)
        if isinstance(value, Fraction):
            return self._from_fraction(value.numerator, value.denominator)
        if isinstance(value, float):
            return self._coerce_float(value)
        if isinstance(value, fmpz):
            return self.from_int(int(value))
        if isinstance(value, fmpq):
            return self._from_fraction(int(value.p), int(value.q))
        if isinstance(value, arb):
            return self._coerce_arb(value)
        if isinstance(value, fmpz_poly) and self.backend is ArithmeticBackend.FMPZ_POLY:
            return value
        if isinstance(value, fmpq_poly) and self.backend is ArithmeticBackend.FMPQ_POLY:
            return self.truncate(value)
        if isinstance(value, arb_poly) and self.backend is ArithmeticBackend.ARB_POLY:
            return value
        if (
            isinstance(value, fmpq_mpoly)
            and self.backend is ArithmeticBackend.FMPQ_MPOLY
        ):
            return self.truncate(
                value.project_to_context(self._mpoly_ctx(fmpq_mpoly_ctx))
            )
        if (
            isinstance(value, fmpq_mpoly)
            and self.backend is ArithmeticBackend.FMPQ_POLY
        ):
            if value.context().nvars() != 1:
                raise ArithmeticBackendError(
                    "Cannot coerce a multivariate fmpq_mpoly into fmpq_poly"
                )
            coefficients = [fmpq(0)] * (max(value.degrees(), default=0) + 1)
            for monomial, coefficient in value.to_dict().items():
                coefficients[monomial[0]] += coefficient
            return self.truncate(fmpq_poly(coefficients))
        raise ArithmeticBackendError(
            f"Cannot coerce {type(value).__name__!s} into backend {self.backend!s}"
        )

    def symbol(self, name: str):
        """Return one solver-ring generator declared by this branch plan."""

        if name not in self.symbolic_variables:
            raise ValueError(f"Unknown arithmetic symbol: {name!r}")
        index = self._symbol_indices[name]
        if self.backend is ArithmeticBackend.FMPQ_MPOLY:
            return self.truncate(self._mpoly_ctx(fmpq_mpoly_ctx).gen(index))
        if self.backend is ArithmeticBackend.FMPZ_MPOLY:
            return self._mpoly_ctx(fmpz_mpoly_ctx).gen(index)
        if index != 0:
            raise ArithmeticBackendError(
                f"Backend {self.backend!s} cannot expose symbol {name!r}"
            )
        if self.backend is ArithmeticBackend.FMPQ_POLY:
            return self.truncate(fmpq_poly([0, 1]))
        if self.backend is ArithmeticBackend.FMPZ_POLY:
            return fmpz_poly([0, 1])
        if self.backend is ArithmeticBackend.ARB_POLY:
            return arb_poly([0, 1])
        raise ArithmeticBackendError(
            f"Backend {self.backend!s} does not support symbolic values"
        )

    def is_zero(self, value) -> bool:
        """Return whether *value* is the additive identity.

        Comparing with the Python literal avoids constructing a backend zero,
        which matters in polynomial hot loops.
        """

        return value == 0

    def is_one(self, value) -> bool:
        """Return whether *value* is the multiplicative identity."""

        return value == 1

    def multiply(self, left, right):
        """Multiply two backend values with exact zero/one fast paths."""

        if self._direct_fmpq:
            return left * right

        if self.is_zero(left):
            return left
        if self.is_zero(right):
            return right
        if self.is_one(left):
            return self.truncate(right)
        if self.is_one(right):
            return self.truncate(left)
        return self.truncate(left * right)

    def power(self, base, exponent: int):
        """Raise a backend value to an integer power with identity fast paths."""

        if self._direct_fmpq:
            return self.one() if exponent == 0 else base**exponent

        if exponent == 0:
            return self.one()
        if exponent == 1:
            return self.truncate(base)
        if self.is_one(base):
            return base
        if exponent > 0 and self.is_zero(base):
            return base
        if (
            exponent > 1
            and self.degree_limits
            and isinstance(base, (fmpq_poly, fmpq_mpoly))
        ):
            result = self.one()
            factor = base
            remaining = exponent
            while remaining:
                if remaining & 1:
                    result = self.multiply(result, factor)
                remaining >>= 1
                if remaining:
                    factor = self.multiply(factor, factor)
            return result
        return self.truncate(base**exponent)

    def add(self, left, right):
        """Add two values that already satisfy the truncation invariant."""

        if self._direct_fmpq:
            return left + right

        if self.is_zero(left):
            return right
        if self.is_zero(right):
            return left
        # Addition cannot increase any monomial degree, so bounded operands
        # remain bounded without another degree scan.
        return left + right

    def add_product(self, accumulator, left, right):
        """Return ``accumulator + left * right`` in the active backend.

        Plain scalar rationals need neither identity checks nor degree
        truncation, so their hot path performs the two FLINT operations
        directly.  Symbolic and rounded backends retain the regular public
        operations and therefore all existing truncation semantics.
        """

        if self._direct_fmpq:
            return accumulator + left * right
        # Multiplication may exceed a degree limit, while adding the bounded
        # accumulator cannot.  Fuse both operations and truncate only once.
        return self.truncate(accumulator + left * right)

    def truncate(self, value):
        """Discard monomials above proven-safe internal degree limits."""

        if not self.degree_limits:
            return value
        if isinstance(value, fmpq_poly):
            limit = self._univariate_degree_limit
            if limit is None or value.degree() <= limit:
                return value
            return value.truncate(limit + 1)
        if isinstance(value, fmpq_mpoly):
            indexed_limits = self._indexed_degree_limits
            if not indexed_limits:
                return value
            max_degrees = value.degrees()
            if all(max_degrees[index] <= limit for index, limit in indexed_limits):
                return value
            terms = {
                monomial: coefficient
                for monomial, coefficient in value.to_dict().items()
                if all(
                    monomial[index] <= limit
                    for index, limit in indexed_limits
                )
            }
            return value.context().from_dict(terms)
        return value

    def sum(self, values: Iterable):
        """Sum an iterable of coercible values in this backend's type."""
        total = self.zero()
        for value in values:
            total = self.add(total, self.coerce(value))
        return total

    def project_to_output(self, value):
        """Remove internal solver symbols after every decoder has run.

        Cardinality reductions temporarily extend an ``fmpq_mpoly`` ring with
        marker variables.  Decoder correction factors must still run in that
        extended ring, so projection belongs at the engine boundary rather
        than inside the cardinality decoder.
        """

        if isinstance(value, fmpq_poly):
            if not self.output_symbols and value.degree() <= 0:
                return value[0]
            return value
        if not isinstance(value, fmpq_mpoly):
            return value
        projected = value.project_to_context(
            fmpq_mpoly_ctx.get(list(self.output_symbols), "lex")
        )
        return projected.leading_coefficient() if projected.is_constant() else projected

    # -- backend dispatch ------------------------------------------------

    def _from_fraction(self, numerator: int, denominator: int):
        backend = self.backend
        if backend is ArithmeticBackend.FMPZ:
            self._require_unit_denominator(backend, numerator, denominator)
            return fmpz(numerator)
        if backend is ArithmeticBackend.FMPQ:
            return fmpq(numerator, denominator)
        if backend is ArithmeticBackend.FLOAT:
            return numerator / denominator
        if backend is ArithmeticBackend.ARB:
            return arb(numerator) / arb(denominator)
        if backend is ArithmeticBackend.FMPZ_POLY:
            self._require_unit_denominator(backend, numerator, denominator)
            return fmpz_poly([numerator])
        if backend is ArithmeticBackend.FMPQ_POLY:
            return fmpq_poly([fmpq(numerator, denominator)])
        if backend is ArithmeticBackend.ARB_POLY:
            return arb_poly([arb(numerator) / arb(denominator)])
        if backend is ArithmeticBackend.FMPZ_MPOLY:
            return self._mpoly_ctx(fmpz_mpoly_ctx).constant(numerator)
        if backend is ArithmeticBackend.FMPQ_MPOLY:
            return self._mpoly_ctx(fmpq_mpoly_ctx).constant(
                fmpq(numerator, denominator)
            )
        raise ArithmeticBackendError(
            f"ArithmeticContext cannot create values for backend {backend!s}"
        )

    def _coerce_float(self, value: float):
        if self.backend is ArithmeticBackend.FLOAT:
            return value
        if self.backend is ArithmeticBackend.ARB:
            return arb(value)
        if self.backend is ArithmeticBackend.ARB_POLY:
            return arb_poly([arb(value)])
        raise ArithmeticBackendError(
            f"Cannot coerce float into non-float backend {self.backend!s}"
        )

    def _coerce_arb(self, value):
        if self.backend is ArithmeticBackend.ARB:
            return value
        if self.backend is ArithmeticBackend.ARB_POLY:
            return arb_poly([value])
        raise ArithmeticBackendError(
            f"Cannot coerce arb into non-arb backend {self.backend!s}"
        )

    def _mpoly_ctx(self, ctx_factory):
        if self._cached_mpoly_context is not None:
            if (
                self.backend is ArithmeticBackend.FMPQ_MPOLY
                and ctx_factory is fmpq_mpoly_ctx
            ) or (
                self.backend is ArithmeticBackend.FMPZ_MPOLY
                and ctx_factory is fmpz_mpoly_ctx
            ):
                return self._cached_mpoly_context
        names = self.symbolic_variables or ("_w",)
        return ctx_factory.get(list(names), "lex")

    @staticmethod
    def _require_unit_denominator(
        backend: ArithmeticBackend, numerator: int, denominator: int
    ) -> None:
        if denominator != 1:
            raise ValueError(
                f"{backend!s} backend cannot represent the non-integer "
                f"fraction {numerator}/{denominator}"
            )


__all__ = [
    "ArithmeticBackend",
    "ArithmeticContext",
    "ArithmeticValue",
    "choose_arithmetic_backend",
]
