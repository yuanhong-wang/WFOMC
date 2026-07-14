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

from dataclasses import dataclass
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

    FMPZ = "fmpz"
    FMPQ = "fmpq"
    FLOAT = "float"
    ARB = "arb"
    FMPZ_POLY = "fmpz_poly"
    FMPQ_POLY = "fmpq_poly"
    ARB_POLY = "arb_poly"
    FMPZ_MPOLY = "fmpz_mpoly"
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
        if symbol_count == 1:
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

    # -- public numeric factory API --------------------------------------

    def zero(self):
        """Additive identity for this backend."""
        return self.from_int(0)

    def one(self):
        """Multiplicative identity for this backend."""
        return self.from_int(1)

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
            return value
        if isinstance(value, arb_poly) and self.backend is ArithmeticBackend.ARB_POLY:
            return value
        if (
            isinstance(value, fmpq_mpoly)
            and self.backend is ArithmeticBackend.FMPQ_MPOLY
        ):
            return value.project_to_context(self._mpoly_ctx(fmpq_mpoly_ctx))
        raise ArithmeticBackendError(
            f"Cannot coerce {type(value).__name__!s} into backend {self.backend!s}"
        )

    def symbol(self, name: str):
        """Return one solver-ring generator declared by this branch plan."""

        if name not in self.symbolic_variables:
            raise ValueError(f"Unknown arithmetic symbol: {name!r}")
        index = self.symbolic_variables.index(name)
        if self.backend is ArithmeticBackend.FMPQ_MPOLY:
            return self._mpoly_ctx(fmpq_mpoly_ctx).gen(index)
        if self.backend is ArithmeticBackend.FMPZ_MPOLY:
            return self._mpoly_ctx(fmpz_mpoly_ctx).gen(index)
        if index != 0:
            raise ArithmeticBackendError(
                f"Backend {self.backend!s} cannot expose symbol {name!r}"
            )
        if self.backend is ArithmeticBackend.FMPQ_POLY:
            return fmpq_poly([0, 1])
        if self.backend is ArithmeticBackend.FMPZ_POLY:
            return fmpz_poly([0, 1])
        if self.backend is ArithmeticBackend.ARB_POLY:
            return arb_poly([0, 1])
        raise ArithmeticBackendError(
            f"Backend {self.backend!s} does not support symbolic values"
        )

    def is_zero(self, value) -> bool:
        return value == self.zero()

    def sum(self, values: Iterable):
        """Sum an iterable of coercible values in this backend's type."""
        total = self.zero()
        for value in values:
            total = total + self.coerce(value)
        return total

    def project_to_output(self, value):
        """Remove internal solver symbols after every decoder has run.

        Cardinality reductions temporarily extend an ``fmpq_mpoly`` ring with
        marker variables.  Decoder correction factors must still run in that
        extended ring, so projection belongs at the engine boundary rather
        than inside the cardinality decoder.
        """

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
