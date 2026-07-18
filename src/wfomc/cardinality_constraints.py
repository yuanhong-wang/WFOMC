"""Public, solver-independent cardinality-constraint model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Comparator(Enum):
    """Comparison operator for counting quantifiers and cardinality constraints."""

    # Equality.
    EQ = "="
    # Inequality.
    NE = "!="
    # Strict upper bound.
    LT = "<"
    # Inclusive upper bound.
    LE = "<="
    # Strict lower bound.
    GT = ">"
    # Inclusive lower bound.
    GE = ">="
    # Congruence modulo a positive modulus.
    MOD = "mod"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CardinalityTerm:
    """One integer-weighted predicate count in a linear constraint."""

    predicate: object
    coefficient: int = 1


@dataclass(frozen=True)
class LinearCardinalityConstraint:
    """A comparison over a linear combination of predicate cardinalities."""

    terms: tuple[CardinalityTerm, ...]
    comparator: Comparator
    rhs: int
    modulus: int | None = None

    def __post_init__(self) -> None:
        if self.comparator is Comparator.MOD:
            if self.modulus is None or self.modulus <= 0:
                raise ValueError("MOD cardinality constraints require a positive modulus")
        elif self.modulus is not None:
            raise ValueError("Only MOD cardinality constraints accept a modulus")

    def accepts(self, value: int) -> bool:
        """Return whether ``value`` satisfies this comparison."""

        if self.comparator is Comparator.EQ:
            return value == self.rhs
        if self.comparator is Comparator.NE:
            return value != self.rhs
        if self.comparator is Comparator.LT:
            return value < self.rhs
        if self.comparator is Comparator.LE:
            return value <= self.rhs
        if self.comparator is Comparator.GT:
            return value > self.rhs
        if self.comparator is Comparator.GE:
            return value >= self.rhs
        if self.comparator is Comparator.MOD:
            assert self.modulus is not None
            return value % self.modulus == self.rhs % self.modulus
        raise ValueError(f"Unsupported cardinality comparator: {self.comparator}")


@dataclass(frozen=True)
class CardinalityConstraints:
    """The cardinality constraints attached to a problem."""

    constraints: tuple[LinearCardinalityConstraint, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.constraints

    def cache_key_parts(self) -> tuple[object, ...]:
        return tuple(
            (
                tuple(
                    (str(term.predicate), term.coefficient)
                    for term in constraint.terms
                ),
                str(constraint.comparator),
                constraint.rhs,
                constraint.modulus,
            )
            for constraint in self.constraints
        )

__all__ = [
    "CardinalityConstraints",
    "CardinalityTerm",
    "Comparator",
    "LinearCardinalityConstraint",
]
