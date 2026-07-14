"""Public, solver-independent cardinality-constraint model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum


class Comparator(Enum):
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
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

    def safe_predicate_upper_bounds(self) -> dict[object, int]:
        """Return independently safe count caps implied by the constraints.

        A finite cap is available for ``=``, ``<=`` and ``<`` constraints
        whose normalized predicate coefficients are all non-negative. Mixed
        signs are intentionally ignored because one predicate's high degree
        may then be cancelled by another predicate.
        """

        bounds: dict[object, int] = {}
        for constraint in self.constraints:
            if constraint.comparator in (Comparator.EQ, Comparator.LE):
                total_upper_bound = constraint.rhs
            elif constraint.comparator is Comparator.LT:
                total_upper_bound = constraint.rhs - 1
            else:
                continue
            if total_upper_bound < 0:
                continue

            coefficients: defaultdict[object, int] = defaultdict(int)
            for term in constraint.terms:
                coefficients[term.predicate] += term.coefficient
            if any(coefficient < 0 for coefficient in coefficients.values()):
                continue

            for predicate, coefficient in coefficients.items():
                if coefficient <= 0:
                    continue
                upper_bound = total_upper_bound // coefficient
                current = bounds.get(predicate)
                bounds[predicate] = (
                    upper_bound if current is None else min(current, upper_bound)
                )
        return bounds


def combine_cardinality_constraints(
    *groups: CardinalityConstraints,
) -> CardinalityConstraints:
    """Concatenate constraint groups without changing their meaning or order."""

    return CardinalityConstraints(
        tuple(
            constraint
            for group in groups
            for constraint in group.constraints
        )
    )


__all__ = [
    "CardinalityConstraints",
    "CardinalityTerm",
    "Comparator",
    "LinearCardinalityConstraint",
    "combine_cardinality_constraints",
]
