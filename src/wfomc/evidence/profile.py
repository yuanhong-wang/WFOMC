"""Reduced unary-evidence profile data shared with algorithm materializers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from wfomc.fol import Atom, Literal, Not


@dataclass(frozen=True)
class EvidenceProfile:
    """A fixed-size unary literal assignment profile."""

    literals: frozenset[Literal]
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Evidence profile sizes must be positive.")
        object.__setattr__(
            self,
            "literals",
            frozenset(_as_profile_literal(literal) for literal in self.literals),
        )

    def cache_key_parts(self) -> tuple[int, tuple[object, ...]]:
        return (
            self.size,
            tuple(sorted(literal.cache_key_parts() for literal in self.literals)),
        )


@dataclass(frozen=True)
class ProfileCapacityConstraint:
    """Reduction artifact: unary evidence encoded as profile-size capacities."""

    profiles: tuple[EvidenceProfile, ...]
    domain_size: int
    assignment_count: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if self.domain_size < 0:
            raise ValueError("Domain size must be non-negative.")
        if sum(profile.size for profile in self.profiles) != self.domain_size:
            raise ValueError("Evidence profile sizes must sum to the domain size.")
        if not isinstance(self.assignment_count, Fraction):
            object.__setattr__(
                self,
                "assignment_count",
                Fraction(self.assignment_count),
            )

    @property
    def is_empty(self) -> bool:
        return not self.profiles

    def cache_key_parts(self) -> tuple[int, object, tuple[object, ...]]:
        return (
            self.domain_size,
            repr(self.assignment_count),
            tuple(profile.cache_key_parts() for profile in self.profiles),
        )


def _as_profile_literal(value: Literal | Atom | Not) -> Literal:
    if isinstance(value, Literal):
        literal = value
    elif isinstance(value, Atom):
        literal = Literal(value, True)
    elif isinstance(value, Not) and isinstance(value.body, Atom):
        literal = Literal(value.body, False)
    else:
        raise TypeError(f"Unsupported evidence profile literal: {value!r}")
    if literal.predicate.arity != 1:
        raise ValueError("Evidence profile literals must be unary.")
    return literal


__all__ = [
    "EvidenceProfile",
    "ProfileCapacityConstraint",
]
