"""Immutable public evidence input data."""

from __future__ import annotations

from dataclasses import dataclass
from wfomc.fol import Atom, FOLContext, Literal


@dataclass(frozen=True)
class GroundUnaryLiteral:
    predicate: object
    constant: object
    positive: bool = True

    def to_profile_literal(self) -> Literal:
        return _profile_literal(self.predicate, self.positive)

    def to_ground_literal(self) -> Literal:
        return Literal(_atom(self.predicate, self.constant), self.positive)

    def cache_key_parts(self) -> tuple[object, str, bool]:
        return (_predicate_key(self.predicate), str(self.constant), self.positive)


@dataclass(frozen=True)
class UnaryEvidence:
    literals: tuple[GroundUnaryLiteral, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.literals

    def cache_key_parts(self) -> tuple[object, ...]:
        return tuple(sorted(literal.cache_key_parts() for literal in self.literals))


@dataclass(frozen=True)
class GroundBinaryLiteral:
    predicate: object
    left: object
    right: object
    positive: bool = True

    def to_ground_literal(self) -> Literal:
        return Literal(_atom(self.predicate, self.left, self.right), self.positive)

    def cache_key_parts(self) -> tuple[object, str, str, bool]:
        return (
            _predicate_key(self.predicate),
            str(self.left),
            str(self.right),
            self.positive,
        )


@dataclass(frozen=True)
class BinaryEvidence:
    literals: tuple[GroundBinaryLiteral, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.literals

    def cache_key_parts(self) -> tuple[object, ...]:
        return tuple(sorted(literal.cache_key_parts() for literal in self.literals))


@dataclass(frozen=True)
class Evidence:
    """Unified input data aggregating unary and binary evidence."""

    unary: UnaryEvidence = UnaryEvidence()
    binary: BinaryEvidence = BinaryEvidence()

    @property
    def is_empty(self) -> bool:
        return self.unary.is_empty and self.binary.is_empty

    def cache_key_parts(self) -> tuple[object, object]:
        return (
            self.unary.cache_key_parts(),
            self.binary.cache_key_parts(),
        )


def _predicate_key(predicate: object) -> object:
    from wfomc.fol import Predicate

    if isinstance(predicate, Predicate):
        return predicate.cache_key_parts()
    return (str(predicate), None)


def _atom(predicate: object, *terms: object) -> Atom:
    if isinstance(predicate, str):
        ctx = FOLContext()
        predicate = ctx.predicate(predicate, len(terms))
        return ctx.atom(predicate, *terms)
    return predicate(*terms)


def _profile_literal(predicate: object, positive: bool) -> Literal:
    ctx = FOLContext()
    if isinstance(predicate, str):
        predicate = ctx.predicate(predicate, 1)
    return Literal(ctx.atom(predicate, ctx.variable("X")), positive)


__all__ = [
    "BinaryEvidence",
    "Evidence",
    "GroundBinaryLiteral",
    "GroundUnaryLiteral",
    "UnaryEvidence",
]
