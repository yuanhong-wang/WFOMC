"""Signed atoms for typed FOL formulas."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.fol.syntax import Atom, Predicate


@dataclass(frozen=True, slots=True)
class Literal:
    """A signed atom: positive means the atom holds, negative means it does not."""

    atom: Atom
    positive: bool = True

    @property
    def pred(self) -> object:
        return self.atom.predicate

    @property
    def predicate(self) -> object:
        return self.atom.predicate

    @property
    def args(self) -> tuple[object, ...]:
        return self.atom.terms

    @property
    def terms(self) -> tuple[object, ...]:
        return self.atom.terms

    def __invert__(self) -> "Literal":
        return Literal(self.atom, not self.positive)

    def make_positive(self) -> "Literal":
        return Literal(self.atom, True)

    def substitute(self, mapping: dict[object, object]) -> "Literal":
        from wfomc.fol.rewrite import substitute

        new_atom = substitute(self.atom, mapping)
        if isinstance(new_atom, Atom):
            return Literal(new_atom, self.positive)
        return self

    def cache_key_parts(self) -> tuple[object, bool]:
        predicate = self.atom.predicate
        if isinstance(predicate, Predicate):
            predicate_key = predicate.cache_key_parts()
        else:
            predicate_key = (str(predicate), None)
        return (predicate_key, self.positive)

    def __str__(self) -> str:
        sign = "" if self.positive else "~"
        return f"{sign}{self.atom}"

    def __hash__(self) -> int:
        return hash((self.atom, self.positive))


def positive_atom(lit_or_atom: Atom | Literal) -> Atom:
    if isinstance(lit_or_atom, Literal):
        return lit_or_atom.atom
    return lit_or_atom


__all__ = ["Literal", "positive_atom"]
