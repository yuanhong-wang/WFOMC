"""Small, algorithm-independent C2 normal-form IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wfomc.fol import Atom, Formula, Variable


@dataclass(frozen=True)
class ForallCountSection:
    """Direct row constraint ``forall X: count Y body(X,Y)``."""

    comparator: str
    count: int | tuple[int, int]
    body: "Atom"
    outer_var: "Variable"
    counted_var: "Variable"


@dataclass(frozen=True)
class CountSection:
    """Direct global constraint ``count X body(X)``."""

    comparator: str
    count: int | tuple[int, int]
    body: "Atom"
    counted_var: "Variable"


@dataclass(frozen=True)
class CountDefinition:
    """Meaning of a marker replacing a count inside a boolean formula."""

    marker: "Atom"
    section: CountSection | ForallCountSection


@dataclass(frozen=True)
class C2NormalForm:
    """Normalized C2 theory before choosing an algorithm-specific reduction."""

    qf_formula: "Formula | None" = None
    forall_exists: tuple["Formula", ...] = ()
    exists: tuple["Formula", ...] = ()
    forall_counts: tuple[ForallCountSection, ...] = ()
    counts: tuple[CountSection, ...] = ()
    count_definitions: tuple[CountDefinition, ...] = ()
    requires_nonempty_domain: bool = False

    @property
    def has_counting(self) -> bool:
        return bool(self.counts or self.forall_counts or self.count_definitions)


__all__ = [
    "C2NormalForm",
    "CountDefinition",
    "CountSection",
    "ForallCountSection",
]
