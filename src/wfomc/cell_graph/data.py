"""Immutable data structures shared across cell-graph construction and use."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wfomc.arithmetic import ArithmeticValue
from wfomc.fol import Literal, Predicate
from wfomc.fol import X as _X
from wfomc.fol.syntax import Term

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext
    from wfomc.cell_graph.evidence import CellEvidenceAllocation


@dataclass(frozen=True)
class Cell:
    """A satisfiable unary/diagonal type represented by predicate polarities."""

    code: tuple[bool, ...] = field(hash=False, compare=False)
    preds: tuple[Predicate, ...] = field(hash=False, compare=False)
    _identifier: frozenset[tuple[Predicate, bool]] = field(
        repr=False, init=False, hash=True, compare=True
    )

    def __post_init__(self):
        object.__setattr__(self, "_identifier", frozenset(zip(self.preds, self.code)))

    @functools.lru_cache(maxsize=None)
    def get_evidences(self, term: Term) -> frozenset[Literal]:
        """Return the typed signed-literal evidence set for *term*.

        Each predicate of arity ``k`` is applied to ``term`` repeated ``k``
        times; the polarity follows the cell code.
        """
        evidences: set[Literal] = set()
        for i, p in enumerate(self.preds):
            atom = p(*([term] * _arity(p)))
            evidences.add(Literal(atom, bool(self.code[i])))
        return frozenset(evidences)

    @functools.lru_cache(maxsize=None)
    def is_positive(self, pred: Predicate) -> bool:
        return self.code[_pred_index(self.preds, pred)]

    def __str__(self):
        evidences: frozenset[Literal] = self.get_evidences(_X)
        lits = [str(lit) for lit in evidences]
        lits.sort()
        return "^".join(lits)

    def __repr__(self):
        return self.__str__()


def _arity(pred: Predicate) -> int:
    return pred.arity


def _predicate_key(pred: Predicate) -> tuple[str, int]:
    """Stable ``(name, arity)`` identity for typed predicates."""
    return pred.cache_key_parts()


def _pred_index(preds: tuple[Predicate, ...], pred: Predicate) -> int:
    """Index of *pred* in *preds* by stable ``(name, arity)`` identity."""

    try:
        return preds.index(pred)
    except ValueError:
        key = _predicate_key(pred)
        for i, p in enumerate(preds):
            if _predicate_key(p) == key:
                return i
        raise ValueError(f"predicate {pred!r} not in {preds!r}")


@dataclass(frozen=True)
class PairFactor:
    """Exact pair weight with optional incremental3 mask projection."""

    total_weight: ArithmeticValue
    counting_weights: tuple[tuple[int, ArithmeticValue], ...] = ()


@dataclass(frozen=True)
class CellGraphData:
    """Plain shared output of cell-graph construction.

    Algorithms derive their own scalar, ordered, counting, or optimized inputs
    from these immutable weighted tables.
    """

    cells: tuple[Cell, ...]
    arithmetic: "ArithmeticContext"
    cell_weights: tuple[ArithmeticValue, ...]
    pair_factors: tuple[tuple[PairFactor, ...], ...]
    predecessor_pair_factors: tuple[
        tuple[int, tuple[tuple[PairFactor, ...], ...]], ...
    ] = ()

    def pair_weights(self) -> tuple[tuple[ArithmeticValue, ...], ...]:
        """Project the base two-tables to their unconditional weights."""

        return tuple(
            tuple(factor.total_weight for factor in row) for row in self.pair_factors
        )


PairWeightMatrix = tuple[tuple[ArithmeticValue, ...], ...]


@dataclass(frozen=True)
class CellGraphComponent:
    """Algorithm-facing scalar projection of one built cell graph.

    Algorithm-specific inputs extend this common materialized shape, whereas
    :class:`CellGraphData` remains the exact output of cell-graph construction.
    """

    cells: tuple[object, ...] = ()
    cell_weights: tuple[ArithmeticValue, ...] = ()
    pair_weights: PairWeightMatrix = ()
    graph_weight: ArithmeticValue | int = 1
    cell_evidence_allocation: "CellEvidenceAllocation | None" = None
