"""Input and factor-graph contracts for bounded-treewidth solving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from wfomc.algo.core import AlgoInput


class FactorGraphKind(Enum):
    LIFTED_CELL = "lifted-cell"
    GROUND = "ground"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Factor:
    name: str
    scope: tuple[object, ...]
    weight: object = None
    payload: object | None = None


@dataclass(frozen=True)
class FactorGraph:
    kind: FactorGraphKind
    variables: tuple[object, ...] = ()
    factors: tuple[Factor, ...] = ()
    evidence_factors: tuple[Factor, ...] = ()

    @property
    def all_factors(self) -> tuple[Factor, ...]:
        return self.factors + self.evidence_factors

    @property
    def variable_set(self) -> frozenset[object]:
        if self.variables:
            return frozenset(self.variables)
        return frozenset(
            variable for factor in self.all_factors for variable in factor.scope
        )


@dataclass(frozen=True)
class TreeBag:
    name: str
    variables: frozenset[object]
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class TreeDecomposition:
    bags: tuple[TreeBag, ...]
    root: str | None = None

    @property
    def width(self) -> int:
        if not self.bags:
            return -1
        return max(len(bag.variables) for bag in self.bags) - 1

    @property
    def bag_index(self) -> dict[str, TreeBag]:
        return {bag.name: bag for bag in self.bags}

    def validate_against(self, factor_graph: FactorGraph) -> None:
        if not self.bags and factor_graph.all_factors:
            raise ValueError("Tree decomposition must contain bags for factors.")
        bag_names = set(self.bag_index)
        if self.root is not None and self.root not in bag_names:
            raise ValueError(f"Tree decomposition root is unknown: {self.root}")
        for bag in self.bags:
            missing_children = set(bag.children) - bag_names
            if missing_children:
                missing = ", ".join(sorted(missing_children))
                raise ValueError(
                    f"Tree decomposition has unknown child bag(s): {missing}"
                )
        for factor in factor_graph.all_factors:
            factor_scope = frozenset(factor.scope)
            if factor_scope and not any(
                factor_scope <= bag.variables for bag in self.bags
            ):
                raise ValueError(
                    f"Factor {factor.name!r} scope is not covered by any bag."
                )


@dataclass(frozen=True)
class TreeDecompositionInput(AlgoInput):
    factor_graph: FactorGraph | None = None
    bags: tuple[TreeBag, ...] = ()
    local_factors: tuple[Factor, ...] = ()
    evidence_factors: tuple[Factor, ...] = ()


__all__ = [
    "Factor",
    "FactorGraph",
    "FactorGraphKind",
    "TreeBag",
    "TreeDecomposition",
    "TreeDecompositionInput",
]
