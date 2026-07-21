"""Boundary-profile partitions and decomposition-tree search.

The optimal decomposition is expensive to find, so the planner compares an
exact subset search on small cell sets with deterministic structural and
ordering heuristics.  All structural keys use interned integer interaction
labels because FLINT polynomial and Arb values are intentionally unhashable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import combinations
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence

from wfomc.arithmetic import ArithmeticValue

if TYPE_CHECKING:
    from .input import BoundaryProfileComponent


class BPNodeKind(Enum):
    """Evaluation rule attached to one decomposition node."""

    # A singleton cell whose table is the precomputed local W table.
    LEAF = "leaf"
    # An ordinary binary Boundary-Profile join.
    JOIN = "join"
    # A mutually non-interacting block with exponential local tables.
    INDEPENDENT = "independent"
    # Repeated identical cells combined by balanced table exponentiation.
    SYMMETRIC = "symmetric"


@dataclass(frozen=True)
class BoundaryPartition:
    """Canonical boundary classes for one cell subset."""

    classes: tuple[tuple[int, ...], ...]
    cell_to_class: tuple[int, ...]


@dataclass(frozen=True)
class BoundaryProfileNode:
    """One postorder node of a compiled Boundary-Profile plan."""

    kind: BPNodeKind
    members_mask: int
    members: tuple[int, ...]
    classes: tuple[tuple[int, ...], ...]
    left: int | None = None
    right: int | None = None
    left_projection: tuple[int, ...] = ()
    right_projection: tuple[int, ...] = ()
    cross_nonunit: tuple[tuple[int, int, ArithmeticValue], ...] = ()


@dataclass(frozen=True)
class BoundaryProfileTreeNode:
    """Domain-free postorder node stored in an input template."""

    kind: BPNodeKind
    members_mask: int
    members: tuple[int, ...]
    classes: tuple[tuple[int, ...], ...]
    left: int | None = None
    right: int | None = None
    left_projection: tuple[int, ...] = ()
    right_projection: tuple[int, ...] = ()
    cross_pairs: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class BPPlanEstimate:
    """Planner diagnostics for one candidate decomposition."""

    strategy: str
    bp_width: int
    join_width: int
    estimated_states: int
    estimated_join_pairs: int
    estimated_log_work: float


@dataclass(frozen=True)
class BoundaryProfilePlan:
    """Selected postorder decomposition and all candidate estimates."""

    tree: "BoundaryProfileTree"
    nodes: tuple[BoundaryProfileNode, ...]
    root: int
    strategy: str
    bp_width: int
    join_width: int
    estimated_states: int
    estimated_join_pairs: int
    estimated_log_work: float
    candidate_estimates: tuple[BPPlanEstimate, ...] = ()


@dataclass(frozen=True)
class _Tree:
    mask: int
    kind: BPNodeKind
    left: "_Tree | None"
    right: "_Tree | None"
    estimated_log_work: float
    peak_log_join: float


@dataclass(frozen=True)
class BoundaryProfileTree:
    """Selected domain-free BP topology reused by concrete domains."""

    nodes: tuple[BoundaryProfileTreeNode, ...]
    root: int
    strategy: str
    bp_width: int
    join_width: int
    _topology: _Tree | None
    _candidates: tuple[tuple[str, _Tree], ...] = ()


class _ComponentLike(Protocol):
    cell_weights: tuple[ArithmeticValue, ...]
    r_matrix: tuple[tuple[ArithmeticValue, ...], ...]


def build_boundary_profile_plan(
    component: _ComponentLike,
    domain_size: int,
    arithmetic,
    *,
    tree_reference_domain_size: int | None = None,
) -> BoundaryProfilePlan:
    """Convenience wrapper that builds and materializes one BP tree."""

    tree = build_boundary_profile_tree(
        component.cell_weights,
        component.r_matrix,
        arithmetic,
        reference_domain_size=tree_reference_domain_size,
    )
    return materialize_boundary_profile_plan(
        tree,
        component,
        domain_size,
        arithmetic,
    )


def build_boundary_profile_tree(
    cell_weights: tuple[ArithmeticValue, ...],
    r_matrix: tuple[tuple[ArithmeticValue, ...], ...],
    arithmetic,
    *,
    reference_domain_size: int | None = None,
) -> BoundaryProfileTree:
    """Search once for a domain-free decomposition topology."""

    context = _PlanningContext(
        cell_weights,
        r_matrix,
        arithmetic,
        planning_domain_size=reference_domain_size,
    )
    if context.cell_count == 0:
        return BoundaryProfileTree(
            nodes=(),
            root=-1,
            strategy="empty",
            bp_width=0,
            join_width=0,
            _topology=None,
        )

    candidates: list[tuple[str, _Tree]] = []
    all_cells = tuple(range(context.cell_count))

    tail_order = context.tail_order_lookahead2(all_cells)
    candidates.append(("tail-caterpillar", context.caterpillar(tail_order)))
    candidates.append(("greedy-agglomerative", context.agglomerative(all_cells)))

    structural = context.structure_aware_tree(all_cells)
    candidates.append(("components-symmetric", structural))

    if context.cell_count <= 12:
        candidates.append(("exact-subset-cost", context.exact_subset_tree(all_cells)))

    independent = context.greedy_independent_set(all_cells)
    if independent:
        independent_set = frozenset(independent)
        remainder = tuple(cell for cell in all_cells if cell not in independent_set)
        independent_tree = context.independent(independent)
        if remainder:
            candidates.append(
                (
                    "independent-tail",
                    context.join(
                        independent_tree,
                        context.caterpillar(
                            context.tail_order_lookahead2(remainder)
                        ),
                    ),
                )
            )
            candidates.append(
                (
                    "independent-agglomerative",
                    context.join(independent_tree, context.agglomerative(remainder)),
                )
            )
        else:
            candidates.append(("all-independent", independent_tree))

    unique: dict[tuple[object, ...], tuple[str, _Tree]] = {}
    for strategy, tree in candidates:
        unique.setdefault(_tree_key(tree), (strategy, tree))

    unique_candidates = tuple(unique.values())
    strategy, selected = min(
        unique_candidates,
        key=lambda candidate: context.selection_key(candidate[1])
        + (candidate[0],),
    )
    return context.compile_tree(
        selected,
        strategy,
        candidates=unique_candidates,
    )


def materialize_boundary_profile_plan(
    tree: BoundaryProfileTree,
    component: _ComponentLike,
    domain_size: int,
    arithmetic,
) -> BoundaryProfilePlan:
    """Bind one cached tree to concrete values and domain-sized estimates."""

    context = _PlanningContext(
        component.cell_weights,
        component.r_matrix,
        arithmetic,
        planning_domain_size=domain_size,
    )
    return context.materialize_plan(tree)


class _PlanningContext:
    def __init__(
        self,
        cell_weights: tuple[ArithmeticValue, ...],
        r_matrix: tuple[tuple[ArithmeticValue, ...], ...],
        arithmetic,
        *,
        planning_domain_size: int | None,
    ):
        self.cell_weights = cell_weights
        self.r_matrix = r_matrix
        self.planning_domain_size = planning_domain_size
        self.arithmetic = arithmetic
        self.cell_count = len(cell_weights)
        self.full_mask = (1 << self.cell_count) - 1
        self._validate_shapes()
        self.interaction_labels = self._intern_interactions()
        self._validate_symmetry()

    def _validate_shapes(self) -> None:
        if self.planning_domain_size is not None and self.planning_domain_size < 0:
            raise ValueError("planning domain size must be non-negative")
        if len(self.r_matrix) != self.cell_count:
            raise ValueError("Interaction matrix size must match cell weights")
        if any(len(row) != self.cell_count for row in self.r_matrix):
            raise ValueError("Interaction matrix must be square")

    def _intern_interactions(self) -> tuple[tuple[int, ...], ...]:
        buckets: dict[tuple[type, str], list[tuple[int, object]]] = {}
        next_label = 0
        rows: list[tuple[int, ...]] = []
        for row in self.r_matrix:
            labels = []
            for value in row:
                bucket_key = (type(value), str(value))
                bucket = buckets.setdefault(bucket_key, [])
                label = None
                for candidate_label, candidate_value in bucket:
                    if _value_equal(value, candidate_value):
                        label = candidate_label
                        break
                if label is None:
                    label = next_label
                    next_label += 1
                    bucket.append((label, value))
                labels.append(label)
            rows.append(tuple(labels))
        return tuple(rows)

    def _validate_symmetry(self) -> None:
        for left in range(self.cell_count):
            for right in range(left + 1, self.cell_count):
                if not _value_equal(
                    self.r_matrix[left][right],
                    self.r_matrix[right][left],
                ):
                    raise ValueError(
                        "Boundary-Profile requires a symmetric interaction matrix"
                    )

    def members(self, mask: int) -> tuple[int, ...]:
        return tuple(cell for cell in range(self.cell_count) if mask & (1 << cell))

    @lru_cache(maxsize=None)
    def partition(self, mask: int) -> BoundaryPartition:
        members = self.members(mask)
        if not members:
            raise ValueError("Boundary partitions are undefined for an empty block")
        outside = tuple(
            cell for cell in range(self.cell_count) if not mask & (1 << cell)
        )
        groups: dict[tuple[int, ...], list[int]] = {}
        for cell in members:
            signature = tuple(self.interaction_labels[cell][other] for other in outside)
            groups.setdefault(signature, []).append(cell)
        classes = tuple(
            sorted((tuple(group) for group in groups.values()), key=lambda group: group[0])
        )
        cell_to_class = [-1] * self.cell_count
        for class_index, group in enumerate(classes):
            for cell in group:
                cell_to_class[cell] = class_index
        return BoundaryPartition(classes, tuple(cell_to_class))

    @lru_cache(maxsize=None)
    def state_bound(self, mask: int) -> int:
        dimension = len(self.partition(mask).classes)
        if self.planning_domain_size is None:
            return dimension + 1
        return math.comb(self.planning_domain_size + dimension, dimension)

    def log_state_bound(self, mask: int) -> float:
        return math.log(max(1, self.state_bound(mask)))

    @lru_cache(maxsize=None)
    def projection(self, child_mask: int, parent_mask: int) -> tuple[int, ...]:
        child = self.partition(child_mask)
        parent = self.partition(parent_mask)
        result = []
        for child_class in child.classes:
            parent_index = parent.cell_to_class[child_class[0]]
            if parent_index < 0 or any(
                parent.cell_to_class[cell] != parent_index for cell in child_class
            ):
                raise AssertionError("A child boundary class split at its parent")
            result.append(parent_index)
        return tuple(result)

    @lru_cache(maxsize=None)
    def cross_pairs(
        self,
        left_mask: int,
        right_mask: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        left = self.partition(left_mask)
        right = self.partition(right_mask)
        result = []
        for left_index, left_class in enumerate(left.classes):
            for right_index, right_class in enumerate(right.classes):
                representative_label = self.interaction_labels[left_class[0]][right_class[0]]
                if any(
                    self.interaction_labels[left_cell][right_cell]
                    != representative_label
                    for left_cell in left_class
                    for right_cell in right_class
                ):
                    raise AssertionError("Cross interaction is not constant on BP classes")
                result.append(
                    (
                        left_index,
                        right_index,
                        left_class[0],
                        right_class[0],
                    )
                )
        return tuple(result)

    def cross_nonunit_count(self, left_mask: int, right_mask: int) -> int:
        return sum(
            1
            for _left_index, _right_index, left_cell, right_cell in self.cross_pairs(
                left_mask,
                right_mask,
            )
            if not self.arithmetic.is_one(self.r_matrix[left_cell][right_cell])
        )

    def leaf(self, cell: int) -> _Tree:
        mask = 1 << cell
        work = self.log_state_bound(mask)
        return _Tree(mask, BPNodeKind.LEAF, None, None, work, work)

    def independent(self, cells: Sequence[int]) -> _Tree:
        mask = _mask(cells)
        if not self.is_independent(mask):
            raise ValueError("Requested block is not exponentially independent")
        work = self.log_state_bound(mask)
        return _Tree(mask, BPNodeKind.INDEPENDENT, None, None, work, work)

    def symmetric(self, cells: Sequence[int]) -> _Tree:
        mask = _mask(cells)
        if not self.is_symmetric_clique(mask):
            raise ValueError("Requested block is not a symmetric clique")
        combines = max(1, math.ceil(math.log2(len(cells))))
        domain_scale = (
            self.planning_domain_size + 1
            if self.planning_domain_size is not None
            else len(self.partition(mask).classes) + 1
        )
        work = math.log(combines) + 2.0 * math.log(domain_scale)
        return _Tree(mask, BPNodeKind.SYMMETRIC, None, None, work, work)

    def join(self, left: _Tree, right: _Tree) -> _Tree:
        if left.mask & right.mask:
            raise ValueError("Boundary-Profile children must be disjoint")
        if _tree_order_key(right) < _tree_order_key(left):
            left, right = right, left
        mask = left.mask | right.mask
        cross_count = self.cross_nonunit_count(left.mask, right.mask)
        if mask == self.full_mask and (
            left.kind is BPNodeKind.INDEPENDENT
            or right.kind is BPNodeKind.INDEPENDENT
        ):
            ordinary = right if left.kind is BPNodeKind.INDEPENDENT else left
            independent = left if left.kind is BPNodeKind.INDEPENDENT else right
            join_log = self.log_state_bound(ordinary.mask) + math.log(
                max(
                    1,
                    len(self.members(independent.mask))
                    * len(self.partition(ordinary.mask).classes),
                )
            )
        else:
            join_log = (
                self.log_state_bound(left.mask)
                + self.log_state_bound(right.mask)
                + math.log(max(1, cross_count))
            )
        total_work = _logsumexp(
            left.estimated_log_work,
            right.estimated_log_work,
            join_log,
        )
        return _Tree(
            mask,
            BPNodeKind.JOIN,
            left,
            right,
            total_work,
            max(left.peak_log_join, right.peak_log_join, join_log),
        )

    def caterpillar(self, order: Sequence[int]) -> _Tree:
        if not order:
            raise ValueError("A caterpillar requires at least one cell")
        tree = self.leaf(order[0])
        for cell in order[1:]:
            tree = self.join(tree, self.leaf(cell))
        return tree

    def agglomerative(self, cells: Sequence[int] | Sequence[_Tree]) -> _Tree:
        if not cells:
            raise ValueError("Agglomerative planning requires at least one block")
        first = cells[0]
        forest = (
            list(cells)
            if isinstance(first, _Tree)
            else [self.leaf(int(cell)) for cell in cells]
        )
        while len(forest) > 1:
            best = None
            for left_index, right_index in combinations(range(len(forest)), 2):
                joined = self.join(forest[left_index], forest[right_index])
                key = self._tree_cost_key(joined)
                candidate = (key, left_index, right_index, joined)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            assert best is not None
            _key, left_index, right_index, joined = best
            forest = [
                tree
                for index, tree in enumerate(forest)
                if index not in (left_index, right_index)
            ]
            forest.append(joined)
            forest.sort(key=_tree_order_key)
        return forest[0]

    def _tree_cost_key(self, tree: _Tree) -> tuple[object, ...]:
        if self.planning_domain_size is None:
            peak_join_width, bp_width, cross_count, imbalance = (
                self._structural_metrics(tree)
            )
            return (
                peak_join_width,
                bp_width,
                cross_count,
                imbalance,
                _tree_key(tree),
            )
        return (
            tree.peak_log_join,
            tree.estimated_log_work,
            len(self.partition(tree.mask).classes),
            abs(_bit_count(tree.left.mask) - _bit_count(tree.right.mask))
            if tree.left is not None and tree.right is not None
            else 0,
            _tree_key(tree),
        )

    def selection_key(self, tree: _Tree) -> tuple[object, ...]:
        """Return the deterministic cost used to select one cached tree."""

        return self._tree_cost_key(tree)

    @lru_cache(maxsize=None)
    def _structural_metrics(self, tree: _Tree) -> tuple[int, int, int, int]:
        bp_width = len(self.partition(tree.mask).classes)
        if tree.left is None or tree.right is None:
            return (0, bp_width, 0, 0)
        left = self._structural_metrics(tree.left)
        right = self._structural_metrics(tree.right)
        join_width = len(self.partition(tree.left.mask).classes) + len(
            self.partition(tree.right.mask).classes
        )
        return (
            max(left[0], right[0], join_width),
            max(left[1], right[1], bp_width),
            left[2]
            + right[2]
            + self.cross_nonunit_count(tree.left.mask, tree.right.mask),
            left[3]
            + right[3]
            + abs(_bit_count(tree.left.mask) - _bit_count(tree.right.mask)),
        )

    def exact_subset_tree(self, cells: Sequence[int]) -> _Tree:
        """Optimize the planner cost exactly over all binary subset splits."""

        target_mask = _mask(cells)
        best: dict[int, _Tree] = {1 << cell: self.leaf(cell) for cell in cells}
        for size in range(2, len(cells) + 1):
            for subset in combinations(cells, size):
                mask = _mask(subset)
                lowest = mask & -mask
                candidate_best = None
                left_mask = (mask - 1) & mask
                while left_mask:
                    right_mask = mask ^ left_mask
                    if right_mask and left_mask & lowest and left_mask in best and right_mask in best:
                        joined = self.join(best[left_mask], best[right_mask])
                        key = self._tree_cost_key(joined)
                        if candidate_best is None or key < candidate_best[0]:
                            candidate_best = (key, joined)
                    left_mask = (left_mask - 1) & mask
                assert candidate_best is not None
                best[mask] = candidate_best[1]
        return best[target_mask]

    def tail_order_lookahead2(self, cells: Sequence[int]) -> tuple[int, ...]:
        tail = sorted(cells)
        classes: set[tuple[int, ...]] = set()
        order = []
        while tail:
            if len(tail) == 1:
                order.append(tail[0])
                break
            positions = {cell: index for index, cell in enumerate(tail)}
            best = None
            for cell in tail:
                index = positions[cell]
                tail1 = [other for other in tail if other != cell]
                classes1 = {
                    signature[:index] + signature[index + 1 :]
                    for signature in classes
                }
                classes1.add(
                    tuple(self.interaction_labels[cell][other] for other in tail1)
                )
                score1 = len(classes1)
                score2 = score1
                if tail1:
                    score2 = min(
                        len(
                            {
                                signature[:next_index] + signature[next_index + 1 :]
                                for signature in classes1
                            }
                            | {
                                tuple(
                                    self.interaction_labels[next_cell][other]
                                    for other in tail1
                                    if other != next_cell
                                )
                            }
                        )
                        for next_index, next_cell in enumerate(tail1)
                    )
                candidate = (score2, score1, cell, classes1)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            assert best is not None
            _score2, _score1, selected, selected_classes = best
            order.append(selected)
            tail.remove(selected)
            classes = selected_classes
        return tuple(order)

    def greedy_independent_set(self, cells: Sequence[int]) -> tuple[int, ...]:
        allowed = {
            cell for cell in cells if self._is_exponential_table(cell)
        }
        result = []
        while allowed:
            selected = min(
                allowed,
                key=lambda cell: (
                    sum(
                        1
                        for other in allowed
                        if other != cell
                        and not self.arithmetic.is_one(
                            self.r_matrix[cell][other]
                        )
                    ),
                    cell,
                ),
            )
            result.append(selected)
            neighbours = {
                other
                for other in allowed
                if other != selected
                and not self.arithmetic.is_one(
                    self.r_matrix[selected][other]
                )
            }
            allowed.remove(selected)
            allowed -= neighbours
        return tuple(result)

    @lru_cache(maxsize=None)
    def _is_exponential_table(self, cell: int) -> bool:
        return self.arithmetic.is_zero(
            self.cell_weights[cell]
        ) or self.arithmetic.is_one(self.r_matrix[cell][cell])

    @lru_cache(maxsize=None)
    def is_independent(self, mask: int) -> bool:
        members = self.members(mask)
        return all(self._is_exponential_table(cell) for cell in members) and all(
            self.arithmetic.is_one(self.r_matrix[left][right])
            for left, right in combinations(members, 2)
        )

    def symmetric_groups(self, cells: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        remaining = set(cells)
        result = []
        while remaining:
            seed = min(remaining)
            group = [seed]
            for candidate in sorted(remaining - {seed}):
                proposed = tuple(group + [candidate])
                if self.is_symmetric_clique(_mask(proposed)):
                    group.append(candidate)
            for cell in group:
                remaining.remove(cell)
            result.append(tuple(group))
        return tuple(result)

    @lru_cache(maxsize=None)
    def is_symmetric_clique(self, mask: int) -> bool:
        members = self.members(mask)
        if len(members) <= 1:
            return True
        base = members[0]
        if any(
            not _value_equal(self.cell_weights[base], self.cell_weights[other])
            or (
                not self.arithmetic.is_zero(self.cell_weights[base])
                and not _value_equal(
                    self.r_matrix[base][base],
                    self.r_matrix[other][other],
                )
            )
            for other in members[1:]
        ):
            return False
        member_set = set(members)
        outside = [cell for cell in range(self.cell_count) if cell not in member_set]
        if any(
            self.interaction_labels[base][external]
            != self.interaction_labels[other][external]
            for other in members[1:]
            for external in outside
        ):
            return False
        internal_label = self.interaction_labels[members[0]][members[1]]
        if any(
            self.interaction_labels[left][right] != internal_label
            for left, right in combinations(members, 2)
        ):
            return False
        return len(self.partition(mask).classes) == 1

    def connected_components(self, cells: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        remaining = set(cells)
        components = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            stack = [start]
            component = []
            while stack:
                cell = stack.pop()
                component.append(cell)
                neighbours = {
                    other
                    for other in remaining
                    if not self.arithmetic.is_one(
                        self.r_matrix[cell][other]
                    )
                }
                remaining -= neighbours
                stack.extend(sorted(neighbours, reverse=True))
            components.append(tuple(sorted(component)))
        return tuple(sorted(components, key=lambda group: (len(group), group)))

    def structure_aware_tree(self, cells: Sequence[int]) -> _Tree:
        component_trees = []
        for component in self.connected_components(cells):
            atoms = []
            for group in self.symmetric_groups(component):
                if len(group) > 1:
                    atoms.append(self.symmetric(group))
                else:
                    atoms.append(self.leaf(group[0]))
            component_trees.append(self.agglomerative(atoms))
        return self.agglomerative(component_trees)

    def compile_tree(
        self,
        tree: _Tree,
        strategy: str,
        *,
        candidates: tuple[tuple[str, _Tree], ...],
    ) -> BoundaryProfileTree:
        """Compile one selected topology into reusable structural nodes."""

        nodes: list[BoundaryProfileTreeNode] = []

        def visit(current: _Tree) -> int:
            partition = self.partition(current.mask)
            if current.kind is not BPNodeKind.JOIN:
                node = BoundaryProfileTreeNode(
                    kind=current.kind,
                    members_mask=current.mask,
                    members=self.members(current.mask),
                    classes=partition.classes,
                )
                nodes.append(node)
                return len(nodes) - 1

            assert current.left is not None and current.right is not None
            left_index = visit(current.left)
            right_index = visit(current.right)
            node = BoundaryProfileTreeNode(
                kind=BPNodeKind.JOIN,
                members_mask=current.mask,
                members=self.members(current.mask),
                classes=partition.classes,
                left=left_index,
                right=right_index,
                left_projection=self.projection(current.left.mask, current.mask),
                right_projection=self.projection(current.right.mask, current.mask),
                cross_pairs=self.cross_pairs(current.left.mask, current.right.mask),
            )
            nodes.append(node)
            return len(nodes) - 1

        root = visit(tree)
        bp_width = max(len(node.classes) for node in nodes)
        join_width = max(
            (
                len(nodes[node.left].classes) + len(nodes[node.right].classes)
                for node in nodes
                if node.kind is BPNodeKind.JOIN
                and node.left is not None
                and node.right is not None
            ),
            default=1,
        )
        return BoundaryProfileTree(
            nodes=tuple(nodes),
            root=root,
            strategy=strategy,
            bp_width=bp_width,
            join_width=join_width,
            _topology=tree,
            _candidates=candidates,
        )

    def materialize_plan(self, tree: BoundaryProfileTree) -> BoundaryProfilePlan:
        """Bind concrete pair values and estimates to a cached tree."""

        if tree.root < 0:
            estimate = BPPlanEstimate("empty", 0, 0, 0, 0, 0.0)
            return BoundaryProfilePlan(
                tree=tree,
                nodes=(),
                root=-1,
                strategy="empty",
                bp_width=0,
                join_width=0,
                estimated_states=0,
                estimated_join_pairs=0,
                estimated_log_work=0.0,
                candidate_estimates=(estimate,),
            )

        nodes = tuple(
            BoundaryProfileNode(
                kind=node.kind,
                members_mask=node.members_mask,
                members=node.members,
                classes=node.classes,
                left=node.left,
                right=node.right,
                left_projection=node.left_projection,
                right_projection=node.right_projection,
                cross_nonunit=tuple(
                    (left_index, right_index, value)
                    for left_index, right_index, left_cell, right_cell
                    in node.cross_pairs
                    if not self.arithmetic.is_one(
                        value := self.r_matrix[left_cell][right_cell]
                    )
                ),
            )
            for node in tree.nodes
        )
        if tree._topology is None:
            raise AssertionError("Non-empty Boundary-Profile tree has no topology")
        selected = self.estimate(tree._topology, tree.strategy)
        estimates = tuple(
            self.estimate(candidate, strategy)
            for strategy, candidate in sorted(
                tree._candidates,
                key=lambda item: item[0],
            )
        )
        return BoundaryProfilePlan(
            tree=tree,
            nodes=nodes,
            root=tree.root,
            strategy=tree.strategy,
            bp_width=tree.bp_width,
            join_width=tree.join_width,
            estimated_states=selected.estimated_states,
            estimated_join_pairs=selected.estimated_join_pairs,
            estimated_log_work=selected.estimated_log_work,
            candidate_estimates=estimates,
        )

    def estimate(self, tree: _Tree, strategy: str) -> BPPlanEstimate:
        """Estimate one existing topology without searching for another tree."""

        tuned = self._retune(tree)
        indexed: list[tuple[_Tree, int | None, int | None]] = []

        def visit(current: _Tree) -> int:
            left_index = visit(current.left) if current.left is not None else None
            right_index = visit(current.right) if current.right is not None else None
            indexed.append((current, left_index, right_index))
            return len(indexed) - 1

        root = visit(tuned)
        bp_width = max(
            len(self.partition(node.mask).classes) for node, _left, _right in indexed
        )
        join_width = max(
            (
                len(self.partition(indexed[left][0].mask).classes)
                + len(self.partition(indexed[right][0].mask).classes)
                for node, left, right in indexed
                if node.kind is BPNodeKind.JOIN
                and left is not None
                and right is not None
            ),
            default=1,
        )
        skipped: set[int] = set()
        root_node, root_left, root_right = indexed[root]
        if (
            root_node.kind is BPNodeKind.JOIN
            and root_left is not None
            and root_right is not None
        ):
            if indexed[root_left][0].kind is BPNodeKind.INDEPENDENT:
                skipped.add(root_left)
            elif indexed[root_right][0].kind is BPNodeKind.INDEPENDENT:
                skipped.add(root_right)
        estimated_states = sum(
            self.state_bound(node.mask)
            for index, (node, _left, _right) in enumerate(indexed)
            if index not in skipped
        )
        estimated_join_pairs = 0
        for node, left_index, right_index in indexed:
            if (
                node.kind is not BPNodeKind.JOIN
                or left_index is None
                or right_index is None
            ):
                continue
            left = indexed[left_index][0]
            right = indexed[right_index][0]
            if node.mask == self.full_mask and (
                left.kind is BPNodeKind.INDEPENDENT
                or right.kind is BPNodeKind.INDEPENDENT
            ):
                ordinary = right if left.kind is BPNodeKind.INDEPENDENT else left
                estimated_join_pairs += self.state_bound(ordinary.mask)
            else:
                estimated_join_pairs += self.state_bound(
                    left.mask
                ) * self.state_bound(right.mask)
        return BPPlanEstimate(
            strategy=strategy,
            bp_width=bp_width,
            join_width=join_width,
            estimated_states=estimated_states,
            estimated_join_pairs=estimated_join_pairs,
            estimated_log_work=tuned.estimated_log_work,
        )

    def _retune(self, tree: _Tree) -> _Tree:
        if tree.kind is BPNodeKind.LEAF:
            return self.leaf(self.members(tree.mask)[0])
        if tree.kind is BPNodeKind.INDEPENDENT:
            return self.independent(self.members(tree.mask))
        if tree.kind is BPNodeKind.SYMMETRIC:
            return self.symmetric(self.members(tree.mask))
        if tree.left is None or tree.right is None:
            raise AssertionError("Join topology is missing a child")
        return self.join(self._retune(tree.left), self._retune(tree.right))


def _mask(cells: Iterable[int]) -> int:
    result = 0
    for cell in cells:
        result |= 1 << int(cell)
    return result


def _bit_count(mask: int) -> int:
    return mask.bit_count()


def _value_equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _logsumexp(*values: float) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _tree_order_key(tree: _Tree) -> tuple[object, ...]:
    return (tree.mask, tree.kind.value, _tree_key(tree))


def _tree_key(tree: _Tree) -> tuple[object, ...]:
    if tree.left is None or tree.right is None:
        return (tree.kind.value, tree.mask)
    children = sorted((_tree_key(tree.left), _tree_key(tree.right)))
    return (tree.kind.value, tree.mask, children[0], children[1])


__all__ = [
    "BPNodeKind",
    "BPPlanEstimate",
    "BoundaryPartition",
    "BoundaryProfileNode",
    "BoundaryProfilePlan",
    "BoundaryProfileTree",
    "BoundaryProfileTreeNode",
    "build_boundary_profile_plan",
    "build_boundary_profile_tree",
    "materialize_boundary_profile_plan",
]
