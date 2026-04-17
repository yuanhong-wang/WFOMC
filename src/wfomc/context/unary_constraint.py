from __future__ import annotations
import numpy as np
from wfomc.fol.syntax import Pred


class UnaryConstraintHandler:
    """
    Manages unary cardinality constraints (mod, equality, less-than-or-equal)
    and provides mask-building and fast constraint-checking for IncrementalWFOMC3.
    """

    def __init__(self):
        self.mod_constraints: list[tuple[Pred, int, int]] = []   # (pred, r, k)
        self.eq_constraints: list[tuple[Pred, int]] = []          # (pred, k)
        self.le_constraints: list[tuple[Pred, int]] = []          # (pred, k_max)

    def add_mod(self, pred: Pred, r: int, k: int) -> None:
        self.mod_constraints.append((pred, r, k))

    def add_eq(self, pred: Pred, k: int) -> None:
        self.eq_constraints.append((pred, k))

    def add_le(self, pred: Pred, k_max: int) -> None:
        self.le_constraints.append((pred, k_max))

    # ------------------------------------------------------------------
    # Mask building (called once per cell graph)
    # ------------------------------------------------------------------

    def build_mask(self, cells) -> tuple[list, list, list]:
        """Return (mod_masks, eq_masks, le_masks) for the given cell list."""
        return (
            self.build_mod_mask(cells),
            self.build_eq_mask(cells),
            self.build_le_mask(cells),
        )

    def build_mod_mask(self, cells) -> list:
        n_cells = len(cells)
        return [
            (
                np.fromiter(
                    (1 if cell.is_positive(pred) else 0 for cell in cells),
                    dtype=np.int8, count=n_cells,
                ),
                r, k,
            )
            for pred, r, k in self.mod_constraints
        ]

    def build_eq_mask(self, cells) -> list:
        n_cells = len(cells)
        return [
            (
                np.fromiter(
                    (1 if cell.is_positive(pred) else 0 for cell in cells),
                    dtype=np.int8, count=n_cells,
                ),
                k_eq,
            )
            for pred, k_eq in self.eq_constraints
        ]

    def build_le_mask(self, cells) -> list:
        n_cells = len(cells)
        return [
            (
                np.fromiter(
                    (1 if cell.is_positive(pred) else 0 for cell in cells),
                    dtype=np.int8, count=n_cells,
                ),
                k_max,
            )
            for pred, k_max in self.le_constraints
        ]

    # ------------------------------------------------------------------
    # Constraint checking (called in the hot loop)
    # ------------------------------------------------------------------

    def check(self, config, mask) -> tuple[bool, bool, bool]:
        """Return (mod_violated, eq_violated, le_violated)."""
        return (
            self.check_mod(config, mask[0]),
            self.check_eq(config, mask[1]),
            self.check_le(config, mask[2]),
        )

    def check_mod(self, config, mod_mask) -> bool:
        for mask, r_mod, k_mod in mod_mask:
            if (mask @ np.fromiter(config, dtype=np.int32)) % k_mod != r_mod:
                return True
        return False

    def check_eq(self, config, eq_mask) -> bool:
        for mask, k_eq in eq_mask:
            if (mask @ np.fromiter(config, dtype=np.int32)) != k_eq:
                return True
        return False

    def check_le(self, config, le_mask) -> bool:
        vec = np.fromiter(config, dtype=np.int32)
        for mask, k_max in le_mask:
            if (mask @ vec) > k_max:
                return True
        return False
