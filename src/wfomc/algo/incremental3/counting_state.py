"""Counting-state data for incremental3 materialization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np

from wfomc.fol import Literal, Predicate
from wfomc.fol.normal_form import C2NormalForm


@dataclass(frozen=True)
class CountingState:
    ext_preds: tuple[Predicate, ...]
    cnt_preds: tuple[Predicate, ...]
    cnt_params: tuple[int, ...]
    cnt_remainder: tuple[int | None, ...]
    exist_mod: bool
    mod_pred_index: tuple[int, ...]
    exist_le: bool
    le_index: tuple[int, ...]
    binary_evidence: tuple[frozenset[Literal], ...]
    c_type_shape: tuple[int, ...]


class UnaryCardinalityMasks:
    def __init__(self) -> None:
        self.mod_constraints: list[tuple[Predicate, int, int]] = []
        self.eq_constraints: list[tuple[Predicate, int]] = []
        self.le_constraints: list[tuple[Predicate, int]] = []
        self.ge_constraints: list[tuple[Predicate, int]] = []

    def add_mod(self, pred: Predicate, r: int, k: int) -> None:
        self.mod_constraints.append((pred, r, k))

    def add_eq(self, pred: Predicate, k: int) -> None:
        self.eq_constraints.append((pred, k))

    def add_le(self, pred: Predicate, k_max: int) -> None:
        self.le_constraints.append((pred, k_max))

    def add_ge(self, pred: Predicate, k_min: int) -> None:
        self.ge_constraints.append((pred, k_min))

    def required_predicates(self) -> frozenset[Predicate]:
        """Unary predicates that must be represented in materialized cells."""

        return frozenset(
            predicate
            for constraints in (
                self.mod_constraints,
                self.eq_constraints,
                self.le_constraints,
                self.ge_constraints,
            )
            for predicate, *_parameters in constraints
        )

    def build_mask(self, cells) -> tuple[list, list, list, list]:
        return (
            self.build_mod_mask(cells),
            self.build_eq_mask(cells),
            self.build_le_mask(cells),
            self.build_ge_mask(cells),
        )

    def build_mod_mask(self, cells) -> list:
        n_cells = len(cells)
        return [
            (
                np.fromiter(
                    (1 if cell.is_positive(pred) else 0 for cell in cells),
                    dtype=np.int8,
                    count=n_cells,
                ),
                r,
                k,
            )
            for pred, r, k in self.mod_constraints
        ]

    def build_eq_mask(self, cells) -> list:
        n_cells = len(cells)
        return [
            (
                np.fromiter(
                    (1 if cell.is_positive(pred) else 0 for cell in cells),
                    dtype=np.int8,
                    count=n_cells,
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
                    dtype=np.int8,
                    count=n_cells,
                ),
                k_max,
            )
            for pred, k_max in self.le_constraints
        ]

    def build_ge_mask(self, cells) -> list:
        n_cells = len(cells)
        return [
            (
                np.fromiter(
                    (1 if cell.is_positive(pred) else 0 for cell in cells),
                    dtype=np.int8,
                    count=n_cells,
                ),
                k_min,
            )
            for pred, k_min in self.ge_constraints
        ]

    def check(self, config, mask) -> tuple[bool, bool, bool, bool]:
        vec = np.fromiter(config, dtype=np.int32)
        return (
            self.check_mod(config, mask[0], vec),
            self.check_eq(config, mask[1], vec),
            self.check_le(config, mask[2], vec),
            self.check_ge(config, mask[3], vec),
        )

    def check_mod(self, config, mod_mask, vec=None) -> bool:
        if vec is None:
            vec = np.fromiter(config, dtype=np.int32)
        for mask, r_mod, k_mod in mod_mask:
            if (mask @ vec) % k_mod != r_mod:
                return True
        return False

    def check_eq(self, config, eq_mask, vec=None) -> bool:
        if vec is None:
            vec = np.fromiter(config, dtype=np.int32)
        for mask, k_eq in eq_mask:
            if (mask @ vec) != k_eq:
                return True
        return False

    def check_le(self, config, le_mask, vec=None) -> bool:
        if vec is None:
            vec = np.fromiter(config, dtype=np.int32)
        for mask, k_max in le_mask:
            if (mask @ vec) > k_max:
                return True
        return False

    def check_ge(self, config, ge_mask, vec=None) -> bool:
        if vec is None:
            vec = np.fromiter(config, dtype=np.int32)
        for mask, k_min in ge_mask:
            if (mask @ vec) < k_min:
                return True
        return False


def build_counting_state_for_normal_form(
    normal_form: C2NormalForm,
) -> tuple[CountingState, UnaryCardinalityMasks]:
    if normal_form.count_definitions:
        from wfomc.errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            "incremental3 does not support counting quantifiers embedded "
            "in boolean contexts"
        )
    masks = UnaryCardinalityMasks()
    cnt_preds: list[object] = []
    cnt_params: list[int] = []
    cnt_remainder: list[object] = []
    mod_pred_index: list[int] = []
    le_pred: list[object] = []
    ge_pred: list[object] = []

    for section in normal_form.counts:
        pred = _counting_predicate("unary", section.body)
        comparator = section.comparator
        count = section.count
        if comparator == "mod":
            r, k = count
            masks.add_mod(pred, int(r), int(k))
        elif comparator == "=":
            masks.add_eq(pred, int(count))
        elif comparator == "<=":
            masks.add_le(pred, int(count))
        elif comparator == ">=":
            masks.add_ge(pred, int(count))
        else:
            raise ValueError(
                f"incremental3 does not support global count comparator "
                f"{comparator!r}"
            )

    for section in normal_form.forall_counts:
        pred = _counting_predicate("binary", section.body)
        comparator = section.comparator
        count = section.count
        idx = len(cnt_preds)
        if comparator == "mod":
            r, k = count
            mod_pred_index.append(idx)
            cnt_remainder.append(int(r))
            cnt_params.append(int(k))
            cnt_preds.append(pred)
        elif comparator == "=":
            cnt_remainder.append(None)
            cnt_params.append(int(count))
            cnt_preds.append(pred)
        elif comparator == "<=":
            cnt_remainder.append(None)
            cnt_params.append(int(count))
            cnt_preds.append(pred)
            le_pred.append(pred)
        elif comparator == ">=":
            if int(count) != 1:
                raise ValueError(
                    "incremental3 only supports row lower bounds of >= 1"
                )
            ge_pred.append(pred)
        else:
            raise ValueError(
                f"incremental3 does not support row count comparator "
                f"{comparator!r}"
            )

    ext_preds = list(_existential_predicates(normal_form))
    ext_preds.extend(pred for pred in ge_pred if pred not in ext_preds)
    c_type_shape = [2 for _ in ext_preds]
    for idx, k in enumerate(cnt_params):
        c_type_shape.append(k if idx in mod_pred_index else k + 1)

    all_preds = ext_preds + cnt_preds
    le_index = [all_preds.index(pred) for pred in le_pred]
    state = CountingState(
        ext_preds=tuple(ext_preds),
        cnt_preds=tuple(cnt_preds),
        cnt_params=tuple(cnt_params),
        cnt_remainder=tuple(cnt_remainder),
        exist_mod=bool(mod_pred_index),
        mod_pred_index=tuple(mod_pred_index),
        exist_le=bool(le_pred),
        le_index=tuple(le_index),
        binary_evidence=build_binary_evidence(tuple(ext_preds), tuple(cnt_preds)),
        c_type_shape=tuple(c_type_shape),
    )
    return state, masks


def build_binary_evidence(
    ext_preds: tuple[object, ...],
    cnt_preds: tuple[object, ...],
) -> tuple[frozenset, ...]:
    from wfomc.fol import FOLContext, Literal

    ctx = FOLContext()
    a = ctx.constant("a")
    b = ctx.constant("b")

    def lit(pred: object, left: object, right: object, positive: bool) -> Literal:
        return Literal(pred(left, right), positive)

    binary_atoms = [
        (
            (lit(pred, a, b, False), lit(pred, b, a, False)),
            (lit(pred, a, b, False), lit(pred, b, a, True)),
            (lit(pred, a, b, True), lit(pred, b, a, False)),
            (lit(pred, a, b, True), lit(pred, b, a, True)),
        )
        for pred in reversed(tuple(cnt_preds) + tuple(ext_preds))
    ]
    return tuple(frozenset(sum(atoms, start=())) for atoms in product(*binary_atoms))


def _counting_predicate(kind: str, body: object) -> Predicate:
    from wfomc.fol import Atom

    expected_arity = 1 if kind == "unary" else 2
    if not isinstance(body, Atom) or len(body.terms) != expected_arity:
        raise TypeError(
            f"{kind} counting section requires an atomic body with arity "
            f"{expected_arity}, got {body}"
        )
    return body.predicate


def _existential_predicates(
    normal_form: C2NormalForm,
) -> tuple[Predicate, ...]:
    from wfomc.fol import Atom, Quantifier, predicates

    found: list[Predicate] = []
    for formula in (*normal_form.forall_exists, *normal_form.exists):
        inner = formula
        while isinstance(inner, Quantifier):
            inner = inner.body
        candidates = (
            (inner.predicate,) if isinstance(inner, Atom) else tuple(predicates(inner))
        )
        for pred in sorted(
            candidates,
            key=lambda p: (p.name, p.arity),
        ):
            if pred.arity == 2 and pred not in found:
                found.append(pred)
    return tuple(found)


__all__ = [
    "CountingState",
    "UnaryCardinalityMasks",
    "build_binary_evidence",
    "build_counting_state_for_normal_form",
]
