from __future__ import annotations
import math
from dataclasses import dataclass
from itertools import product

from loguru import logger

from wfomc.fol.syntax import *
from wfomc.fol import tseitin_transform
from wfomc.problems import WFOMCProblem
from wfomc.utils import RingElement

from .wfomc_context import WFOMCContext
from .unary_cardinality import UnaryCardinalityConstraintHandler
from .unary_evidence import UnaryEvidenceStrategy


@dataclass
class CountingState:
    """Counting-quantifier state produced during IncrementalWFOMC3Context._build()."""
    ext_preds: list[Pred]
    cnt_preds: list[Pred]
    cnt_params: list[int]
    cnt_remainder: list
    exist_mod: bool
    mod_pred_index: list[int]
    exist_le: bool
    le_index: list[int]
    binary_evidence: list[frozenset]
    c_type_shape: list[int]


def _build_binary_evidence(ext_preds: list[Pred], cnt_preds: list[Pred]) -> list[frozenset]:
    """Generate all truth assignments for binary predicates between elements a and b."""
    ext_atoms = list(
        (
            (~pred(a, b), ~pred(b, a)),
            (~pred(a, b), pred(b, a)),
            (pred(a, b), ~pred(b, a)),
            (pred(a, b), pred(b, a)),
        )
        for pred in ext_preds[::-1]
    )
    cnt_atoms = list(
        (
            (~pred(a, b), ~pred(b, a)),
            (~pred(a, b), pred(b, a)),
            (pred(a, b), ~pred(b, a)),
            (pred(a, b), pred(b, a)),
        )
        for pred in cnt_preds[::-1]
    )
    evidence = []
    for atoms in product(*cnt_atoms, *ext_atoms):
        evidence.append(frozenset(sum(atoms, start=())))
    return evidence


class IncrementalWFOMC3Context(WFOMCContext):
    def __init__(
        self,
        problem: WFOMCProblem,
        unary_evidence_strategy: UnaryEvidenceStrategy = UnaryEvidenceStrategy.AUTO,
    ):
        # Initialise IncrementalWFOMC3-specific mutable state before calling
        # super().__init__(), because _build() is dispatched from within
        # WFOMCContext.__init__ and needs these to be ready.
        self.unary_handler = UnaryCardinalityConstraintHandler()

        # Counting-quantifier parsing state (populated by _handle_* during _build)
        self._ext_preds: list[Pred] = []
        self._cnt_preds: list[Pred] = []
        self._cnt_params: list[int] = []
        self._cnt_remainder: list = []
        self._mod_pred_index: list[int] = []
        self._exist_mod: bool = False
        self._exist_le: bool = False
        self._le_pred: list[Pred] = []
        self._comparator_handlers = {
            "mod": self._handle_mod,
            "=": self._handle_eq,
            "<=": self._handle_le,
        }

        # Exposed after _build() completes
        self.counting_state: CountingState | None = None

        super().__init__(
            problem,
            unary_evidence_strategy,
        )

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def decode_result(self, res: RingElement) -> RingElement:
        if self.leq_pred is not None:
            res = res * math.factorial(len(self.domain))
        res = res / self.repeat_factor
        if res == 0:
            return res
        if self.contain_cardinality_constraint():
            res = self.cardinality_constraint.decode_poly(res)
        return res

    # ------------------------------------------------------------------
    # _build and helpers
    # ------------------------------------------------------------------

    def _build(self):
        self.sentence = tseitin_transform(self.sentence)
        logger.info('sentence after tseitin transform: \n{}', self.sentence)

        self.formula = self.sentence.uni_formula
        while not isinstance(self.formula, QFFormula):
            self.formula = self.formula.quantified_formula

        if self.unary_evidence:
            self._apply_unary_evidence()

        if self.sentence.contain_counting_quantifier():
            self._handle_counting_quantifier()

        for ext_formula in self.sentence.ext_formulas:
            while isinstance(ext_formula, QuantifiedFormula):
                ext_formula = ext_formula.quantified_formula
            self._ext_preds.append(ext_formula.pred)

        self._prepare_cardinality_weights()

        self._handle_linear_order_axiom()

        c_type_shape = [2 for _ in self._ext_preds]
        for idx, k in enumerate(self._cnt_params):
            c_type_shape.append(k if idx in self._mod_pred_index else k + 1)

        all_preds = self._ext_preds + self._cnt_preds
        le_index = [all_preds.index(pred) for pred in self._le_pred]

        self.counting_state = CountingState(
            ext_preds=self._ext_preds,
            cnt_preds=self._cnt_preds,
            cnt_params=self._cnt_params,
            cnt_remainder=self._cnt_remainder,
            exist_mod=self._exist_mod,
            mod_pred_index=self._mod_pred_index,
            exist_le=self._exist_le,
            le_index=le_index,
            binary_evidence=_build_binary_evidence(self._ext_preds, self._cnt_preds),
            c_type_shape=c_type_shape,
        )

        self._convert_weights_to_ring_elements()

    def _handle_counting_quantifier(self):
        for formula in self.sentence.cnt_formulas:
            if isinstance(formula.quantified_formula, QuantifiedFormula):
                inner = formula.quantified_formula
                kind = "binary"
                inner_formula = inner.quantified_formula
                qscope = inner.quantifier_scope
            else:
                kind = "unary"
                inner_formula = formula.quantified_formula
                qscope = formula.quantifier_scope

            comparator = qscope.comparator
            cnt_param_raw = qscope.count_param
            idx = len(self._cnt_preds)
            if comparator not in self._comparator_handlers:
                raise ValueError(
                    f"Unsupported comparator '{comparator}'. "
                    f"Supported: {list(self._comparator_handlers.keys())}"
                )
            self._comparator_handlers[comparator](
                kind, idx, inner_formula, qscope, cnt_param_raw, comparator
            )

    def _get_counting_predicate(self, kind: str, inner_formula: Formula) -> Pred:
        expected_arity = 1 if kind == "unary" else 2
        if not (
            isinstance(inner_formula, AtomicFormula) and
            inner_formula.pred.arity == expected_arity
        ):
            raise TypeError(
                f"{kind.capitalize()} counting quantifier requires a "
                f"{'unary' if kind == 'unary' else 'binary'} atomic formula "
                f"inside, but got {inner_formula}"
            )
        return inner_formula.pred

    def _handle_mod(self, kind, idx, inner_formula, qscope, param, _):
        """Handle ∃_{≡r (mod k)}."""
        r, k = param
        pred = self._get_counting_predicate(kind, inner_formula)
        if kind == "unary":
            self.unary_handler.add_mod(pred, r, k)
        else:
            self._exist_mod = True
            self._mod_pred_index.append(idx)
            self._cnt_remainder.append(r)
            self._cnt_params.append(k)
            self._cnt_preds.append(pred)

    def _handle_eq(self, kind, idx, inner_formula, qscope, param, _):
        """Handle ∃_{=k}."""
        pred = self._get_counting_predicate(kind, inner_formula)
        if kind == "unary":
            self.unary_handler.add_eq(pred, param)
        else:
            self._cnt_remainder.append(None)
            self._cnt_params.append(param)
            self._cnt_preds.append(pred)

    def _handle_le(self, kind, idx, inner_formula, qscope, param, _):
        """Handle ∃_{≤k}."""
        pred = self._get_counting_predicate(kind, inner_formula)
        if kind == "unary":
            self.unary_handler.add_le(pred, param)
        else:
            self._cnt_remainder.append(None)
            self._cnt_params.append(param)
            self._cnt_preds.append(pred)
            self._le_pred.append(pred)
            self._exist_le = True
