"""
Propositional WFOMC: ground the (Skolemized) sentence over the domain and
hand the resulting weighted CNF to an external propositional counter.

For a universally quantified quantifier-free formula ``phi`` over the two
variables ``X, Y``, the model count over a domain ``D`` is, by definition,
the weighted model count of the grounding ``AND_{(a,b) in D*D} phi(a, b)``.
This module computes that directly as a textbook ground-truth baseline.

The linear-order axioms ``LEQ``, ``PRED`` (= ``PRED1``, immediate predecessor),
and ``CIRCULAR_PRED`` are handled in one of two ways, selected by the
module-level constant ``LINEAR_ORDER_ENCODING``:

* ``"pin"`` -- pin every ground atom of those predicates to its value under
  a canonical sorted order / cycle on the domain. ``decode_result``'s ``n!``
  multiplier then recovers the labeled-arrangement count over all linear
  orders (and over all rotations of all n-cycles, for circular). Cheap:
  ``O(n²)`` unit clauses per predicate and no auxiliary variables. Fast.

* ``"axioms"`` -- emit the FO³ axioms / definitions explicitly. ``LEQ`` is
  axiomatized as a total order (reflexivity, antisymmetry, totality, FO³
  transitivity); ``PRED1`` is defined as "Y is immediately below X in LEQ"
  with Tseitin auxiliaries for the inner ``∀Z`` conjunctions;
  ``CIRCULAR_PRED`` is defined as ``PRED1 ∨ (X is LEQ-min ∧ Y is LEQ-max)``.
  Because the axioms make ``LEQ`` range over all linear orders directly,
  the ``n!`` multiplier is skipped on decode. Pedagogically pure but
  materially heavier (~``O(n³)`` clauses), and notably slow under ganak's
  ``--mode 3`` polynomial weighting.

``PREDk`` for ``k > 1`` is rejected with a clear error in both encodings.

Unary evidence can be handled directly when the effective unary evidence
strategy is ``UnaryEvidenceStrategy.AUTO``; this module then emits one ground
unit clause per evidence atom. The solver resolves AUTO to CCS for the
pin-and-multiply linear-order path where direct element-specific evidence
would break the symmetry argument.
"""
from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Callable, Optional, Union

from loguru import logger
from sympy.logic.boolalg import And, BooleanFalse, BooleanTrue, Not, Or, to_cnf

from flint import fmpq_mpoly, fmpq_mpoly_ctx

from wfomc.context import UnaryEvidenceStrategy, WFOMCContext
from wfomc.fol import boolean_algebra as backend
from wfomc.fol.syntax import AtomicFormula, Bot, Const, Pred, Top, X, Y
from wfomc.utils import Rational, RingElement
from wfomc.utils.polynomial_flint import align_ctx

from .ganak import GanakError, ganak_count


# ---------------------------------------------------------------------------
# Encoding selector
# ---------------------------------------------------------------------------
class LinearOrderEncoding(Enum):
    """Choice of how to encode the order axioms (LEQ / PRED1 / CIRCULAR_PRED)
    in the propositional counter.

    * ``PIN`` -- pin every ground atom of those predicates to its value
      under a canonical sorted order/cycle on the domain; ``decode_result``'s
      ``n!`` multiplier recovers the full count. Cheap (``O(n²)`` unit
      clauses, ganak ``--mode 1`` when no other cardinality constraints).
    * ``AXIOMS`` -- emit the FO³ axioms / definitions explicitly. Pin-free
      and pedagogically pure, but materially heavier (``O(n³)`` clauses).
    """
    PIN = 'pin'
    AXIOMS = 'axioms'

    def __str__(self) -> str:
        return self.value


# Module-level default. Used when callers do not pass an explicit
# ``linear_order_encoding`` argument. Can be overridden globally by
# rebinding this constant, but the recommended way is via the
# ``linear_order_encoding`` parameter on :func:`propositional_wfomc` or
# :func:`wfomc.wfomc` (or the ``--linear-order-encoding`` CLI flag).
LINEAR_ORDER_ENCODING: LinearOrderEncoding = LinearOrderEncoding.PIN


def resolve_linear_order_encoding(
        encoding: Optional[Union[LinearOrderEncoding, str]] = None,
) -> LinearOrderEncoding:
    """Return the effective :class:`LinearOrderEncoding`.

    Accepts ``None`` (use the module default), a :class:`LinearOrderEncoding`
    instance, or its string value (``"pin"`` / ``"axioms"``). Raises
    :class:`GanakError` for any other input.
    """
    if encoding is None:
        return LINEAR_ORDER_ENCODING
    if isinstance(encoding, LinearOrderEncoding):
        return encoding
    if isinstance(encoding, str):
        try:
            return LinearOrderEncoding(encoding)
        except ValueError:
            pass
    raise GanakError(
        f'linear_order_encoding must be a LinearOrderEncoding or one of '
        f'{[e.value for e in LinearOrderEncoding]}; got {encoding!r}'
    )


# ---------------------------------------------------------------------------
# Grounding the formula into CNF
# ---------------------------------------------------------------------------
def _collect_clauses(formula, domain: list[Const]) -> tuple[
        dict[AtomicFormula, int], dict[int, Pred], list[frozenset[int]], bool]:
    """Ground ``formula`` over every ordered pair of domain elements.

    Returns ``(atom2id, id2pred, clauses, unsat)``.
    """
    atom2id: dict[AtomicFormula, int] = {}
    id2pred: dict[int, Pred] = {}

    def var_id(atom: AtomicFormula) -> int:
        key = atom.make_positive()
        if key not in atom2id:
            atom2id[key] = len(atom2id) + 1
            id2pred[atom2id[key]] = key.pred
        return atom2id[key]

    # Enumerate the whole vocabulary first so that every ground atom is a
    # counted variable -- even one that ends up in no clause is then a free
    # variable whose weight ganak still multiplies in.
    for pred in sorted(formula.preds(), key=lambda p: (p.name, p.arity)):
        for args in product(domain, repeat=pred.arity):
            var_id(pred(*args))

    clauses: set[frozenset[int]] = set()
    for a, b in product(domain, repeat=2):
        cnf = to_cnf(formula.substitute({X: a, Y: b}).expr)
        if isinstance(cnf, BooleanTrue):
            continue
        if isinstance(cnf, BooleanFalse):
            return atom2id, id2pred, [], True
        conjuncts = cnf.args if isinstance(cnf, And) else (cnf,)
        for conjunct in conjuncts:
            literals = conjunct.args if isinstance(conjunct, Or) else (conjunct,)
            clause: set[int] = set()
            tautology = False
            for literal in literals:
                if isinstance(literal, Not):
                    symbol, negated = literal.args[0], True
                else:
                    symbol, negated = literal, False
                signed = var_id(backend.get_atom(symbol))
                if negated:
                    signed = -signed
                if -signed in clause:
                    tautology = True
                    break
                clause.add(signed)
            if not tautology and clause:
                clauses.add(frozenset(clause))
    return atom2id, id2pred, list(clauses), False


# ---------------------------------------------------------------------------
# Path A: pin-and-multiply
# ---------------------------------------------------------------------------
def _pin_linear_order_atoms(
        context: WFOMCContext, domain: list[Const],
        atom2id: dict[AtomicFormula, int]) -> list[frozenset[int]]:
    """Pin LEQ / PRED1 / CIRCULAR_PRED ground atoms to a canonical order/cycle.

    For a sorted domain ``[a_0, ..., a_{n-1}]``:

    * ``LEQ(a_i, a_j)``           iff ``i <= j``
    * ``PRED1(a_i, a_j)``         iff ``j == i - 1``
    * ``CIRCULAR_PRED(a_i, a_j)`` iff ``j == (i - 1) mod n``

    ``WFOMCContext.decode_result`` multiplies the raw count by ``n!``, which
    recovers the labeled-arrangement count over all linear orders / cycles
    (the formula's domain-symmetry guarantees that any single pinning gives
    the same count as any other).
    """
    pinned: list[frozenset[int]] = []
    formula_preds = context.formula.preds()
    n = len(domain)
    cpred = context.circular_predecessor_pred

    def pin(atom: AtomicFormula, truth: bool) -> None:
        vid = atom2id.get(atom.make_positive())
        if vid is None:
            return
        pinned.append(frozenset([vid if truth else -vid]))

    if context.leq_pred is not None and context.leq_pred in formula_preds:
        for i, a in enumerate(domain):
            for j, b in enumerate(domain):
                pin(context.leq_pred(a, b), i <= j)

    for k, pred in (context.predecessor_preds or {}).items():
        if pred == cpred or pred not in formula_preds:
            continue
        for i, a in enumerate(domain):
            for j, b in enumerate(domain):
                pin(pred(a, b), j == i - k)  # k == 1 after validation

    if cpred is not None and cpred in formula_preds:
        for i, a in enumerate(domain):
            for j, b in enumerate(domain):
                pin(cpred(a, b), j == (i - 1) % n)

    return pinned


# ---------------------------------------------------------------------------
# Path B: FO³ axiomatization
# ---------------------------------------------------------------------------
def _ground_leq_axioms(domain: list[Const], leq: Pred,
                       atom2id: dict[AtomicFormula, int]) -> list[frozenset[int]]:
    """Reflexivity, antisymmetry, totality, transitivity (the FO³ axiom)."""
    clauses: list[frozenset[int]] = []
    n = len(domain)

    def vid(a: Const, b: Const) -> int:
        return atom2id[leq(a, b).make_positive()]

    for a in domain:
        clauses.append(frozenset([vid(a, a)]))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = domain[i], domain[j]
            clauses.append(frozenset([-vid(a, b), -vid(b, a)]))
            clauses.append(frozenset([vid(a, b), vid(b, a)]))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                a, b, c = domain[i], domain[j], domain[k]
                clauses.append(frozenset([-vid(a, b), -vid(b, c), vid(a, c)]))
    return clauses


def _ground_pred1_definition(domain: list[Const], pred1: Pred, leq: Pred,
                             atom2id: dict[AtomicFormula, int],
                             fresh: Callable[[], int]) -> list[frozenset[int]]:
    """``PRED1(X, Y) ↔ Y ≠ X ∧ LEQ(Y, X) ∧ ¬∃Z ∉ {X, Y}: LEQ(Y, Z) ∧ LEQ(Z, X)``.

    Tseitin auxiliaries ``M_{abc} ↔ LEQ(b, c) ∧ LEQ(c, a)`` keep the
    bidirectional implication in CNF without distributive blow-up.
    """
    clauses: list[frozenset[int]] = []

    def lvid(a: Const, b: Const) -> int:
        return atom2id[leq(a, b).make_positive()]

    def pvid(a: Const, b: Const) -> int:
        return atom2id[pred1(a, b).make_positive()]

    for a in domain:
        for b in domain:
            p = pvid(a, b)
            if a == b:
                clauses.append(frozenset([-p]))
                continue
            ba = lvid(b, a)
            clauses.append(frozenset([-p, ba]))
            m_aux: list[int] = []
            for c in domain:
                if c == a or c == b:
                    continue
                bc, ca = lvid(b, c), lvid(c, a)
                m = fresh()
                m_aux.append(m)
                clauses.append(frozenset([-m, bc]))
                clauses.append(frozenset([-m, ca]))
                clauses.append(frozenset([-bc, -ca, m]))
                clauses.append(frozenset([-p, -m]))
            clauses.append(frozenset([p, -ba] + m_aux))
    return clauses


def _ground_circular_pred_definition(
        domain: list[Const], circ: Pred, pred1: Pred, leq: Pred,
        atom2id: dict[AtomicFormula, int],
        fresh: Callable[[], int]) -> list[frozenset[int]]:
    """``CIRCULAR_PRED(X, Y) ↔ PRED1(X, Y) ∨ (MIN(X) ∧ MAX(Y))``."""
    clauses: list[frozenset[int]] = []

    def lvid(a: Const, b: Const) -> int:
        return atom2id[leq(a, b).make_positive()]

    def pvid(a: Const, b: Const) -> int:
        return atom2id[pred1(a, b).make_positive()]

    def cvid(a: Const, b: Const) -> int:
        return atom2id[circ(a, b).make_positive()]

    min_var = {a: fresh() for a in domain}
    max_var = {a: fresh() for a in domain}
    for a in domain:
        ma, Ma = min_var[a], max_var[a]
        neg_min, neg_max = [], []
        for z in domain:
            laz, lza = lvid(a, z), lvid(z, a)
            clauses.append(frozenset([-ma, laz]))
            clauses.append(frozenset([-Ma, lza]))
            neg_min.append(-laz)
            neg_max.append(-lza)
        clauses.append(frozenset([ma] + neg_min))
        clauses.append(frozenset([Ma] + neg_max))

    for a in domain:
        for b in domain:
            c, p = cvid(a, b), pvid(a, b)
            ma, Mb = min_var[a], max_var[b]
            clauses.append(frozenset([-c, p, ma]))
            clauses.append(frozenset([-c, p, Mb]))
            clauses.append(frozenset([-p, c]))
            clauses.append(frozenset([-ma, -Mb, c]))
    return clauses


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _validate_supported(context: WFOMCContext) -> None:
    """Reject features outside the propositional counter's scope."""
    cpred = context.circular_predecessor_pred
    for k, pred in (context.predecessor_preds or {}).items():
        if pred == cpred:
            continue
        if k != 1:
            raise GanakError(
                f'PRED{k} (k > 1) is not supported by the propositional '
                'counter; only PRED (= PRED1, immediate predecessor) and '
                'CIRCULAR_PRED are supported'
            )


def _decode_skipping_factorial(context: WFOMCContext,
                                res: RingElement) -> RingElement:
    """Decode without the ``n!`` multiplier (axioms mode already covers all
    linear orders)."""
    res = res / context.repeat_factor
    if context.contain_cardinality_constraint():
        res = context.cardinality_constraint.decode_poly(res)
    return res


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def propositional_wfomc(
        context: WFOMCContext,
        ganak_path: str = None,
        linear_order_encoding: Optional[Union[LinearOrderEncoding, str]] = None,
) -> RingElement:
    """Compute WFOMC by grounding the sentence and counting with ganak.

    Args:
        context: The WFOMC context (Skolemized formula, weights, constraints).
        ganak_path: Explicit path to the ganak binary, or None to auto-detect.
        linear_order_encoding: How to encode the linear-order axioms.
            Pass a :class:`LinearOrderEncoding` value (or its string form),
            or ``None`` to use the module default ``LINEAR_ORDER_ENCODING``.

    Returns:
        The fully decoded WFOMC value as a rational. Decoding is performed
        in-house; callers should not re-decode the result.

    Raises:
        GanakError: When the problem is out of scope, the encoding argument
            is invalid, or ganak fails.
    """
    encoding = resolve_linear_order_encoding(linear_order_encoding)
    _validate_supported(context)

    formula = context.formula
    domain = sorted(context.domain, key=lambda c: c.name)

    if isinstance(formula, Bot):
        logger.info('Formula is unsatisfiable; propositional WFOMC is 0')
        return context.decode_result(Rational(0, 1))
    if isinstance(formula, Top):
        logger.info('Formula is trivially true; propositional WFOMC is 1')
        return context.decode_result(Rational(1, 1))

    # Detect symbolic mode below from the actual weight types, so we cover
    # callers (e.g. cofola) that put polynomial weights on the context
    # directly without ever materialising a cardinality_constraint.
    symbolic = False
    npolyvars, poly_ctx = 0, None

    atom2id, id2pred, clauses, unsat = _collect_clauses(formula, domain)
    if unsat:
        logger.info('Grounding is unsatisfiable; propositional WFOMC is 0')
        return context.decode_result(Rational(0, 1))

    has_order = context.leq_pred is not None
    use_axioms = (encoding == LinearOrderEncoding.AXIOMS) and has_order

    next_id = [len(atom2id)]

    def fresh() -> int:
        next_id[0] += 1
        return next_id[0]

    # --- Unary evidence: direct unit-clause encoding ------------------------
    # Only valid when the solver has chosen the AUTO strategy.
    # Direct evidence is element-specific, which breaks the symmetry argument
    # behind pin-and-multiply for the linear-order axioms; the solver
    # therefore keeps CCS encoding in the ``pin`` + order-axiom case, and the
    # CCS path is handled by the standard ``context.formula`` + cardinality
    # constraint already populated by ``WFOMCContext._build``.
    if context.unary_evidence and \
            context.unary_evidence_strategy == UnaryEvidenceStrategy.AUTO:
        evidence_preds = sorted(
            {atom.pred for atom in context.unary_evidence},
            key=lambda pred: (pred.name, pred.arity),
        )
        for pred in evidence_preds:
            for const in domain:
                key = pred(const).make_positive()
                if key not in atom2id:
                    vid = fresh()
                    atom2id[key] = vid
                    id2pred[vid] = pred
        added = 0
        for atom in sorted(context.unary_evidence,
                            key=lambda a: (a.pred.name, str(a.args), a.positive)):
            key = atom.make_positive()
            vid = atom2id[key]
            clauses.append(frozenset([vid if atom.positive else -vid]))
            added += 1
        logger.info('Added {} unary-evidence unit clauses', added)

    if encoding == LinearOrderEncoding.PIN and has_order:
        # Cheap path: pin atoms to a canonical order/cycle and rely on the
        # n! multiplier already applied by decode_result.
        clauses = clauses + _pin_linear_order_atoms(context, domain, atom2id)

    elif use_axioms:
        # FO³ axiomatization path.
        leq = context.leq_pred
        pred1 = Pred('PRED1', 2)
        circ = context.circular_predecessor_pred
        # PRED1 ground atoms are needed if PRED1 is in the formula or if
        # CIRCULAR_PRED is in use (its definition references PRED1).
        need_pred1 = (pred1 in id2pred.values()) or (circ is not None)

        for pred in [leq] + ([pred1] if need_pred1 else []):
            for a in domain:
                for b in domain:
                    atom = pred(a, b).make_positive()
                    if atom not in atom2id:
                        vid = fresh()
                        atom2id[atom] = vid
                        id2pred[vid] = pred

        clauses = clauses + _ground_leq_axioms(domain, leq, atom2id)
        if need_pred1:
            clauses = clauses + _ground_pred1_definition(
                domain, pred1, leq, atom2id, fresh,
            )
        if circ is not None:
            clauses = clauses + _ground_circular_pred_definition(
                domain, circ, pred1, leq, atom2id, fresh,
            )

    n_vars = next_id[0]
    # Use _get_weight (FLINT RingElement) instead of get_weight (sympy Expr).
    # ganak's mode-3 polynomial parser needs FLINT-printable input; sympy's
    # ``**`` power notation is rejected. Lifted algos already use this side
    # of the API.
    weights = {vid: context._get_weight(pred) for vid, pred in id2pred.items()}

    # Symbolic iff any weight is a non-constant FLINT mpoly. We collect every
    # polynomial weight, align them to a common context (FLINT mpoly
    # operations require shared contexts), and re-key the weights dict.
    poly_weights = [
        w for pos, neg in weights.values() for w in (pos, neg)
        if isinstance(w, fmpq_mpoly)
    ]
    if poly_weights:
        symbolic = True
        aligned = align_ctx(poly_weights)
        poly_ctx = aligned[0].context()
        npolyvars = poly_ctx.nvars()
        # Re-project every polynomial weight onto the common context.
        weights = {
            vid: (
                pos.project_to_context(poly_ctx) if isinstance(pos, fmpq_mpoly) else pos,
                neg.project_to_context(poly_ctx) if isinstance(neg, fmpq_mpoly) else neg,
            )
            for vid, (pos, neg) in weights.items()
        }

    logger.info(
        'Propositional WFOMC: {} vars, {} clauses, {} weights, {} encoding',
        n_vars, len(clauses), 'polynomial' if symbolic else 'rational',
        f'{encoding}' + (' (active)' if has_order else ' (no-op)'),
    )

    raw = ganak_count(
        n_vars, clauses, weights,
        symbolic=symbolic, npolyvars=npolyvars, poly_ctx=poly_ctx,
        ganak_path=ganak_path,
    )

    # Decode: in axioms mode, the n! multiplier has already been "absorbed"
    # by the axioms ranging LEQ over all orders, so skip decode_result's
    # factorial. In pin mode, decode_result applies the n! correctly.
    if use_axioms:
        return _decode_skipping_factorial(context, raw)
    return context.decode_result(raw)
