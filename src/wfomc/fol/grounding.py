"""Finite-domain grounding helpers for typed FOL formulas."""

from __future__ import annotations

from enum import Enum
from itertools import product
from typing import Callable, Optional, Union

from wfomc.fol.cnf import encode_tseitin
from wfomc.fol.syntax import Atom, Constant, Formula, Predicate, Variable
from wfomc.fol.analysis import free_vars, is_quantifier_free, predicates
from wfomc.fol.rewrite import simplify_boolean, substitute


# ---------------------------------------------------------------------------
# Encoding choice for order predicates (pin-and-multiply vs FO^3 axioms)
# ---------------------------------------------------------------------------
class LinearOrderEncoding(Enum):
    """Choice of how to encode order predicates in the propositional counter."""

    PIN = "pin"
    AXIOMS = "axioms"

    def __str__(self) -> str:
        return self.value


def resolve_linear_order_encoding(
    encoding: Optional[Union[LinearOrderEncoding, str]] = None,
) -> LinearOrderEncoding:
    if encoding is None:
        return LinearOrderEncoding.PIN
    if isinstance(encoding, LinearOrderEncoding):
        return encoding
    if isinstance(encoding, str):
        try:
            return LinearOrderEncoding(encoding)
        except ValueError:
            pass
    raise ValueError(
        f"linear_order_encoding must be a LinearOrderEncoding or one of "
        f"{[e.value for e in LinearOrderEncoding]}; got {encoding!r}"
    )


X = Variable("X")
Y = Variable("Y")


def ground_on_tuple(
    formula: Formula,
    first: object,
    second: object | None = None,
) -> Formula:
    """Substitute the free variables of a QF formula with one constant tuple."""

    if not is_quantifier_free(formula):
        raise ValueError("ground_on_tuple requires a quantifier-free formula")

    variables = tuple(sorted(free_vars(formula), key=lambda variable: variable.name))
    if len(variables) > 2:
        raise RuntimeError("Can only ground out FO2")
    constants = (
        [first] if len(variables) == 1 else [first, first if second is None else second]
    )
    grounded = substitute(formula, dict(zip(variables, constants)))
    if isinstance(grounded, Formula):
        grounded = simplify_boolean(grounded)
    return grounded


def ground_qf_formula(
    formula: Formula, domain: list[Constant]
) -> tuple[
    dict[Atom, int],
    dict[int, Predicate],
    list[frozenset[int]],
    bool,
]:
    """Instantiate a QF formula over a finite domain and return ground CNF data.

    The result is ``(atom_to_id, id_to_predicate, clauses, unsatisfiable)``.
    Quantifier elimination and FO2 normalization belong to earlier reduction
    stages; this function only performs propositional grounding.
    """

    if not is_quantifier_free(formula):
        raise ValueError("ground_qf_formula requires a quantifier-free formula")
    atom2id: dict[Atom, int] = {}
    id2pred: dict[int, Predicate] = {}
    next_aux = [0]

    def var_id(atom: Atom) -> int:
        positive = atom.make_positive()
        if positive not in atom2id:
            next_aux[0] += 1
            atom2id[positive] = next_aux[0]
            id2pred[atom2id[positive]] = positive.predicate
        return atom2id[positive]

    def aux_id(local_id: int, aux_map: dict[int, int]) -> int:
        if local_id not in aux_map:
            next_aux[0] += 1
            aux_map[local_id] = next_aux[0]
        return aux_map[local_id]

    for pred in sorted(predicates(formula), key=lambda p: (p.name, p.arity)):
        for args in product(domain, repeat=pred.arity):
            var_id(pred(*args))

    clauses: set[frozenset[int]] = set()
    for a, b in product(domain, repeat=2):
        grounded = ground_on_tuple(formula, a, b)
        encoding = encode_tseitin(grounded)
        atoms = encoding.atoms
        local_clauses = encoding.clauses
        aux_map: dict[int, int] = {}
        for clause in local_clauses:
            mapped = []
            for lit in clause:
                sign = 1 if lit > 0 else -1
                local = abs(lit)
                if local <= len(atoms):
                    mapped.append(sign * var_id(atoms[local - 1]))
                else:
                    mapped.append(sign * aux_id(local, aux_map))
            if not mapped:
                return atom2id, id2pred, [], True
            clauses.add(frozenset(mapped))
    return atom2id, id2pred, list(clauses), False


# ---------------------------------------------------------------------------
# Path A: pin-and-multiply
# ---------------------------------------------------------------------------
def pin_linear_order_atoms(
    context: object,
    domain: list[Constant],
    atom2id: dict[Atom, int],
) -> list[frozenset[int]]:
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

    def pin(atom: Atom, truth: bool) -> None:
        vid = atom2id.get(_positive_atom(atom))
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
def ground_leq_axioms(
    domain: list[Constant],
    leq: Predicate,
    atom2id: dict[Atom, int],
) -> list[frozenset[int]]:
    """Reflexivity, antisymmetry, totality, transitivity (the FO³ axiom)."""
    clauses: list[frozenset[int]] = []
    n = len(domain)

    def vid(a: Constant, b: Constant) -> int:
        return atom2id[_positive_atom(leq(a, b))]

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


def ground_pred1_definition(
    domain: list[Constant],
    pred1: Predicate,
    leq: Predicate,
    atom2id: dict[Atom, int],
    fresh: Callable[[], int],
) -> list[frozenset[int]]:
    """``PRED1(X, Y) ↔ Y ≠ X ∧ LEQ(Y, X) ∧ ¬∃Z ∉ {X, Y}: LEQ(Y, Z) ∧ LEQ(Z, X)``.

    Tseitin auxiliaries ``M_{abc} ↔ LEQ(b, c) ∧ LEQ(c, a)`` keep the
    bidirectional implication in CNF without distributive blow-up.
    """
    clauses: list[frozenset[int]] = []

    def lvid(a: Constant, b: Constant) -> int:
        return atom2id[_positive_atom(leq(a, b))]

    def pvid(a: Constant, b: Constant) -> int:
        return atom2id[_positive_atom(pred1(a, b))]

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


def ground_circular_pred_definition(
    domain: list[Constant],
    circ: Predicate,
    pred1: Predicate,
    leq: Predicate,
    atom2id: dict[Atom, int],
    fresh: Callable[[], int],
) -> list[frozenset[int]]:
    """``CIRCULAR_PRED(X, Y) ↔ PRED1(X, Y) ∨ (MIN(X) ∧ MAX(Y))``."""
    clauses: list[frozenset[int]] = []

    def lvid(a: Constant, b: Constant) -> int:
        return atom2id[_positive_atom(leq(a, b))]

    def pvid(a: Constant, b: Constant) -> int:
        return atom2id[_positive_atom(pred1(a, b))]

    def cvid(a: Constant, b: Constant) -> int:
        return atom2id[_positive_atom(circ(a, b))]

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


def _positive_atom(atom: Atom) -> Atom:
    return atom.make_positive()


# ---------------------------------------------------------------------------
# Dispatch: pin-and-multiply (Path A) vs FO^3 axiomatization (Path B)
# ---------------------------------------------------------------------------
def linear_order_clauses(
    context: object,
    domain: list[Constant],
    atom_to_id: dict[Atom, int],
    id_to_predicate: dict[int, Predicate],
    fresh: Callable[[], int],
    encoding: LinearOrderEncoding,
) -> list[frozenset[int]]:
    """Build ground CNF clauses encoding the linear-order / cycle structure.

    Under :attr:`LinearOrderEncoding.PIN` the order atoms are pinned to a
    canonical arrangement and the raw count is multiplied by ``n!`` at decode
    time. Under :attr:`LinearOrderEncoding.AXIOMS` the LEQ / PRED1 /
    CIRCULAR_PRED relations are axiomatized in FO^3 so the counter enumerates
    every arrangement directly.
    """
    if context.leq_pred is None:
        return []
    if encoding == LinearOrderEncoding.PIN:
        return pin_linear_order_atoms(context, domain, atom_to_id)
    if encoding != LinearOrderEncoding.AXIOMS:
        return []

    leq = context.leq_pred
    pred1 = Predicate("PRED1", 2)
    circ = context.circular_predecessor_pred
    need_pred1 = pred1 in id_to_predicate.values() or circ is not None

    for pred in [leq] + ([pred1] if need_pred1 else []):
        for a in domain:
            for b in domain:
                atom = _positive_atom(pred(a, b))
                if atom not in atom_to_id:
                    vid = fresh()
                    atom_to_id[atom] = vid
                    id_to_predicate[vid] = pred

    order_clauses = list(ground_leq_axioms(domain, leq, atom_to_id))
    if need_pred1:
        order_clauses.extend(
            ground_pred1_definition(domain, pred1, leq, atom_to_id, fresh)
        )
    if circ is not None:
        order_clauses.extend(
            ground_circular_pred_definition(domain, circ, pred1, leq, atom_to_id, fresh)
        )
    return order_clauses


__all__ = [
    "LinearOrderEncoding",
    "ground_circular_pred_definition",
    "ground_leq_axioms",
    "ground_pred1_definition",
    "ground_qf_formula",
    "linear_order_clauses",
    "pin_linear_order_atoms",
    "resolve_linear_order_encoding",
]
