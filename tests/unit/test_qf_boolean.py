"""Correctness gates for typed quantifier-free Boolean formulas."""

from __future__ import annotations

import random
from itertools import product

from wfomc.fol import And, Atom, BoolConst, Iff, Implies, Not, Or
from wfomc.fol import is_satisfiable, models
from wfomc.fol import Variable, atom, conjunction, disjunction, false, true


X = Variable("X")
Y = Variable("Y")
ATOMS = (
    atom("P", X),
    atom("Q", X),
    atom("R", X),
    atom("S", X),
    atom("E", X, Y),
)


def _random_spec(depth: int, rng: random.Random):
    if depth <= 0 or rng.random() < 0.35:
        r = rng.random()
        if r < 0.08:
            return ("const", True)
        if r < 0.16:
            return ("const", False)
        return ("atom", rng.randrange(len(ATOMS)))
    op = rng.choice(["not", "and", "or", "implies", "iff"])
    if op == "not":
        return ("not", _random_spec(depth - 1, rng))
    if op in {"implies", "iff"}:
        return (op, _random_spec(depth - 1, rng), _random_spec(depth - 1, rng))
    return (op, [_random_spec(depth - 1, rng) for _ in range(rng.randint(2, 3))])


def _realize(spec):
    kind = spec[0]
    if kind == "const":
        return true() if spec[1] else false()
    if kind == "atom":
        return ATOMS[spec[1]]
    if kind == "not":
        return ~_realize(spec[1])
    if kind == "implies":
        return _realize(spec[1]).implies(_realize(spec[2]))
    if kind == "iff":
        return _realize(spec[1]).equivalent(_realize(spec[2]))
    if kind == "and":
        return conjunction(*(_realize(s) for s in spec[1]))
    if kind == "or":
        return disjunction(*(_realize(s) for s in spec[1]))
    raise AssertionError(spec)


def _formula_atoms(formula) -> tuple[Atom, ...]:
    seen: dict[Atom, None] = {}

    def walk(node) -> None:
        if isinstance(node, Atom):
            seen.setdefault(node, None)
        elif isinstance(node, Not):
            walk(node.body)
        elif isinstance(node, (And, Or)):
            for arg in node.args:
                walk(arg)
        elif isinstance(node, (Implies, Iff)):
            walk(node.left)
            walk(node.right)

    walk(formula)
    return tuple(sorted(seen, key=str))


def _evaluate(formula, assignment: dict[Atom, bool]) -> bool:
    if isinstance(formula, Atom):
        return assignment[formula]
    if isinstance(formula, BoolConst):
        return formula.value
    if isinstance(formula, Not):
        return not _evaluate(formula.body, assignment)
    if isinstance(formula, And):
        return all(_evaluate(arg, assignment) for arg in formula.args)
    if isinstance(formula, Or):
        return any(_evaluate(arg, assignment) for arg in formula.args)
    if isinstance(formula, Implies):
        return (not _evaluate(formula.left, assignment)) or _evaluate(
            formula.right,
            assignment,
        )
    if isinstance(formula, Iff):
        return _evaluate(formula.left, assignment) == _evaluate(
            formula.right,
            assignment,
        )
    raise AssertionError(formula)


def _brute_models(formula) -> frozenset[frozenset[tuple[str, bool]]]:
    formula_atoms = _formula_atoms(formula)
    out = set()
    for values in product((False, True), repeat=len(formula_atoms)):
        assignment = dict(zip(formula_atoms, values))
        if _evaluate(formula, assignment):
            out.add(
                frozenset(
                    (str(atom_), value) for atom_, value in assignment.items()
                )
            )
    return frozenset(out)


def _qf_models(formula) -> frozenset[frozenset[tuple[str, bool]]]:
    return frozenset(
        frozenset((str(atom_), value) for atom_, value in assignment.items())
        for assignment in models(formula)
    )


def test_typed_qf_model_enumeration_matches_brute_force():
    for seed in range(30):
        rng = random.Random(seed)
        for _ in range(30):
            spec = _random_spec(rng.randint(1, 4), rng)
            formula = _realize(spec)
            brute = _brute_models(formula)

            assert is_satisfiable(formula) == bool(brute), spec
            assert _qf_models(formula) == brute, spec


def test_atomless_true_formula_has_one_empty_model():
    assert list(models(true())) == [{}]


def test_unsatisfiable_formula_has_no_models():
    p = atom("RegressionP", X)

    assert list(models(p & ~p)) == []


def test_atomic_and_tautological_models_keep_both_polarities():
    p = atom("RegressionP", X)

    assert _qf_models(p) == frozenset({frozenset({("RegressionP(X)", True)})})
    assert _qf_models(p | ~p) == frozenset(
        {
            frozenset({("RegressionP(X)", False)}),
            frozenset({("RegressionP(X)", True)}),
        }
    )


def test_implication_and_equivalence_models():
    p = atom("RegressionP", X)
    q = atom("RegressionQ", X)

    assert _qf_models(p.implies(q)) == _brute_models(p.implies(q))
    assert _qf_models(p.equivalent(q)) == _brute_models(p.equivalent(q))
