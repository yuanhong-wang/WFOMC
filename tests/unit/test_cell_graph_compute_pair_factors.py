from __future__ import annotations

import importlib

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cell_graph.data import Cell
from wfomc.cell_graph.compute_pair_factors import compute_pair_factors
from wfomc.errors import GanakError
from wfomc.fol import FOLContext, model_literals
from wfomc.fol.cnf import encode_tseitin


pair_factor_module = importlib.import_module(
    "wfomc.cell_graph.compute_pair_factors"
)


def test_exact_sdd_scalar_pair_factors_match_native_models():
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    p = ctx.predicate("P", 1)
    relation = ctx.predicate("R", 2)
    formula = (p(first) | relation(first, second)) & (
        p(second) | relation(second, first)
    )
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    weights = {
        p: (arithmetic.from_int(101), arithmetic.from_int(103)),
        relation: (arithmetic.from_int(2), arithmetic.from_int(3)),
    }
    cells = (Cell((False,), (p,)), Cell((True,), (p,)))

    matrix = compute_pair_factors(
        encode_tseitin(formula),
        cells,
        weights,
        arithmetic,
    )

    for left_index, left in enumerate(cells):
        for right_index, right in enumerate(cells):
            expected = _native_weight(
                formula,
                p,
                relation,
                left.code[0],
                right.code[0],
                arithmetic,
            )
            assert matrix[left_index][right_index].total_weight == expected
            assert matrix[left_index][right_index].counting_weights == ()


def test_exact_sdd_counting_factor_uses_incremental3_bit_order():
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    relation = ctx.predicate("R", 2)
    forward = relation(first, second)
    reverse = relation(second, first)
    formula = (forward | ~forward) & (reverse | ~reverse)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    cells = (Cell((), ()),)

    factor = compute_pair_factors(
        encode_tseitin(formula),
        cells,
        {relation: (arithmetic.from_int(2), arithmetic.from_int(3))},
        arithmetic,
        projected_binary_preds=(relation,),
    )[0][0]

    assert factor.total_weight == arithmetic.from_int(25)
    assert factor.counting_weights == (
        (0, arithmetic.from_int(9)),
        (1, arithmetic.from_int(6)),  # R(b,a)
        (2, arithmetic.from_int(6)),  # R(a,b)
        (3, arithmetic.from_int(4)),
    )


def test_exact_sdd_pair_factor_uses_symbolic_arithmetic_context():
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    relation = ctx.predicate("R", 2)
    forward = relation(first, second)
    reverse = relation(second, first)
    formula = (forward | ~forward) & (reverse | ~reverse)
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_POLY,
        symbolic_variables=("w",),
    )
    symbol = arithmetic.symbol("w")

    factor = compute_pair_factors(
        encode_tseitin(formula),
        (Cell((), ()),),
        {relation: (symbol, arithmetic.one())},
        arithmetic,
    )[0][0]

    assert factor.total_weight == (symbol + arithmetic.one()) ** 2


def test_missing_offdiagonal_atoms_contribute_free_symbolic_weights():
    ctx = FOLContext()
    relation = ctx.predicate("R", 2)
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_POLY,
        symbolic_variables=("w",),
    )
    symbol = arithmetic.symbol("w")

    factor = compute_pair_factors(
        encode_tseitin(ctx.true()),
        (Cell((False,), (relation,)),),
        {relation: (symbol, arithmetic.one())},
        arithmetic,
    )[0][0]

    assert factor.total_weight == (symbol + arithmetic.one()) ** 2
    assert factor.counting_weights == ()


def test_missing_projected_atoms_expand_incremental3_masks():
    ctx = FOLContext()
    relation = ctx.predicate("R", 2)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)

    factor = compute_pair_factors(
        encode_tseitin(ctx.true()),
        (Cell((False,), (relation,)),),
        {relation: (arithmetic.from_int(2), arithmetic.from_int(3))},
        arithmetic,
        projected_binary_preds=(relation,),
    )[0][0]

    assert factor.total_weight == arithmetic.from_int(25)
    assert factor.counting_weights == (
        (0, arithmetic.from_int(9)),
        (1, arithmetic.from_int(6)),  # R(b,a)
        (2, arithmetic.from_int(6)),  # R(a,b)
        (3, arithmetic.from_int(4)),
    )


def test_small_pair_formula_uses_bounded_pysat_before_sdd(monkeypatch):
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    relation = ctx.predicate("R", 2)
    formula = relation(first, second) | ~relation(first, second)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)

    def fail_if_compiled(*_args, **_kwargs):
        raise AssertionError("small pair formula should not compile an SDD")

    monkeypatch.setattr(pair_factor_module, "_SddCircuit", fail_if_compiled)

    factor = compute_pair_factors(
        encode_tseitin(formula),
        (Cell((False,), (relation,)),),
        {relation: (arithmetic.from_int(2), arithmetic.from_int(3))},
        arithmetic,
    )[0][0]

    assert factor.total_weight == arithmetic.from_int(25)


def test_pair_factor_uses_ganak_when_pysat_budget_is_exhausted(monkeypatch):
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    relation = ctx.predicate("R", 2)
    formula = relation(first, second) | ~relation(first, second)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    calls = []

    def fake_ganak(*_args, **_kwargs):
        calls.append(True)
        return {0: arithmetic.from_int(5)}

    def fail_if_compiled(*_args, **_kwargs):
        raise AssertionError("successful Ganak must avoid PySDD")

    monkeypatch.setattr(pair_factor_module, "_PYSAT_MODEL_LIMIT", 0)
    monkeypatch.setattr(
        pair_factor_module,
        "compute_factor_with_ganak",
        fake_ganak,
    )
    monkeypatch.setattr(pair_factor_module, "_SddCircuit", fail_if_compiled)

    factor = compute_pair_factors(
        encode_tseitin(formula),
        (Cell((False,), (relation,)),),
        {relation: (arithmetic.from_int(2), arithmetic.from_int(3))},
        arithmetic,
    )[0][0]

    assert calls == [True]
    assert factor.total_weight == arithmetic.from_int(25)


def test_pair_factor_falls_back_to_sdd_when_ganak_is_unavailable(
    monkeypatch,
):
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    relation = ctx.predicate("R", 2)
    formula = relation(first, second) | ~relation(first, second)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    original_circuit = pair_factor_module._SddCircuit
    compiled = []

    def tracking_circuit(*args, **kwargs):
        compiled.append(True)
        return original_circuit(*args, **kwargs)

    def unavailable_ganak(*_args, **_kwargs):
        raise GanakError("not installed")

    monkeypatch.setattr(pair_factor_module, "_PYSAT_MODEL_LIMIT", 0)
    monkeypatch.setattr(
        pair_factor_module,
        "compute_factor_with_ganak",
        unavailable_ganak,
    )
    monkeypatch.setattr(pair_factor_module, "_SddCircuit", tracking_circuit)

    factor = compute_pair_factors(
        encode_tseitin(formula),
        (Cell((False,), (relation,)),),
        {relation: (arithmetic.from_int(2), arithmetic.from_int(3))},
        arithmetic,
    )[0][0]

    assert compiled == [True]
    assert factor.total_weight == arithmetic.from_int(25)


def test_equivalent_cell_pairs_call_ganak_once_and_reuse_factor(monkeypatch):
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    free_cell_predicate = ctx.predicate("Free", 1)
    relation = ctx.predicate("R", 2)
    formula = relation(first, second) | ~relation(first, second)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    cells = (
        Cell((False, False), (free_cell_predicate, relation)),
        Cell((True, False), (free_cell_predicate, relation)),
    )
    calls = []

    def fake_ganak(*_args, **_kwargs):
        calls.append(True)
        return {0: arithmetic.from_int(5)}

    monkeypatch.setattr(pair_factor_module, "_PYSAT_MODEL_LIMIT", 0)
    monkeypatch.setattr(
        pair_factor_module,
        "compute_factor_with_ganak",
        fake_ganak,
    )

    matrix = compute_pair_factors(
        encode_tseitin(formula),
        cells,
        {relation: (arithmetic.from_int(2), arithmetic.from_int(3))},
        arithmetic,
    )

    first_factor = matrix[0][0]
    assert calls == [True]
    assert first_factor.total_weight == arithmetic.from_int(25)
    assert all(factor is first_factor for row in matrix for factor in row)


def test_equivalent_cell_pairs_reuse_one_pair_factor_instance():
    ctx = FOLContext()
    first, second = ctx.constants("a b")
    free_cell_predicate = ctx.predicate("Free", 1)
    relation = ctx.predicate("R", 2)
    formula = relation(first, second) | ~relation(first, second)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    cells = (
        Cell((False,), (free_cell_predicate,)),
        Cell((True,), (free_cell_predicate,)),
    )

    matrix = compute_pair_factors(
        encode_tseitin(formula),
        cells,
        {relation: (arithmetic.from_int(2), arithmetic.from_int(3))},
        arithmetic,
    )

    first_factor = matrix[0][0]
    assert all(factor is first_factor for row in matrix for factor in row)


def _native_weight(
    formula,
    unary,
    relation,
    left_value: bool,
    right_value: bool,
    arithmetic: ArithmeticContext,
):
    first, second = FOLContext().constants("a b")
    result = arithmetic.zero()
    for model in model_literals(formula):
        truth = {literal.atom: literal.positive for literal in model}
        if truth[unary(first)] != left_value or truth[unary(second)] != right_value:
            continue
        weight = arithmetic.one()
        for atom in (relation(first, second), relation(second, first)):
            weight *= arithmetic.from_int(2 if truth[atom] else 3)
        result += weight
    return result
