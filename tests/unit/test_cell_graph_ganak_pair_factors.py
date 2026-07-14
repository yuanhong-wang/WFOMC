from __future__ import annotations

import importlib

from flint import fmpq

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.fol import FOLContext
from wfomc.fol.cnf import encode_tseitin


ganak_pair_module = importlib.import_module(
    "wfomc.cell_graph.compute_pair_factors_ganak"
)


def _forced_atom_cnf():
    context = FOLContext()
    atom = context.predicate("P", 0)()
    return encode_tseitin(atom)


def test_ganak_pair_factor_recovers_rational_marker_coefficient(monkeypatch):
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    cnf = _forced_atom_cnf()
    weights = (
        (arithmetic.one(), arithmetic.one()),
        (arithmetic.from_int(2), arithmetic.from_int(3)),
    )

    def fake_ganak_count(_n_vars, _clauses, ganak_weights, **kwargs):
        assert kwargs["symbolic"] is True
        assert kwargs["npolyvars"] == 1
        marker = kwargs["poly_ctx"].gen(0)
        assert ganak_weights[1][0] == 2 * marker
        assert ganak_weights[1][1] == 3
        return 2 * marker

    monkeypatch.setattr(ganak_pair_module, "ganak_count", fake_ganak_count)

    assert ganak_pair_module.compute_factor_with_ganak(
        cnf,
        weights,
        {1: 0},
        arithmetic,
        timeout=1,
    ) == {1: arithmetic.from_int(2)}


def test_ganak_pair_factor_preserves_one_symbol_polynomial(monkeypatch):
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_POLY,
        symbolic_variables=("w",),
    )
    symbol = arithmetic.symbol("w")
    cnf = _forced_atom_cnf()
    weights = (
        (arithmetic.one(), arithmetic.one()),
        (symbol + arithmetic.one(), arithmetic.from_int(3)),
    )

    def fake_ganak_count(_n_vars, _clauses, ganak_weights, **kwargs):
        context = kwargs["poly_ctx"]
        weight_symbol = context.gen(0)
        marker = context.gen(1)
        expected = (weight_symbol + 1) * marker
        assert ganak_weights[1][0] == expected
        return expected

    monkeypatch.setattr(ganak_pair_module, "ganak_count", fake_ganak_count)

    assert ganak_pair_module.compute_factor_with_ganak(
        cnf,
        weights,
        {1: 0},
        arithmetic,
        timeout=1,
    ) == {1: symbol + arithmetic.one()}


def test_ganak_pair_factor_preserves_multi_symbol_coefficients(monkeypatch):
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_MPOLY,
        symbolic_variables=("w", "z"),
    )
    w = arithmetic.symbol("w")
    z = arithmetic.symbol("z")
    cnf = _forced_atom_cnf()
    weights = (
        (arithmetic.one(), arithmetic.one()),
        (w * z + arithmetic.from_int(2), w),
    )

    def fake_ganak_count(_n_vars, _clauses, _weights, **kwargs):
        context = kwargs["poly_ctx"]
        w_out, z_out, marker = (context.gen(index) for index in range(3))
        return (w_out * z_out + 2) * marker + w_out

    monkeypatch.setattr(ganak_pair_module, "ganak_count", fake_ganak_count)

    assert ganak_pair_module.compute_factor_with_ganak(
        cnf,
        weights,
        {1: 0},
        arithmetic,
        timeout=1,
    ) == {
        0: w,
        1: w * z + arithmetic.from_int(2),
    }


def test_ganak_pair_factor_uses_rational_mode_without_factor_bits(monkeypatch):
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    cnf = _forced_atom_cnf()
    weights = (
        (arithmetic.one(), arithmetic.one()),
        (arithmetic.from_int(2), arithmetic.from_int(3)),
    )

    def fake_ganak_count(_n_vars, _clauses, _weights, **kwargs):
        assert kwargs["symbolic"] is False
        return fmpq(2)

    monkeypatch.setattr(ganak_pair_module, "ganak_count", fake_ganak_count)

    assert ganak_pair_module.compute_factor_with_ganak(
        cnf,
        weights,
        {},
        arithmetic,
        timeout=1,
    ) == {0: arithmetic.from_int(2)}


def test_ganak_pair_factor_declines_rounded_arithmetic(monkeypatch):
    arithmetic = ArithmeticContext(ArithmeticBackend.FLOAT)
    cnf = _forced_atom_cnf()
    weights = ((1.0, 1.0), (2.0, 3.0))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("rounded arithmetic must not invoke Ganak")

    monkeypatch.setattr(ganak_pair_module, "ganak_count", fail_if_called)

    assert (
        ganak_pair_module.compute_factor_with_ganak(
            cnf,
            weights,
            {1: 0},
            arithmetic,
            timeout=1,
        )
        is None
    )
