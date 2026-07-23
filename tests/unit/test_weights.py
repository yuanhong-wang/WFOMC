"""Tests for weight configuration, symbol discovery, and compilation."""

from __future__ import annotations

import pytest
from flint import (
    arb,
    arb_poly,
    fmpq,
    fmpq_mpoly,
    fmpq_mpoly_ctx,
    fmpq_poly,
    fmpq_series,
)

from wfomc.arithmetic import (
    ArithmeticBackend,
    ArithmeticContext,
    choose_arithmetic_backend,
)
from wfomc.errors import ArithmeticBackendError
from wfomc.weights import (
    WeightOptions,
    collect_symbolic_weight_variables,
    compile_weight_mapping,
)


# --- WeightOptions defaults ---


def test_weight_options_defaults_to_exact():
    opts = WeightOptions()
    assert opts.precision == "exact"
    assert opts.rounded_backend == "arb"
    assert opts.exact_symbolic_backend == "auto"


def test_weight_options_is_frozen():
    opts = WeightOptions()
    with pytest.raises(Exception):
        opts.precision = "round"  # type: ignore[misc]


def test_weight_options_rejects_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        WeightOptions(precision="bad")  # type: ignore[arg-type]


def test_weight_options_rejects_invalid_rounded_backend():
    with pytest.raises(ValueError, match="rounded_backend"):
        WeightOptions(rounded_backend="bad")  # type: ignore[arg-type]


def test_weight_options_rejects_invalid_exact_symbolic_backend():
    with pytest.raises(ValueError, match="exact_symbolic_backend"):
        WeightOptions(exact_symbolic_backend="bad")  # type: ignore[arg-type]


def test_backend_selection_routes_supported_weight_shapes():
    exact = WeightOptions()
    exact_poly = WeightOptions(exact_symbolic_backend="fmpq_poly")
    exact_mpoly = WeightOptions(exact_symbolic_backend="fmpq_mpoly")
    rounded_arb = WeightOptions(precision="round", rounded_backend="arb")
    rounded_float = WeightOptions(precision="round", rounded_backend="float")
    cases = (
        (exact, (), ArithmeticBackend.FMPQ),
        (exact, ("x",), ArithmeticBackend.FMPQ_POLY),
        (exact, ("x", "y"), ArithmeticBackend.FMPQ_MPOLY),
        (exact_poly, ("x",), ArithmeticBackend.FMPQ_POLY),
        (exact_mpoly, ("x",), ArithmeticBackend.FMPQ_MPOLY),
        (rounded_arb, (), ArithmeticBackend.ARB),
        (rounded_arb, ("x",), ArithmeticBackend.ARB_POLY),
        (rounded_float, (), ArithmeticBackend.FLOAT),
    )

    for options, symbols, expected in cases:
        assert choose_arithmetic_backend(options, symbolic_variables=symbols) is expected


def test_exact_fmpq_poly_rejects_multiple_symbolic_variables():
    with pytest.raises(ArithmeticBackendError, match="exactly one"):
        choose_arithmetic_backend(
            WeightOptions(exact_symbolic_backend="fmpq_poly"),
            symbolic_variables=("x", "y"),
        )


def test_round_arb_multiple_symbolic_variables_unsupported():
    with pytest.raises(ArithmeticBackendError, match="arb"):
        choose_arithmetic_backend(
            WeightOptions(precision="round", rounded_backend="arb"),
            symbolic_variables=("x", "y"),
        )


def test_round_float_with_symbolic_variables_unsupported():
    with pytest.raises(ArithmeticBackendError, match="float"):
        choose_arithmetic_backend(
            WeightOptions(precision="round", rounded_backend="float"),
            symbolic_variables=("x",),
        )


# --- collect_symbolic_weight_variables ---


def test_collect_symbolic_weight_variables_empty_when_none():
    from wfomc.fol import true as _true
    from wfomc.problem import Problem

    problem = Problem(sentence=_true())
    assert collect_symbolic_weight_variables(problem.weights) == ()


def test_cardinality_constraints_are_not_weight_symbols_before_reduction():
    from wfomc.cardinality_constraints import (
        CardinalityConstraints,
        CardinalityTerm,
        Comparator,
        LinearCardinalityConstraint,
    )
    from wfomc.fol import true as _true
    from wfomc.problem import Problem

    problem = Problem(
        sentence=_true(),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    terms=(CardinalityTerm("c", 1),),
                    comparator=Comparator.EQ,
                    rhs=1,
                ),
            )
        ),
    )
    assert collect_symbolic_weight_variables(problem.weights) == ()


def test_collect_symbolic_weight_variables_only_reads_actual_weights():
    from wfomc.cardinality_constraints import (
        CardinalityConstraints,
        CardinalityTerm,
        Comparator,
        LinearCardinalityConstraint,
    )
    from wfomc.fol import true as _true
    from wfomc.problem import Problem

    context = fmpq_mpoly_ctx.get(["w"], "lex")

    problem = Problem(
        sentence=_true(),
        weights={("P", 1): (context.gen(0), 1)},
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    terms=(CardinalityTerm("c", 1),),
                    comparator=Comparator.EQ,
                    rhs=1,
                ),
            )
        ),
    )
    assert collect_symbolic_weight_variables(problem.weights) == ("w",)


# --- compile_weight_mapping ---


def _exact_ctx() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FMPQ)


def _float_ctx() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FLOAT)


def _arb_ctx() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.ARB)


def test_compile_weight_mapping_exact_creates_fmpq_values():
    from fractions import Fraction

    weights = {("P", 1): (Fraction(2, 1), Fraction(-3, 1))}
    compiled = compile_weight_mapping(weights, _exact_ctx())
    assert compiled == {("P", 1): (fmpq(2), fmpq(-3))}


def test_compile_weight_mapping_py_float_converts_to_floats():
    from fractions import Fraction

    weights = {("P", 1): (Fraction(2, 1), Fraction(-3, 1))}
    compiled = compile_weight_mapping(weights, _float_ctx())
    positive, negative = compiled[("P", 1)]
    assert positive == 2.0
    assert negative == -3.0
    assert isinstance(positive, float)
    assert isinstance(negative, float)


def test_compile_weight_mapping_arb_converts_to_arb_values():
    weights = {("P", 1): (2, -3)}
    compiled = compile_weight_mapping(weights, _arb_ctx())
    positive, negative = compiled[("P", 1)]
    assert isinstance(positive, arb)
    assert isinstance(negative, arb)
    assert positive == arb(2)
    assert negative == arb(-3)


def test_compile_weight_mapping_fmpq_poly_matches_plan_backend():
    context = fmpq_mpoly_ctx.get(("x0",), "lex")
    x0 = context.gen(0)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ_POLY, ("x0",))
    compiled = compile_weight_mapping({("P", 1): (x0, 1)}, arithmetic)
    positive, negative = compiled[("P", 1)]

    assert isinstance(positive, fmpq_poly)
    assert isinstance(negative, fmpq_poly)
    assert positive == fmpq_poly([0, 1])
    assert negative == fmpq_poly([1])


def test_compile_weight_mapping_fmpq_series_rebinds_cached_polynomials():
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_SERIES,
        ("x0",),
        degree_limits=(("x0", 2),),
    )
    compiled = compile_weight_mapping(
        {("P", 1): (fmpq_poly([1, 2, 3, 4]), 1)},
        arithmetic,
    )
    positive, negative = compiled[("P", 1)]

    assert isinstance(positive, fmpq_series)
    assert isinstance(negative, fmpq_series)
    assert positive.prec == 3
    assert positive.coeffs() == [1, 2, 3]
    assert negative.coeffs() == [1]


def test_compile_weight_mapping_arb_poly_matches_plan_backend():
    context = fmpq_mpoly_ctx.get(("x0",), "lex")
    x0 = context.gen(0)
    arithmetic = ArithmeticContext(ArithmeticBackend.ARB_POLY, ("x0",))
    compiled = compile_weight_mapping({("P", 1): (x0, 1)}, arithmetic)
    positive, negative = compiled[("P", 1)]

    assert isinstance(positive, arb_poly)
    assert isinstance(negative, arb_poly)
    assert str(positive) == str(arb_poly([arb(0), arb(1)]))
    assert str(negative) == str(arb_poly([arb(1)]))


def test_compile_weight_mapping_fmpq_mpoly_matches_plan_backend():
    context = fmpq_mpoly_ctx.get(("x0", "x1"), "lex")
    x0, x1 = context.gens()
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_MPOLY,
        ("x0", "x1"),
    )
    compiled = compile_weight_mapping({("P", 1): (x0 + x1, 1)}, arithmetic)
    positive, negative = compiled[("P", 1)]

    assert isinstance(positive, fmpq_mpoly)
    assert isinstance(negative, fmpq_mpoly)
    assert positive.context().names() == ("x0", "x1")
    assert negative.context().names() == ("x0", "x1")


def test_compile_weight_mapping_converts_univariate_poly_to_mpoly():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ_MPOLY, ("x",))
    source = fmpq_poly([1, 0, 3])

    compiled = compile_weight_mapping({("P", 1): (source, 1)}, arithmetic)
    positive, negative = compiled[("P", 1)]

    assert isinstance(positive, fmpq_mpoly)
    assert positive.context().names() == ("x",)
    assert positive.to_dict() == {(0,): fmpq(1), (2,): fmpq(3)}
    assert negative == positive.context().constant(1)


def test_compile_weight_mapping_rejects_ambiguous_poly_to_mpoly_conversion():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ_MPOLY, ("x", "y"))

    with pytest.raises(ValueError, match="which symbolic variable"):
        compile_weight_mapping(
            {("P", 1): (fmpq_poly([1, 1]), 1)},
            arithmetic,
        )


def test_compile_weight_mapping_empty_weights_returns_empty():
    assert compile_weight_mapping({}, _exact_ctx()) == {}
