from pathlib import Path

import pytest
from flint import fmpq as Rational

from wfomc import Algo, Pred, UnaryEvidenceStrategy, X, Y, parse_input, wfomc
from wfomc.cell_graph import CellGraph
from wfomc.context import WFOMCContext
from wfomc.parser.wfomcs_parser import parse as parse_wfomcs
from wfomc.solver import resolve_unary_evidence_strategy


ROOT = Path(__file__).parents[2]
UNARY_EVIDENCE_MODELS = sorted(
    (ROOT / "models" / "unary_evidence").glob("*.wfomcs")
) + sorted((ROOT / "models" / "unary_evidence").glob("*.mln"))

ALGORITHMS = (
    Algo.STANDARD,
    Algo.FAST,
    Algo.FASTv2,
    Algo.INCREMENTAL,
    Algo.INCREMENTAL3,
    Algo.RECURSIVE,
)


@pytest.mark.parametrize(
    "strategy",
    (UnaryEvidenceStrategy.AUTO, UnaryEvidenceStrategy.CCS),
)
@pytest.mark.parametrize("model_file", UNARY_EVIDENCE_MODELS, ids=lambda path: path.stem)
def test_unary_evidence_algorithm_matrix(model_file, strategy):
    results = [
        wfomc(parse_input(str(model_file)), algo, strategy)
        for algo in ALGORITHMS
    ]

    assert all(result == results[0] for result in results)


@pytest.mark.parametrize(
    "name, expected",
    (
        ("evidence-only.wfomcs", 4),
        ("impossible-evidence.wfomcs", 0),
        ("overlapping-profiles.wfomcs", 36),
    ),
)
def test_named_unary_evidence_regressions(name, expected):
    problem = parse_input(str(ROOT / "models" / "unary_evidence" / name))

    assert wfomc(problem, Algo.STANDARD) == expected
    assert wfomc(problem, Algo.STANDARD, UnaryEvidenceStrategy.CCS) == expected


@pytest.mark.parametrize(
    "algo, expected",
    (
        (Algo.STANDARD, UnaryEvidenceStrategy.AUTO),
        (Algo.FAST, UnaryEvidenceStrategy.CCS),
        (Algo.FASTv2, UnaryEvidenceStrategy.AUTO),
        (Algo.INCREMENTAL, UnaryEvidenceStrategy.AUTO),
        (Algo.INCREMENTAL3, UnaryEvidenceStrategy.AUTO),
        (Algo.RECURSIVE, UnaryEvidenceStrategy.CCS),
    ),
)
def test_auto_resolves_to_best_supported_strategy(algo, expected):
    assert resolve_unary_evidence_strategy(
        algo, UnaryEvidenceStrategy.AUTO
    ) == expected


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_explicit_ccs_is_preserved_for_every_algorithm(algo):
    assert resolve_unary_evidence_strategy(
        algo, UnaryEvidenceStrategy.CCS
    ) == UnaryEvidenceStrategy.CCS


def test_named_constants_in_sentence_fail_fast_with_unary_evidence():
    problem = parse_wfomcs(r"""
\forall X: (R(X, domain0))

domain = {domain0, domain1}

P(domain0)
""")

    with pytest.raises(ValueError, match="exchangeable domain"):
        wfomc(problem, Algo.INCREMENTAL)


def test_fastv2_auto_handles_many_singleton_evidence_profiles():
    const_names = ("e_1", "e_10", "e_2", "e_3", "e_4",
                   "e_5", "e_6", "e_7", "e_8", "e_9")
    entity_preds = tuple(f"E{name.removeprefix('e_')}" for name in const_names)
    evidence = [f"S({name})" for name in const_names]
    for const_name, entity_pred in zip(const_names, entity_preds):
        evidence.extend(
            f"{'' if const_name == other_name else '~'}{entity_pred}({other_name})"
            for other_name in const_names
        )

    problem = parse_wfomcs(
        "\\forall X: (S(X) | ~S(X))\n\n"
        f"domain = {{{', '.join(const_names)}}}\n\n"
        f"{', '.join(evidence)}"
    )

    context = WFOMCContext(problem)
    cell_graph, _ = next(context.build_cell_graphs())

    assert len(cell_graph.cells) == len(const_names)
    assert wfomc(problem, Algo.FASTv2) == 1
    assert wfomc(problem, Algo.FASTv2, UnaryEvidenceStrategy.CCS) == 1


def test_profile_guided_two_tables_condition_on_cell_pairs():
    const_names = ("e_1", "e_2", "e_3", "e_4")
    entity_preds = tuple(f"E{name.removeprefix('e_')}" for name in const_names)
    evidence = [f"S({name})" for name in const_names]
    for const_name, entity_pred in zip(const_names, entity_preds):
        evidence.extend(
            f"{'' if const_name == other_name else '~'}{entity_pred}({other_name})"
            for other_name in const_names
        )

    problem = parse_wfomcs(
        "\\forall X: (\\forall Y: ((S(X) | ~S(X)) & (R(X,Y) | ~R(X,Y))))\n\n"
        f"domain = {{{', '.join(const_names)}}}\n\n"
        f"{', '.join(evidence)}"
    )

    context = WFOMCContext(problem)
    cell_graph, _ = next(context.build_cell_graphs())

    assert any(pred.arity == 2 for pred in cell_graph.gnd_formula_ab.preds())
    assert len(cell_graph.cells) == 2 * len(const_names)
    assert all(two_table.models for two_table in cell_graph.two_tables.values())


def test_profile_selector_two_tables_project_internal_atoms_once():
    p = Pred("ProfileP", 1)
    q = Pred("ProfileQ", 1)
    r = Pred("ProfileR", 2)
    formula = (
        (p(X) | ~p(X))
        & (q(X) | ~q(X))
        & (r(X, Y) | ~r(X, Y))
    )

    def get_weight(pred):
        if pred == r:
            return Rational(2, 1), Rational(3, 1)
        return Rational(1, 1), Rational(1, 1)

    cell_graph = CellGraph(
        formula,
        get_weight,
        cell_formulas=(p(X), p(X) & q(X)),
    )
    overlapping_cell = next(
        cell
        for cell in cell_graph.cells
        if cell.is_positive(p) and cell.is_positive(q)
    )
    table = cell_graph.two_tables[(overlapping_cell, overlapping_cell)]

    assert cell_graph.get_two_table_weight(
        (overlapping_cell, overlapping_cell)
    ) == Rational(25, 1)
    assert not any(
        "@cell_profile" in str(lit)
        for model in table.models
        for lit in model
    )
    assert not any("@cell_profile" in str(lit) for lit in table.gnd_lits)
