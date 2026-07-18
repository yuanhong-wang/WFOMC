"""End-to-end unary-evidence strategy and algorithm matrix."""

from pathlib import Path

import pytest

from wfomc import (
    AlgoName,
    AlgoOptions,
    EvidenceStrategy,
    compile_problem,
    parse_input,
    parse_problem,
    solve,
)


ROOT = Path(__file__).parents[2]
MODEL_DIR = ROOT / "models" / "unary_evidence"
MODEL_FILES = tuple(
    sorted(MODEL_DIR.glob("*.wfomcs")) + sorted(MODEL_DIR.glob("*.mln"))
)
ALGORITHMS = (
    AlgoName.STANDARD,
    AlgoName.FAST,
    AlgoName.FASTV2,
    AlgoName.INCREMENTAL,
    AlgoName.INCREMENTAL3,
    AlgoName.RECURSIVE,
)
AUTO_STRATEGIES = {
    AlgoName.STANDARD: EvidenceStrategy.LIFTED_PROFILES,
    AlgoName.FAST: EvidenceStrategy.CCS,
    AlgoName.FASTV2: EvidenceStrategy.LIFTED_PROFILES,
    AlgoName.INCREMENTAL: EvidenceStrategy.LIFTED_PROFILES,
    AlgoName.INCREMENTAL3: EvidenceStrategy.LIFTED_PROFILES,
    AlgoName.RECURSIVE: EvidenceStrategy.CCS,
}


@pytest.mark.parametrize(
    "strategy",
    (None, EvidenceStrategy.CCS),
    ids=("auto", "ccs"),
)
@pytest.mark.parametrize("model_file", MODEL_FILES, ids=lambda path: path.stem)
def test_unary_evidence_algorithm_matrix(model_file: Path, strategy):
    problem = parse_input(str(model_file))
    options = None if strategy is None else AlgoOptions(evidence_strategy=strategy)

    results = {algo: solve(problem, algo=algo, options=options) for algo in ALGORITHMS}

    first = next(iter(results.values()))
    assert all(result == first for result in results.values()), results


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("evidence-only.wfomcs", 4),
        ("impossible-evidence.wfomcs", 0),
        ("overlapping-profiles.wfomcs", 36),
    ),
)
def test_named_unary_evidence_regressions(name: str, expected: int):
    problem = parse_input(str(MODEL_DIR / name))

    assert solve(problem, algo=AlgoName.STANDARD) == expected
    assert solve(
        problem,
        algo=AlgoName.STANDARD,
        options=AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
    ) == expected


@pytest.mark.parametrize(("algo", "expected"), tuple(AUTO_STRATEGIES.items()))
def test_auto_resolves_to_algorithm_evidence_default(algo, expected):
    problem = parse_input(str(MODEL_DIR / "evidence-only.wfomcs"))

    artifacts = compile_problem(problem, algo=algo)

    assert artifacts.algo_options.evidence_strategy is expected


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_explicit_ccs_is_preserved_for_every_algorithm(algo):
    problem = parse_input(str(MODEL_DIR / "evidence-only.wfomcs"))

    artifacts = compile_problem(
        problem,
        algo=algo,
        options=AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
    )

    assert artifacts.algo_options.evidence_strategy is EvidenceStrategy.CCS


def test_named_constants_in_sentence_are_rejected_with_unary_evidence():
    problem = parse_problem(
        r"""
\forall X: (R(X, domain0))
domain = {domain0, domain1}
P(domain0)
"""
    )

    # Lifted evidence assumes exchangeable unnamed domain elements, so named
    # constants remain unsupported for this execution path.
    with pytest.raises(ValueError):
        solve(problem, algo=AlgoName.INCREMENTAL)


def test_fastv2_handles_many_singleton_evidence_profiles():
    const_names = ("e_1", "e_10", "e_2", "e_3", "e_4", "e_5", "e_6", "e_7", "e_8", "e_9")
    entity_preds = tuple(f"E{name.removeprefix('e_')}" for name in const_names)
    evidence = [f"S({name})" for name in const_names]
    for const_name, entity_pred in zip(const_names, entity_preds):
        evidence.extend(
            f"{'' if const_name == other_name else '~'}{entity_pred}({other_name})"
            for other_name in const_names
        )
    problem = parse_problem(
        "\\forall X: (S(X) | ~S(X))\n\n"
        f"domain = {{{', '.join(const_names)}}}\n\n"
        f"{', '.join(evidence)}"
    )

    assert solve(problem, algo=AlgoName.FASTV2) == 1
    assert solve(
        problem,
        algo=AlgoName.FASTV2,
        options=AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
    ) == 1
