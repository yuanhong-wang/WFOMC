from __future__ import annotations

import logging
import subprocess
import tomllib
from pathlib import Path

import pytest

import wfomc.cli as cli
from wfomc.cli import build_parser
from wfomc.errors import UnsupportedFeatureError


def test_wfomc_help_smoke():
    completed = subprocess.run(
        ["uv", "run", "wfomc", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Exact WFOMC solver" in completed.stdout
    assert "bounded-treewidth" not in completed.stdout


def test_cli_help_groups_options_by_algorithm_scope():
    help_text = build_parser().format_help()

    common = help_text.split("Common options:", 1)[1].split(
        "Incremental3-only options:", 1
    )[0]
    incremental3 = help_text.split("Incremental3-only options:", 1)[1].split(
        "Propositional-only options:", 1
    )[0]
    propositional = help_text.split("Propositional-only options:", 1)[1].split(
        "Boundary-Profile-only options:", 1
    )[0]
    boundary_profile = help_text.split("Boundary-Profile-only options:", 1)[1]

    assert "--input" in common
    assert "--algo" in common
    assert "--evidence-strategy" in common
    assert "--exact-symbolic-backend" in common
    assert "--existential-strategy" not in common
    assert "--linear-order-encoding" not in common
    assert "--existential-strategy" in incremental3
    assert "--linear-order-encoding" in propositional
    assert "--ganak-path" in propositional
    assert "--bp-tree-reference-domain-size" in boundary_profile


def test_wfomc_requires_input_for_execution():
    completed = subprocess.run(
        ["uv", "run", "wfomc"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--input is required" in completed.stderr


def test_wfomc_runs_input_with_algo():
    completed = subprocess.run(
        [
            "uv",
            "run",
            "wfomc",
            "--input",
            "models/unary_evidence/evidence-only.wfomcs",
            "--algo",
            "standard",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "WFOMC (standard): 4" in completed.stdout


def test_cli_hides_algorithms_that_are_not_directly_runnable():
    parser = build_parser()
    algo_action = next(action for action in parser._actions if action.dest == "algo")

    assert "boundary-profile" in algo_action.choices
    assert "bounded-treewidth" not in algo_action.choices
    assert "propositional" in algo_action.choices
    assert "propositional-reduced" in algo_action.choices


def test_cli_accepts_incremental3_existential_strategy():
    args = build_parser().parse_args(
        [
            "--input",
            "models/existential.wfomcs",
            "--algo",
            "incremental3",
            "--existential-strategy",
            "skolem",
        ]
    )

    assert args.existential_strategy == "skolem"


def test_cli_exposes_only_incremental3_existential_strategies():
    parser = build_parser()
    strategy_action = next(
        action for action in parser._actions if action.dest == "existential_strategy"
    )

    assert strategy_action.choices == ("counting", "skolem")
    assert "only valid for incremental3" in strategy_action.help


def test_cli_accepts_propositional_options():
    args = build_parser().parse_args(
        [
            "--input",
            "models/linear_order/head-middle-tail.wfomcs",
            "--algo",
            "propositional",
            "--evidence-strategy",
            "ground-units",
            "--linear-order-encoding",
            "axioms",
            "--ganak-path",
            "/opt/ganak",
        ]
    )

    assert args.evidence_strategy == "ground-units"
    assert args.linear_order_encoding == "axioms"
    assert args.ganak_path == "/opt/ganak"


def test_cli_accepts_boundary_profile_reference_domain_size():
    args = build_parser().parse_args(
        [
            "--input",
            "models/2-colored-graph.wfomcs",
            "--algo",
            "boundary-profile",
            "--bp-tree-reference-domain-size",
            "8",
        ]
    )

    assert args.bp_tree_reference_domain_size == 8


def test_cli_run_maps_propositional_options_to_engine_contracts(monkeypatch):
    import wfomc.parser as parser_module
    from wfomc.fol.grounding import LinearOrderEncoding
    from wfomc.options import EvidenceStrategy
    from wfomc.result import WFOMCResult

    captured = {}
    problem = object()
    monkeypatch.setattr(parser_module, "parse_problem_file", lambda _path: problem)

    def capture_solve(actual_problem, **kwargs):
        captured["problem"] = actual_problem
        captured.update(kwargs)
        return WFOMCResult(1)

    monkeypatch.setattr(cli, "solve", capture_solve)

    result = cli.run(
        "unused.wfomcs",
        "propositional",
        evidence_strategy="ground-units",
        linear_order_encoding="axioms",
        ganak_path="/opt/ganak",
    )

    assert result.result == 1
    assert captured["problem"] is problem
    assert captured["options"].evidence_strategy is EvidenceStrategy.GROUND_UNITS
    assert (
        captured["options"].linear_order_encoding is LinearOrderEncoding.AXIOMS
    )
    assert captured["runtime"].propositional_ganak_path == "/opt/ganak"


def test_cli_run_maps_boundary_profile_options_to_engine_contract(monkeypatch):
    import wfomc.parser as parser_module
    from wfomc import BoundaryProfileOptions
    from wfomc.result import WFOMCResult

    captured = {}
    problem = object()
    monkeypatch.setattr(parser_module, "parse_problem_file", lambda _path: problem)

    def capture_solve(actual_problem, **kwargs):
        captured["problem"] = actual_problem
        captured.update(kwargs)
        return WFOMCResult(1)

    monkeypatch.setattr(cli, "solve", capture_solve)

    result = cli.run(
        "unused.wfomcs",
        "boundary-profile",
        bp_tree_reference_domain_size=8,
    )

    assert result.result == 1
    assert captured["problem"] is problem
    assert captured["options"].boundary_profile_options == BoundaryProfileOptions(
        tree_reference_domain_size=8,
    )


@pytest.mark.parametrize("option", ("linear_order_encoding", "ganak_path"))
def test_cli_rejects_propositional_only_options_for_other_algorithms(option: str):
    kwargs = {option: "axioms" if option == "linear_order_encoding" else "/opt/ganak"}

    with pytest.raises(ValueError, match="only valid for propositional"):
        cli.run("unused.wfomcs", "fastv2", **kwargs)


def test_cli_rejects_boundary_profile_option_for_other_algorithms():
    with pytest.raises(ValueError, match="only valid for boundary-profile"):
        cli.run(
            "unused.wfomcs",
            "fastv2",
            bp_tree_reference_domain_size=8,
        )


def test_cli_accepts_exact_symbolic_backend_override():
    args = build_parser().parse_args(
        [
            "--input",
            "models/cardinality_constraints_example.wfomcs",
            "--exact-symbolic-backend",
            "fmpq_poly",
        ]
    )

    assert args.exact_symbolic_backend == "fmpq_poly"


def test_cli_defaults_to_automatic_exact_symbolic_backend():
    assert build_parser().parse_args([]).exact_symbolic_backend == "auto"


def test_cli_supports_repeatable_verbose_flag():
    parser = build_parser()

    assert parser.parse_args([]).verbose == 0
    assert parser.parse_args(["-v"]).verbose == 1
    assert parser.parse_args(["-vv"]).verbose == 2


def test_cli_maps_verbosity_to_standard_logging(monkeypatch):
    configured = {}

    def capture_basic_config(**kwargs):
        configured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", capture_basic_config)

    for verbosity, expected_level in (
        (0, logging.WARNING),
        (1, logging.INFO),
        (2, logging.DEBUG),
        (3, logging.DEBUG),
    ):
        configured.clear()
        cli._configure_logging(verbosity)
        assert configured["level"] == expected_level, verbosity
        assert configured["force"] is True


def test_project_exposes_stable_wfomc_entrypoint_only():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    scripts = pyproject["project"]["scripts"]
    assert scripts["wfomc"] == "wfomc.cli:main"
    assert "new_wfomc" not in scripts
    assert "c2_wfomc" not in scripts


def test_cli_formats_expected_wfomc_errors(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise UnsupportedFeatureError("not supported")

    monkeypatch.setattr(cli, "run", fail)

    with pytest.raises(SystemExit, match="2"):
        cli.main(["--input", "unused.wfomcs"])

    assert "UnsupportedFeatureError: not supported" in capsys.readouterr().err


def test_cli_does_not_hide_internal_runtime_errors(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("broken invariant")

    monkeypatch.setattr(cli, "run", fail)

    with pytest.raises(RuntimeError, match="broken invariant"):
        cli.main(["--input", "unused.wfomcs"])


def test_library_logging_is_quiet_by_default():
    handlers = logging.getLogger("wfomc").handlers

    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.NullHandler)


def test_public_errors_share_one_root():
    from wfomc.ganak import GanakError
    from wfomc.errors import ArithmeticBackendError, ExternalToolError, WFOMCError
    from wfomc.fol.normal_form import NormalizeError, NormalFormValidationError

    assert issubclass(UnsupportedFeatureError, WFOMCError)
    assert issubclass(ArithmeticBackendError, WFOMCError)
    assert issubclass(ExternalToolError, WFOMCError)
    assert issubclass(NormalizeError, WFOMCError)
    assert issubclass(NormalFormValidationError, WFOMCError)
    assert issubclass(GanakError, ExternalToolError)
