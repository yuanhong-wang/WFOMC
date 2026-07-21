"""Static checks for the engine/algo/reduction dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_SOURCE = Path(__file__).parents[2] / "src" / "wfomc"


@pytest.mark.parametrize(
    ("path", "forbidden"),
    (
        (
            _SOURCE / "algo",
            ("wfomc.engine", "wfomc.reduction"),
        ),
        (
            _SOURCE / "reduction",
            ("wfomc.engine", "wfomc.algo"),
        ),
    ),
)
def test_layers_have_no_reverse_imports(
    path: Path,
    forbidden: tuple[str, ...],
):
    violations = []
    for source_file in sorted(path.rglob("*.py")):
        for imported in _imported_modules(source_file):
            if any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in forbidden
            ):
                violations.append(
                    f"{source_file.relative_to(_SOURCE.parent)} -> {imported}"
                )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("name", ("problem.py", "stages.py", "options.py"))
def test_neutral_contract_modules_do_not_import_framework_layers(name: str):
    source_file = _SOURCE / name
    imported = _imported_modules(source_file)
    forbidden = ("wfomc.algo", "wfomc.engine", "wfomc.reduction")
    violations = [
        module
        for module in imported
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in forbidden
        )
    ]
    assert not violations


def _imported_modules(source_file: Path) -> tuple[str, ...]:
    tree = ast.parse(source_file.read_text(), filename=str(source_file))
    package = ("wfomc", *source_file.relative_to(_SOURCE).parent.parts)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                prefix = package[:keep]
                suffix = tuple(node.module.split(".")) if node.module else ()
                imported.append(".".join((*prefix, *suffix)))
            elif node.module:
                imported.append(node.module)
    return tuple(imported)
