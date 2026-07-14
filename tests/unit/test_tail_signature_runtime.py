from __future__ import annotations

import sys
from types import ModuleType

import pytest

from wfomc.errors import ExternalToolError
from wfomc.algo.tail_signature.runtime import (
    load_tail_signature_module,
    load_tail_signature_runtime,
)


def test_load_tail_signature_runtime_from_module_path(tmp_path):
    module_path = tmp_path / "fake_tail_engine.py"
    module_path.write_text(
        "\n".join(
            [
                "_ALLOW_INEFFICIENT_FALLBACK_LIBRARIES = False",
                "class TailSignatureWFOMC:",
                "    pass",
            ]
        )
    )

    runtime = load_tail_signature_runtime(
        module_path=module_path,
        allow_inefficient_fallback=True,
    )

    assert runtime.tail_signature_engine_factory.__name__ == "TailSignatureWFOMC"
    module = sys.modules[runtime.tail_signature_engine_factory.__module__]
    assert module._ALLOW_INEFFICIENT_FALLBACK_LIBRARIES is True


def test_load_tail_signature_runtime_from_module_name():
    module = ModuleType("fake_tail_signature_module")

    class TailSignatureWFOMC:
        pass

    module.TailSignatureWFOMC = TailSignatureWFOMC
    sys.modules[module.__name__] = module

    runtime = load_tail_signature_runtime(module_name=module.__name__)

    assert runtime.tail_signature_engine_factory is TailSignatureWFOMC


def test_load_tail_signature_module_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        load_tail_signature_module()

    with pytest.raises(ValueError, match="exactly one"):
        load_tail_signature_module(
            module_name="x",
            module_path=tmp_path / "x.py",
        )


def test_load_tail_signature_runtime_rejects_module_without_runtime_engine(tmp_path):
    module_path = tmp_path / "empty_tail_engine.py"
    module_path.write_text("VALUE = 1\n")

    with pytest.raises(ExternalToolError, match="TailSignatureWFOMC"):
        load_tail_signature_runtime(module_path=module_path)


def test_load_tail_signature_module_reports_missing_path(tmp_path):
    missing = tmp_path / "missing_tail_engine.py"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_tail_signature_module(module_path=missing)
