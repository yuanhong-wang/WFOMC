"""Optional tail-signature engine loading helpers."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from wfomc.errors import ExternalToolError
from wfomc.engine.runtime import RuntimeOptions


def load_tail_signature_runtime(
    *,
    module_name: str | None = None,
    module_path: str | Path | None = None,
    allow_inefficient_fallback: bool = False,
) -> RuntimeOptions:
    """Load an opt27-compatible engine and return runtime options for it.

    The external engine is intentionally loaded on demand instead of vendored
    into the framework package. This keeps parser/engine code independent of
    any one tail-signature implementation.
    """

    module = load_tail_signature_module(
        module_name=module_name,
        module_path=module_path,
    )
    if allow_inefficient_fallback and hasattr(
        module,
        "_ALLOW_INEFFICIENT_FALLBACK_LIBRARIES",
    ):
        setattr(module, "_ALLOW_INEFFICIENT_FALLBACK_LIBRARIES", True)
    engine_factory = getattr(module, "TailSignatureWFOMC", None)
    if engine_factory is None:
        raise ExternalToolError(
            "tail-signature engine module does not expose TailSignatureWFOMC"
        )
    return RuntimeOptions(tail_signature_engine_factory=engine_factory)


def load_tail_signature_module(
    *,
    module_name: str | None = None,
    module_path: str | Path | None = None,
) -> ModuleType:
    if bool(module_name) == bool(module_path):
        raise ValueError(
            "provide exactly one of module_name or module_path for tail-signature engine loading"
        )
    if module_name is not None:
        return importlib.import_module(module_name)

    path = Path(module_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"tail-signature engine path does not exist: {path}")
    generated_name = f"wfomc_external_tail_signature_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(generated_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tail-signature engine from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[generated_name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["load_tail_signature_module", "load_tail_signature_runtime"]
