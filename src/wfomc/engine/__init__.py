"""Engine/runtime orchestration package."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CompileArtifacts": ("wfomc.engine.orchestration", "CompileArtifacts"),
    "FeatureSet": ("wfomc.engine.features", "FeatureSet"),
    "RuntimeCache": ("wfomc.engine.runtime", "RuntimeCache"),
    "RuntimeCacheStats": ("wfomc.engine.runtime", "RuntimeCacheStats"),
    "RuntimeContext": ("wfomc.engine.runtime", "RuntimeContext"),
    "RuntimeOptions": ("wfomc.engine.runtime", "RuntimeOptions"),
    "AlgoName": ("wfomc.algo.core", "AlgoName"),
    "analyze_problem": ("wfomc.engine.orchestration", "analyze_problem"),
    "analyze_features": ("wfomc.engine.features", "analyze_features"),
    "compile_problem": ("wfomc.engine.orchestration", "compile_problem"),
    "solve": ("wfomc.engine.orchestration", "solve"),
    "solve_uncached": ("wfomc.engine.orchestration", "solve_uncached"),
}


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'wfomc.engine' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
