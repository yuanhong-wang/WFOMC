"""Runtime options and per-run caches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeOptions:
    """External runtime dependencies supplied for one engine context."""

    # Explicit Ganak executable; None uses GANAK or PATH discovery.
    propositional_ganak_path: str | None = None
    # Maximum concrete per-domain executions retained by one runtime.
    execution_cache_size: int = 8

    def __post_init__(self) -> None:
        if self.execution_cache_size < 0:
            raise ValueError("execution_cache_size must be non-negative")


@dataclass(frozen=True)
class RuntimeCacheStats:
    """Immutable per-bucket cache counters and current sizes."""

    # Number of successful cache lookups by bucket name.
    hits: Mapping[str, int]
    # Number of values built after a missing cache lookup by bucket name.
    misses: Mapping[str, int]
    # Current number of entries stored in each cache bucket.
    sizes: Mapping[str, int]


@dataclass
class RuntimeCache:
    """Per-runtime in-memory cache owned by the engine.

    The cache is intentionally instance-scoped.  Reuse happens only when callers
    reuse the same :class:`RuntimeContext`.
    """

    # Cache buckets intentionally use dict[object, object] because keys and
    # values span heterogeneous types (dataclasses, enums, FLINT values,
    # formulas, domain constants) at runtime boundaries.
    # Source feature-analysis results keyed independently of domain size.
    features: dict[object, object] = field(default_factory=dict)
    # Reusable domain-free compilations keyed by problem, algorithm, and options.
    compiled_problems: dict[object, object] = field(default_factory=dict)
    # Reusable algorithm-owned input templates keyed by compiled branch.
    algo_input_templates: dict[object, object] = field(default_factory=dict)
    # Concrete per-domain executions. This bucket is bounded by RuntimeOptions.
    executions: dict[object, object] = field(default_factory=dict)
    # Fully decoded WFOMC results keyed by problem, algorithm, and options.
    results: dict[object, object] = field(default_factory=dict)
    _hits: dict[str, int] = field(default_factory=dict)
    _misses: dict[str, int] = field(default_factory=dict)

    def get_or_build(
        self,
        bucket: str,
        key: object,
        build: Callable[[], object],
    ) -> object:
        store = self._bucket(bucket)
        if key in store:
            self._hits[bucket] = self._hits.get(bucket, 0) + 1
            return store[key]
        self._misses[bucket] = self._misses.get(bucket, 0) + 1
        value = build()
        store[key] = value
        return value

    def trim(self, bucket: str, maximum_size: int) -> None:
        """Discard oldest inserted entries until a bucket fits its bound."""

        store = self._bucket(bucket)
        while len(store) > maximum_size:
            oldest = next(iter(store))
            del store[oldest]

    def stats(self) -> RuntimeCacheStats:
        return RuntimeCacheStats(
            hits={bucket: self._hits.get(bucket, 0) for bucket in _CACHE_BUCKETS},
            misses={bucket: self._misses.get(bucket, 0) for bucket in _CACHE_BUCKETS},
            sizes={bucket: len(self._bucket(bucket)) for bucket in _CACHE_BUCKETS},
        )

    def _bucket(self, bucket: str) -> dict[object, object]:
        stores = {
            "features": self.features,
            "compiled_problems": self.compiled_problems,
            "algo_input_templates": self.algo_input_templates,
            "executions": self.executions,
            "results": self.results,
        }
        try:
            return stores[bucket]
        except KeyError:
            raise KeyError(f"unknown runtime cache bucket: {bucket}") from None


@dataclass
class RuntimeContext:
    """Runtime state passed through engine, input builders, and algorithms."""

    # External dependencies and executable overrides for this runtime.
    options: RuntimeOptions = field(default_factory=RuntimeOptions)
    # Instance-scoped cache reused by calls sharing this context.
    cache: RuntimeCache = field(default_factory=RuntimeCache)

    @classmethod
    def from_runtime(
        cls,
        runtime: "RuntimeContext | RuntimeOptions | None",
    ) -> "RuntimeContext":
        if isinstance(runtime, cls):
            return runtime
        if runtime is None:
            return cls()
        if isinstance(runtime, RuntimeOptions):
            return cls(options=runtime)
        raise TypeError(
            "runtime must be RuntimeContext, RuntimeOptions, or None; "
            f"got {type(runtime).__name__}"
        )


_CACHE_BUCKETS = (
    "features",
    "compiled_problems",
    "algo_input_templates",
    "executions",
    "results",
)


__all__ = [
    "RuntimeCache",
    "RuntimeCacheStats",
    "RuntimeContext",
    "RuntimeOptions",
]
