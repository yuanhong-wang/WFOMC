"""Runtime options and per-run caches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


EngineFactory = Callable[..., object]
ValueConverter = Callable[[object], object]


@dataclass(frozen=True)
class RuntimeOptions:
    tail_signature_engine_factory: EngineFactory | None = None
    tail_signature_value_converter: ValueConverter | None = None
    propositional_ganak_path: str | None = None


@dataclass(frozen=True)
class RuntimeCacheStats:
    hits: Mapping[str, int]
    misses: Mapping[str, int]
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
    features: dict[object, object] = field(default_factory=dict)
    algo_inputs: dict[object, object] = field(default_factory=dict)
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

    def has(self, bucket: str, key: object) -> bool:
        return key in self._bucket(bucket)

    def get(self, bucket: str, key: object) -> object:
        store = self._bucket(bucket)
        self._hits[bucket] = self._hits.get(bucket, 0) + 1
        return store[key]

    def store(self, bucket: str, key: object, value: object) -> object:
        self._bucket(bucket)[key] = value
        return value

    def clear(self) -> None:
        for bucket in _CACHE_BUCKETS:
            self._bucket(bucket).clear()
        self._hits.clear()
        self._misses.clear()

    def stats(self) -> RuntimeCacheStats:
        return RuntimeCacheStats(
            hits={bucket: self._hits.get(bucket, 0) for bucket in _CACHE_BUCKETS},
            misses={bucket: self._misses.get(bucket, 0) for bucket in _CACHE_BUCKETS},
            sizes={bucket: len(self._bucket(bucket)) for bucket in _CACHE_BUCKETS},
        )

    def _bucket(self, bucket: str) -> dict[object, object]:
        stores = {
            "features": self.features,
            "algo_inputs": self.algo_inputs,
            "results": self.results,
        }
        try:
            return stores[bucket]
        except KeyError:
            raise KeyError(f"unknown runtime cache bucket: {bucket}") from None


@dataclass
class RuntimeContext:
    """Runtime state passed through engine, input builders, and algorithms."""

    options: RuntimeOptions = field(default_factory=RuntimeOptions)
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
    "algo_inputs",
    "results",
)


__all__ = [
    "RuntimeCache",
    "RuntimeCacheStats",
    "RuntimeContext",
    "RuntimeOptions",
    "EngineFactory",
    "ValueConverter",
]
