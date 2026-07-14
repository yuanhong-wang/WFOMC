"""Algorithm package public surface."""

# Algorithm implementations live in sibling modules and are dispatched by the
# engine.
from wfomc.fol.grounding import (
    LinearOrderEncoding,
    resolve_linear_order_encoding,
)
from .core import (
    AlgoName,
    AlgoMaturity,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    ExistentialStrategy,
    algo_spec,
)

__all__ = [
    "AlgoName",
    "AlgoMaturity",
    "AlgoOptions",
    "AlgoSpec",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "LinearOrderEncoding",
    "resolve_linear_order_encoding",
    "algo_spec",
]
