# ADR-0015: Algorithm Package Convention

## Status

Accepted

## Context

Algorithm packages exposed different builder and solver names, and every
`__init__.py` implemented a lazy forwarding table. Tail-signature built its
input in `solve.py`; bounded-treewidth kept its input models in `solve.py`.
Following registration therefore required learning a different layout for each
algorithm, while the engine registry only needed each package's `spec.SPEC`.

## Decision

Every registered algorithm package owns `spec.py` with exactly one `SPEC`.
Packages that own an implementation use these roles:

- `input.py`: input dataclasses and `build_input`;
- `solve.py`: the public `solve(algo_input, runtime)` entry point;
- `spec.py`: reductions, `prepare`, capability metadata, and `SPEC`;
- `__init__.py`: package documentation only, with no lazy forwarding exports.

Additional files are named for implementation responsibilities such as
`kernel.py`, `graph.py`, `operations.py`, `counting.py`, or `runtime.py`.
FastV2 remains an explicit configuration-only variant and reuses Fast's input
builder and solver rather than adding empty wrapper modules. The unavailable
bounded-treewidth extension owns its input contracts and solver entry point but
does not pretend to have a working builder.

The root `wfomc.algo` package exports framework contracts only. The generic
Ganak subprocess boundary lives at `wfomc.ganak` because both propositional
counting and cell-graph pair-factor construction reuse it; algorithm-specific
weight preparation remains with its caller.

## Consequences

- Specs can be read in the same order for every algorithm.
- Input construction cannot drift into solver modules.
- Package imports no longer trigger implementation-specific lazy machinery.
- Internal algorithm types are imported from their owning module instead of
  being re-exported through package initializers.
- Configuration variants may reuse an implementation explicitly without
  manufacturing files solely for visual symmetry.
