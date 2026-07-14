# ADR-0009: Algorithm Readiness Metadata

## Status

Accepted

## Context

Registration only proves that an algorithm has an adapter. It does not prove
that users can run it without additional software or unfinished integration.
The CLI previously exposed tail-signature and bounded-treewidth even though the
former requires an external engine factory and the latter has no implemented
preparation path.

## Decision

Every `AlgoSpec` declares an `AlgoMaturity` and a tuple of
`external_requirements`. Stable and beta specs are directly selectable from the
CLI. Experimental and unavailable specs remain registered for Python-level
integration and testing but are not advertised as runnable CLI choices.

Current exceptional registrations are:

- propositional: beta; Ganak is its external scalable counting backend;
- tail-signature: experimental; requires an injected engine factory;
- bounded-treewidth: unavailable; its decomposition integration is unfinished.

## Consequences

- Adding an enum member no longer automatically advertises an unusable command.
- Readiness and environment requirements are inspectable without running a
  solver and interpreting its exception.
- The registry remains a single, flat contract; no capability wrapper or
  plugin abstraction is introduced.
- Detailed formula-feature support remains in option resolution until a full
  capability matrix is justified.
