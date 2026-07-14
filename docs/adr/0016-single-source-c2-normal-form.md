# ADR 0016: C2 normal form uses explicit single-source sections

## Status

Accepted — 2026-07-11

## Context

The previous C2 IR stored universal formulas, parsed sections, definition metadata, and source formulas in overlapping forms. Embedded counting definitions appeared both in the direct count lists and in `CountDefinition`, while fresh predicate equivalences appeared both in the universal formula and in `PredicateDefinition`.

## Decision

- Store the universally closed quantifier-free core as `qf_formula` without explicit universal prefixes.
- Keep direct global and row counts only in `counts` and `forall_counts`.
- Keep an embedded count section only inside its `CountDefinition`.
- Keep `CountDefinition`, because a marker replacing an embedded count needs an explicit semantic definition.
- Remove `PredicateDefinition`; its equivalence clause in `qf_formula` is authoritative.
- Make count bodies typed atoms and all scoped variables required.
- Let algorithm-specific reductions decide how to lower comparators and embedded count definitions.

## Consequences

The IR has one owner for each logical fact and callers no longer re-parse a universal formula. Algorithms can distinguish direct constraints from conditional marker definitions without inspecting source formulas. Algorithms that do not support `count_definitions` must reject them explicitly until a suitable lowering pass exists.
