# ADR 0017: Preserve Boolean structure during C2 normalization

## Status

Accepted — 2026-07-12

## Context

The C2 normalizer previously eliminated implications and converted the whole source formula to negation normal form before extracting sections. Expanding an equivalence duplicated quantified subformulas and therefore produced duplicate count definitions. Negated modulo counts were rejected because modulo has no single complementary comparator, even though the IR can represent the original count with a marker under negation.

## Decision

- Require the source passed to `normalize()` to be a closed sentence.
- Keep direct top-level section extraction for `exists`, `forall-exists`, global counts, and row counts.
- Recursively rebuild `Not`, `And`, `Or`, `Implies`, and `Iff` without expanding them.
- Replace only embedded ordinary and counting quantifiers with fresh markers and their definitions.
- Represent a negated modulo count as a negated marker whose `CountDefinition` retains the positive modulo section.
- Keep comparator lowering algorithm-specific.

## Consequences

Normalization is linear in the Boolean syntax rather than duplicating quantified subtrees through equivalence expansion. One source count occurrence creates one count definition. The QF core may contain `Implies` and `Iff`, which are already supported by the FOL semantics and grounding layers. Algorithms that cannot lower an embedded `CountDefinition` continue to reject it explicitly.
