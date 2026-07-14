# ADR 0018: Preserve the predicate universe across cell-graph branches

## Status

Accepted — 2026-07-12; amended — 2026-07-13

## Context

Cell-graph construction derived its predicates independently from each formula after nullary substitution and Boolean simplification. A branch such as `A=true` in `A or P(x)` simplified to `true`, which removed `P` and all of its otherwise free interpretations. A separate shortcut treated every all-unary formula as pair-independent, although formulas such as `P(x) iff P(y)` constrain pairs of unary cell types.

## Decision

- Freeze the source formula's predicate universe once at the `build_cell_graphs` entry point.
- Add explicitly required evidence and order predicates to that universe.
- Pass the frozen universe to every nullary branch builder.
- Keep the complete unary and binary predicate universe as explicit builder
  metadata, even when branch simplification removes a predicate from the
  formula.
- Do not make metadata visible by conjoining `A | ~A`. Projected cell
  enumeration expands predicate bits absent from the diagonal CNF, and pair
  construction accounts analytically for absent off-diagonal atoms. A missing
  ordinary atom contributes `w+ + w-`; a missing incremental3 projection atom
  contributes both weighted mask branches.
- Always enforce the grounded pair formula. Unary-only tables evaluate it under
  the two complete cell assignments; tables with binary predicates enumerate
  compatible pair models. Do not infer pair independence only from predicate
  arity.
- Support the standard solver's zero-size cell configuration. Distinguish a
  missing QF formula (`None`, meaning `true`) from an explicit Boolean `false`.

## Consequences

Nullary branching and local simplification preserve WFOMC vocabulary semantics. Cross-element constraints over unary predicates produce zero-weight incompatible cell pairs. Unary-only formulas retain a direct evaluation path without assuming every cell pair is valid. The standard path handles empty domains for formulas that do not require non-empty Scott abstraction. Correctness is covered against propositional grounding on small domains.

Keeping the universe as metadata avoids Tseitin variables and clauses whose
only purpose was to encode tautologies. Free predicate interpretations remain
part of the exact count without enlarging the Boolean formula compiled by the
pair backend.
