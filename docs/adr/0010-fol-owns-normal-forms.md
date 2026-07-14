# ADR-0010: FOL Owns Normal Forms

## Status

Accepted

## Context

The C2 normal-form IR, normalization pass, and validation lived in a top-level
`normal_form` package even though they consume only FOL syntax and expose a
derived representation of FOL formulas. This made one logical domain appear as
two unrelated top-level packages.

At the same time, moving FOL analysis, rewriting, semantics, and grounding into
a generic `fol/utils` package would hide their distinct domain responsibilities
behind an uninformative namespace.

## Decision

Move the normal-form package to `wfomc.fol.normal_form`, retaining the C2
sub-package and its three direct responsibilities: IR, normalization, and
validation.

Keep the remaining FOL modules flat. Do not introduce `fol.utils`; add a
sub-package only when a concrete responsibility grows beyond a readable module.
Normal-form symbols are imported from `wfomc.fol.normal_form`, not re-exported
from the already broad `wfomc.fol` DSL facade.

## Consequences

- FOL syntax and its derived normal forms have one clear owner.
- The dependency direction remains `fol.syntax -> fol.normal_form -> reduction`.
- Call sites must use the new namespace; no legacy import shim is retained.
- The package gains one meaningful nesting level without creating a generic
  utility bucket or additional wrapper types.
