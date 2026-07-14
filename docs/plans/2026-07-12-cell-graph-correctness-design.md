# Cell-graph correctness design

## Scope

Fix predicate loss during nullary branching, remove the unsound all-unary pair shortcut, and complete the standard algorithm's empty-domain configuration path without adding new public graph abstractions.

## Construction

`build_cell_graphs` freezes the predicates occurring in the original QF formula together with explicitly required evidence and order predicates. Each private builder receives that set. Diagonal grounding includes every `P(c)` or `R(c,c)` atom in the set; pair grounding includes every atom over `{a,b}`. Missing atoms are added as tautologies so model enumeration preserves their free interpretations.

Every pair table enforces the grounded pair formula. When all predicates are unary, the two cells provide a complete assignment and the formula is evaluated directly. Otherwise, the builder enumerates pair models and conditions them on the selected cells. Predicate arity alone is not sufficient to prove pair independence.

## Empty domain

An unconstrained profile of size zero produces the all-zero cell configuration. A size-zero profile with no compatible cell is already satisfied. `MultinomialCoefficients.setup(0)` initializes the zero-order table. Formula fallback checks use `is None`, so an explicit QF `false` is not replaced by `true`. This makes the standard solver distinguish closed `false` from vacuously true universally quantified contradictions on the empty domain.

## Verification

Compare standard cell-graph counts with propositional grounding for domains of size one and two, covering cross-element unary constraints and nullary branches that remove unary or binary predicates. Retain the existing cell-graph structural and solver-matrix tests.
