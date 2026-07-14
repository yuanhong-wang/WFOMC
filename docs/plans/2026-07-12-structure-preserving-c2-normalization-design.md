# Structure-preserving C2 normalization design

## Scope

Improve C2 normalization without adding sort support and without moving general comparator lowering into the common normalizer.

## Pipeline

1. Validate that the source is a closed sentence in the unary/binary two-variable solver fragment.
2. Alpha-rename shadowed binders.
3. Apply only domain-independent trivial-count simplifications.
4. Extract direct top-level sections.
5. Recursively preserve the remaining Boolean syntax while replacing embedded quantifiers with definitions.
6. Validate the resulting `C2NormalForm` invariants.

`CountDefinition` remains the single owner of an embedded count section. Negation belongs to the QF Boolean formula, so modulo does not require a complementary comparator.

## Validation

The validator additionally enforces typed scope variables, closed existential sections, distinct row variables, and unique count-definition marker predicates.

## Tests

- Shape regressions for `Iff`, negated modulo, and open-source rejection.
- Invalid manually constructed IR cases.
- Exhaustive truth comparison over domains of size 0, 1, and 2 for global and row embedded counts.
- Full solver smoke matrix to ensure direct sections retain existing execution behavior.
