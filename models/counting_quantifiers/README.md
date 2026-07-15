# Counting-quantifier sentences

This directory contains small, exact regression models for the counting
quantifiers handled natively by `incremental3`. Each comparator gets its own
model so a failure stays cheap to reproduce and easy to identify.

The filename prefixes describe the sentence shape:

- `global-*`: a counting quantifier over a unary predicate.
- `row-*`: a counting quantifier over each row of a binary predicate.
- `nullary-definition-*`: a weighted nullary predicate defined by a count.
- `unary-definition-*`: a weighted unary predicate defined by a row count.

The suffixes `eq`, `ne`, `le`, `lt`, `ge`, and `gt` stand for `=`, `!=`, `<=`,
`<`, `>=`, and `>`. The remaining files cover embedded row counts, modulo
definitions, and ordinary existential quantifiers.

`facility-location.wfomcs` is a small facility-location instance with
`n = 6`, `m = 3` facilities, client redundancy `k = 2`, and facility load
`h = 2`. Its expected WFOMC is `binom(6, 3) * 3! = 120`.

For example:

```console
wfomc -i models/counting_quantifiers/embedded-row-count.wfomcs --algo incremental3
```
