# ADR-0011: Prepared Branches Own Arithmetic Contexts

## Status

Accepted

## Context

`WeightOptions` selected a backend while only weight compilation used the
resulting `ArithmeticContext`. Cell graphs, solver kernels, evidence
coefficients, and decoders still created exact FLINT values directly. Rounded
weights could therefore compile successfully and then fail when mixed with an
exact accumulator.

A source problem is not the correct owner because algorithm-specific reduction
may introduce cardinality markers or eliminate numeric requirements. Different
prepared branches may consequently need different symbol sets and backends.

## Decision

Create exactly one immutable `ArithmeticContext` after logical reduction for
each final `PreparedBranch`. The context contains every symbol needed while
that branch is solved. `output_symbols` separately records user-visible weight
symbols that remain after auxiliary marker decoding.

The branch's `CompiledProblem` and `AlgoInput` reference the same context.
Materialization, cell-graph tables, evidence coefficients, solver accumulators,
combinatorial coefficients, and decoders create or coerce numeric values only
through that context. Contexts are not shared globally or across branches.

## Supported Backends

- exact scalar, univariate polynomial, and multivariate polynomial;
- rounded FLOAT and ARB scalars;
- rounded single-symbol ARB polynomial.

Rounded cardinality markers remain unsupported because python-flint provides no
`arb_mpoly`. FLOAT with symbolic weights is unsupported. The propositional
adapter remains exact-only because Ganak's serialization and result protocol are
exact.

## Consequences

- Backend selection now governs the complete in-process calculation.
- Symbol-free components in a symbolic branch use constants from the branch's
  polynomial ring, so their results remain directly composable.
- Cache keys include backend and symbol identity to prevent cross-context reuse.
- Adding a numeric backend requires implementing the context factory operations
  and passing the backend matrix, rather than patching individual algorithms.
