# Fast Symmetric-Clique Convolution Design

## Goal

Replace Fast's cell-by-cell symmetric-clique recurrence with the dense twisted
binomial convolution and repeated-squaring strategy already validated by the
boundary-profile solver, without coupling Fast to boundary-profile modules.

## Architecture

Add `wfomc.algo.symmetric_clique`, an algorithm-neutral internal module.  Its
convolver computes

```text
(A star_r B)[n] = sum_i choose(n, i) A[i] B[n-i] r^(i(n-i))
```

over the active `ArithmeticContext`.  The operation is associative and
commutative, so identical local rows can be raised to a cell multiplicity with
binary exponentiation.  The API accepts already-materialized local rows; Fast
therefore retains its existing convention about whether cell weights live
inside or outside the J-term.

Fast materialized operations lazily build and cache one complete J-message per
clique and domain.  Ordinary Fast cliques are homogeneous and always use the
power path.  FastV2 and evidence-aware cliques may contain heterogeneous local
rows: equal rows are grouped and powered, then the group messages are combined.
The existing scalar recurrence remains temporarily available as an oracle and
fallback while the new path is validated.

## Rebase boundary

This branch starts from `devel` and does not touch
`wfomc.algo.boundary_profile`.  When `bp-dp` is rebased, its private
`_Combinatorics`, `_PowerRow`, `_ExponentCache`, and
`_combine_one_dimensional` can be replaced by the shared convolver without a
path-level rebase conflict.

## Correctness and performance

Unit tests compare every coefficient with a direct recurrence across exact and
symbolic arithmetic, verify associativity, and count combines to prove the
logarithmic homogeneous path.  Fast/FastV2 integration tests cover ordinary and
unary-evidence solving.  Benchmarks must compare only paired successful cases
and include clique size because the optimization targets large repeated
cliques.
