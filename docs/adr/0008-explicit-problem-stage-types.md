# ADR-0008: Explicit Problem Stage Types

## Status

Accepted

## Context

The public `Problem` previously represented parser output, C2-normalized
reduction state, and quantifier-free compiled input. Cross-stage `replace()`
calls changed both the runtime type of `sentence` and the representation of
`weights`, so annotations did not describe actual values. Incremental3 also
needed C2 counting sections after the common QF formula had been extracted.

## Decision

Keep three flat problem stages:

- `Problem`: public source formula, raw weights, evidence, and constraints;
- `ReducedProblem`: internal `C2NormalForm`, raw/auxiliary weights, reduction
  artifacts, and remaining evidence;
- `CompiledProblem`: quantifier-free formula and compiled weights consumed by
  algorithm-owned input builders.

`apply_reductions()` converts `Problem` to `ReducedProblem` once. Logical
reductions only update `ReducedProblem`; `compile_reduced_problem()` constructs
a new `CompiledProblem` and re-analyzes execution features. Algorithm builders
then create the existing `AlgoInput` types. Incremental3 compiles counting state
from `ReducedProblem.normal_form` before combining it with `CompiledProblem`.
Decoders remain attached to reduction branches.

## Consequences

- A public `Problem` cannot contain `C2NormalForm` or profile-capacity artifacts.
- A `CompiledProblem` cannot contain quantified formulas.
- Raw and compiled weights no longer share one problem instance.
- The former `NormalFormReductionView` and `reduction/normal_form.py` adapter are
  unnecessary; C2 extraction belongs to `fol/normal_form/c2`.
- Shared fields are repeated across three small dataclasses instead of hidden
  behind inheritance or a phase-generic wrapper.

## Alternatives Considered

- Type aliases only: rejected because runtime-invalid stage combinations remain
  representable.
- One type per reduction pass: rejected as excessive layering.
- A generic `Problem[Phase]`: rejected because it preserves one broad object
  with phase-dependent fields and makes runtime contracts harder to inspect.
