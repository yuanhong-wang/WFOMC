# ADR-0021: Error Ownership and Public Hierarchy

## Status

Accepted

## Context

`PlanningError` represented unrelated failures: unsupported problem features,
wrong algorithm input types, incomplete materialization, inconsistent engine
artifacts, and external tail-signature configuration. Callers could not tell a
supported user error from an internal contract failure. Other domain errors
did not share a common WFOMC base.

## Decision

- `errors.py` owns only cross-subsystem public categories and stable external
  tool errors: `WFOMCError`, `UnsupportedFeatureError`,
  `ArithmeticBackendError`, `ExternalToolError`, and `GanakError`.
- `PlanningError` and `UnsupportedArithmeticBackend` are removed without
  compatibility aliases.
- Domain exceptions remain beside their owners. `NormalizeError` and
  `NormalFormValidationError` inherit `WFOMCError`; shared `GanakError`
  inherits `ExternalToolError`.
- Wrong algorithm input types raise `TypeError`. Broken engine or materialized
  data invariants raise `RuntimeError`. Invalid loader arguments raise
  `ValueError`; missing paths and unloadable modules use the corresponding
  built-in exceptions.
- The CLI formats expected `WFOMCError`, file-system, and value errors but does
  not hide internal `RuntimeError` failures.

## Consequences

- API callers can catch every expected framework error through `WFOMCError` or
  select a stable cross-subsystem category.
- Concrete domain errors keep their local ownership and do not turn
  `errors.py` into a registry of every exception.
- Internal pipeline defects are no longer mislabeled as planning failures.
