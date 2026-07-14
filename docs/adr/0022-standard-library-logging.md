# ADR-0022: Standard-Library Logging

## Status

Accepted

## Context

WFOMC used a global Loguru logger in four modules, disabled the package logger
at import time, and exposed no CLI option to enable it. Some INFO messages
contained complete grounded formulas or propositional CNF input, while the
recursive solver retained a separate mutable `PRINT_TREE` debug path.

## Decision

- Use Python's standard `logging` package with one module logger per file.
- The library installs only a `NullHandler` and never configures output.
- The CLI owns output configuration: no flag means WARNING, `-v` means INFO,
  and `-vv` or more means DEBUG.
- INFO records phase summaries, problem sizes, backend selection, and elapsed
  time. DEBUG records bounded branch/predicate details. WARNING reports
  correctness-preserving degraded execution.
- Never log complete formulas, CNFs, cells, pair factors, SAT models, weight
  tables, or per-state hot-loop progress.
- Delete the Loguru dependency and the recursive debug-print tree rather than
  maintaining a logging adapter.

## Consequences

- Python API users can integrate `wfomc.*` loggers with their existing logging
  configuration without global logger side effects.
- CLI diagnostics are opt-in and useful for performance triage.
- Structured JSON logging is intentionally deferred; this local library/CLI
  has no log aggregation requirement that justifies structlog.
