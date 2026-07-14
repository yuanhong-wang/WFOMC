"""Stable cross-subsystem exception categories for the public WFOMC API."""

from __future__ import annotations


class WFOMCError(Exception):
    """Base class for expected errors raised through the public WFOMC API."""


class UnsupportedFeatureError(WFOMCError):
    """Raised when an algorithm does not support a requested feature."""


class ArithmeticBackendError(WFOMCError):
    """Raised when an arithmetic backend cannot realize a requested operation.

    WFOMC must never silently fall back to ``fmpq`` when a different backend
    was selected: if a context cannot mint or coerce a value (e.g. an
    unwired backend, or coercing a value of an incompatible type such as a
    float into a non-float backend), it fails early with this error.
    """


class ExternalToolError(WFOMCError):
    """Raised when an optional external solver or engine cannot be used."""


class GanakError(ExternalToolError):
    """Raised when Ganak is missing, fails, or returns unparsable output."""


__all__ = [
    "ArithmeticBackendError",
    "ExternalToolError",
    "GanakError",
    "UnsupportedFeatureError",
    "WFOMCError",
]
