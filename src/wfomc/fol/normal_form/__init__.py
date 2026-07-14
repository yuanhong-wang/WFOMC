"""FOL normal-form namespace.

The current implementation is C2-specific and lives in
:mod:`wfomc.fol.normal_form.c2`.
"""

from __future__ import annotations

from .c2 import (
    C2NormalForm,
    CountDefinition,
    CountSection,
    ForallCountSection,
    NormalizeError,
    NormalFormValidationError,
    normalize,
    validate_normal_form,
)

__all__ = [
    "C2NormalForm",
    "CountDefinition",
    "CountSection",
    "ForallCountSection",
    "NormalizeError",
    "NormalFormValidationError",
    "normalize",
    "validate_normal_form",
]
