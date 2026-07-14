"""Validation for the typed C2 normal-form IR."""

from __future__ import annotations

from wfomc.errors import WFOMCError
from wfomc.fol import (
    Atom,
    Formula,
    Quantifier,
    QuantifierKind,
    Variable,
    free_vars,
    is_quantifier_free,
    predicates,
)

from .norm_form import (
    C2NormalForm,
    CountDefinition,
    CountSection,
    ForallCountSection,
)


class NormalFormValidationError(ValueError, WFOMCError):
    """Raised when a normal-form object violates framework invariants."""


_COUNT_COMPARATORS = frozenset({"=", "!=", "<", "<=", ">", ">=", "mod"})


def validate_normal_form(normal_form: C2NormalForm) -> None:
    if not isinstance(normal_form, C2NormalForm):
        raise NormalFormValidationError(
            f"Expected C2NormalForm, got {type(normal_form).__name__}."
        )
    if not isinstance(normal_form.requires_nonempty_domain, bool):
        raise NormalFormValidationError("requires_nonempty_domain must be a bool.")

    if normal_form.qf_formula is not None:
        _require_qf(normal_form.qf_formula, "qf_formula")
    for index, formula in enumerate(normal_form.forall_exists):
        _validate_forall_exists(formula, index)
    for index, formula in enumerate(normal_form.exists):
        _validate_exists(formula, index)
    for index, section in enumerate(normal_form.forall_counts):
        _validate_forall_count(section, f"forall_counts[{index}]")
    for index, section in enumerate(normal_form.counts):
        _validate_count(section, f"counts[{index}]")
    seen_markers: set[object] = set()
    for index, definition in enumerate(normal_form.count_definitions):
        _validate_count_definition(definition, index)
        marker_predicate = definition.marker.predicate
        if marker_predicate in seen_markers:
            raise NormalFormValidationError(
                f"count_definitions[{index}] has a duplicate marker predicate."
            )
        seen_markers.add(marker_predicate)


def _validate_forall_exists(formula: Formula, index: int) -> None:
    label = f"forall_exists[{index}]"
    if not (
        isinstance(formula, Quantifier)
        and formula.kind is QuantifierKind.FORALL
        and len(formula.variables) == 1
    ):
        raise NormalFormValidationError(
            f"{label} must have shape forall X: exists Y: body."
        )
    inner = formula.body
    if not (
        isinstance(inner, Quantifier)
        and inner.kind is QuantifierKind.EXISTS
        and len(inner.variables) == 1
    ):
        raise NormalFormValidationError(
            f"{label} must have shape forall X: exists Y: body."
        )
    outer_var = formula.variables[0]
    inner_var = inner.variables[0]
    _require_variable(outer_var, f"{label} outer variable")
    _require_variable(inner_var, f"{label} inner variable")
    if outer_var == inner_var:
        raise NormalFormValidationError(
            f"{label} outer and inner variables must be different."
        )
    _require_scoped_qf(inner.body, f"{label} body", (outer_var, inner_var))


def _validate_exists(formula: Formula, index: int) -> None:
    label = f"exists[{index}]"
    if not (
        isinstance(formula, Quantifier)
        and formula.kind is QuantifierKind.EXISTS
        and len(formula.variables) == 1
    ):
        raise NormalFormValidationError(f"{label} must have shape exists X: body.")
    variable = formula.variables[0]
    _require_variable(variable, f"{label} variable")
    _require_scoped_qf(formula.body, f"{label} body", (variable,))


def _validate_forall_count(section: object, label: str) -> None:
    if not isinstance(section, ForallCountSection):
        raise NormalFormValidationError(
            f"{label} must be ForallCountSection, got {type(section).__name__}."
        )
    _require_variable(section.outer_var, f"{label} outer_var")
    _require_variable(section.counted_var, f"{label} counted_var")
    if section.outer_var == section.counted_var:
        raise NormalFormValidationError(
            f"{label} outer_var and counted_var must be different."
        )
    _validate_count_constraint(label, section.comparator, section.count)
    _validate_atomic_count_body(
        label,
        section.body,
        (section.outer_var, section.counted_var),
    )


def _validate_count(section: object, label: str) -> None:
    if not isinstance(section, CountSection):
        raise NormalFormValidationError(
            f"{label} must be CountSection, got {type(section).__name__}."
        )
    _require_variable(section.counted_var, f"{label} counted_var")
    _validate_count_constraint(label, section.comparator, section.count)
    _validate_atomic_count_body(label, section.body, (section.counted_var,))


def _validate_count_definition(definition: CountDefinition, index: int) -> None:
    label = f"count_definitions[{index}]"
    if not isinstance(definition, CountDefinition):
        raise NormalFormValidationError(
            f"{label} must be CountDefinition, got {type(definition).__name__}."
        )
    if not isinstance(definition.marker, Atom):
        raise NormalFormValidationError(f"{label} marker must be a typed Atom.")
    section = definition.section
    if isinstance(section, CountSection):
        _validate_count(section, f"{label}.section")
        expected_terms = ()
    elif isinstance(section, ForallCountSection):
        _validate_forall_count(section, f"{label}.section")
        expected_terms = (section.outer_var,)
    else:
        raise NormalFormValidationError(f"{label} has an invalid section.")
    if definition.marker.terms != expected_terms:
        raise NormalFormValidationError(
            f"{label} marker variables do not match its count section."
        )


def _validate_count_constraint(label: str, comparator: object, count: object) -> None:
    if comparator not in _COUNT_COMPARATORS:
        raise NormalFormValidationError(
            f"{label} has unsupported comparator {comparator!r}."
        )
    if comparator == "mod":
        _validate_mod_count(label, count)
        return
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise NormalFormValidationError(
            f"{label} count must be a non-negative integer."
        )


def _validate_mod_count(label: str, count: object) -> None:
    if (
        not isinstance(count, tuple)
        or len(count) != 2
        or any(not isinstance(part, int) or isinstance(part, bool) for part in count)
    ):
        raise NormalFormValidationError(
            f"{label} mod count must be a tuple of integers (remainder, modulus)."
        )
    remainder, modulus = count
    if modulus <= 0:
        raise NormalFormValidationError(f"{label} mod modulus must be positive.")
    if remainder < 0 or remainder >= modulus:
        raise NormalFormValidationError(
            f"{label} mod remainder must satisfy 0 <= remainder < modulus."
        )


def _validate_atomic_count_body(
    label: str,
    body: object,
    variables: tuple[object, ...],
) -> None:
    if not isinstance(body, Atom):
        raise NormalFormValidationError(f"{label} body must be a typed Atom.")
    if body.terms != variables:
        raise NormalFormValidationError(
            f"{label} body terms must match its scoped variables."
        )
    _validate_predicate_arity(body.predicate, label)


def _require_qf(formula: object, label: str) -> None:
    if not isinstance(formula, Formula) or not is_quantifier_free(formula):
        raise NormalFormValidationError(f"{label} must be quantifier-free.")
    if len(free_vars(formula)) > 2:
        raise NormalFormValidationError(f"{label} uses more than two variables.")
    for predicate in predicates(formula):
        _validate_predicate_arity(predicate, label)


def _require_scoped_qf(
    formula: object,
    label: str,
    variables: tuple[Variable, ...],
) -> None:
    _require_qf(formula, label)
    unbound = free_vars(formula) - frozenset(variables)
    if unbound:
        names = ", ".join(sorted(str(variable) for variable in unbound))
        raise NormalFormValidationError(
            f"{label} has unbound variables outside its section: {names}."
        )


def _require_variable(value: object, label: str) -> None:
    if not isinstance(value, Variable):
        raise NormalFormValidationError(f"{label} must be a typed Variable.")


def _validate_predicate_arity(predicate: object, label: str) -> None:
    if predicate.arity > 2:
        raise NormalFormValidationError(
            f"{label} uses unsupported predicate arity {predicate.arity}."
        )


__all__ = ["NormalFormValidationError", "validate_normal_form"]
