"""C2 normal-form conversion.

The typed formula path performs real definitional normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from collections.abc import Iterable

from wfomc.fol import free_vars, predicates
from wfomc.errors import WFOMCError
from wfomc.fol.syntax import formula_children
from wfomc.fol import context_for
from wfomc.fol import (
    And,
    Atom,
    BoolConst,
    CountingQuantifier,
    Eq,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    Quantifier,
    QuantifierKind,
    Variable,
)

from .norm_form import (
    C2NormalForm,
    CountDefinition,
    CountSection,
    ForallCountSection,
)


class NormalizeError(ValueError, WFOMCError):
    """Raised when a sentence is outside the supported C2 normalizer fragment."""


def normalize(
    sentence: Formula | C2NormalForm,
    *,
    reserved_predicate_names: Iterable[str] = (),
) -> C2NormalForm:
    if isinstance(sentence, C2NormalForm):
        from .validation import validate_normal_form

        validate_normal_form(sentence)
        return sentence
    if isinstance(sentence, Formula):
        return _normalize_typed_formula(
            sentence,
            reserved_predicate_names=frozenset(reserved_predicate_names),
        )
    raise TypeError(f"normalize expects Formula or C2NormalForm, got {type(sentence).__name__}")


def _normalize_typed_formula(
    sentence: Formula,
    *,
    reserved_predicate_names: frozenset[str],
) -> C2NormalForm:
    _validate_c2_source(sentence)
    normalized = _alpha_rename_shadowed_binders(sentence)
    normalized = _simplify_trivial_counts(normalized)
    normalizer = _TypedC2Normalizer(normalized, reserved_predicate_names)
    normalizer.add_top_level(normalized)
    normal_form = normalizer.finish()
    from .validation import validate_normal_form

    validate_normal_form(normal_form)
    return normal_form


@dataclass
class _TypedC2Normalizer:
    root: Formula
    reserved_predicate_names: frozenset[str] = frozenset()
    fresh_index: int = 0

    def __post_init__(self) -> None:
        self.ctx = context_for(self.root)
        self.used_predicate_names = {
            *(pred.name for pred in predicates(self.root)),
            *self.reserved_predicate_names,
        }
        self.qf_parts: list[Formula] = []
        self.forall_exists: list[object] = []
        self.forall_counts: list[ForallCountSection] = []
        self.exists: list[object] = []
        self.counts: list[CountSection] = []
        self.count_definitions: list[CountDefinition] = []
        self.requires_nonempty_domain = False

    def add_top_level(self, formula: object) -> None:
        if isinstance(formula, And):
            for conjunct in formula.args:
                self.add_top_level(conjunct)
            return
        if _is_direct_forall_count(formula):
            outer_var, count_formula = _direct_forall_body(formula)
            self.forall_counts.append(
                self._forall_count_section(
                    outer_var=outer_var,
                    count_formula=count_formula,
                )
            )
            return
        if _is_direct_forall_exists(formula):
            outer_var, exists_formula = _direct_forall_body(formula)
            inner_var = _single_quantified_variable(exists_formula)
            body = self.rewrite_embedded(exists_formula.body)
            self.forall_exists.append(
                self.ctx.forall(outer_var, self.ctx.exists(inner_var, body))
            )
            return
        if isinstance(formula, CountingQuantifier):
            self._ensure_base_count_comparator(formula)
            self.counts.append(
                self._count_section(
                    count_formula=formula,
                )
            )
            return
        if isinstance(formula, Quantifier) and formula.kind == QuantifierKind.EXISTS:
            var = _single_quantified_variable(formula)
            self.exists.append(self.ctx.exists(var, self.rewrite_embedded(formula.body)))
            return
        self.add_universal(self.rewrite_universal_prefix(formula))

    def rewrite_universal_prefix(self, formula: object) -> object:
        if isinstance(formula, Quantifier) and formula.kind == QuantifierKind.FORALL:
            _single_quantified_variable(formula)
            return self.rewrite_universal_prefix(formula.body)
        return self.rewrite_embedded(formula)

    def rewrite_embedded(self, formula: object) -> object:
        if isinstance(formula, (Atom, Eq, BoolConst)):
            return formula
        if isinstance(formula, Not):
            return self.ctx.neg(self.rewrite_embedded(formula.body))
        if isinstance(formula, And):
            return self.ctx.conjunction(
                *(self.rewrite_embedded(arg) for arg in formula.args)
            )
        if isinstance(formula, Or):
            return self.ctx.disjunction(
                *(self.rewrite_embedded(arg) for arg in formula.args)
            )
        if isinstance(formula, Implies):
            return self.ctx.implies(
                self.rewrite_embedded(formula.left),
                self.rewrite_embedded(formula.right),
            )
        if isinstance(formula, Iff):
            return self.ctx.iff(
                self.rewrite_embedded(formula.left),
                self.rewrite_embedded(formula.right),
            )
        if isinstance(formula, CountingQuantifier):
            simplified = _simplify_trivial_counts(formula)
            if simplified is not formula:
                return self.rewrite_embedded(simplified)
            return self._abstract_count(formula)
        if isinstance(formula, Quantifier):
            return self._abstract_quantifier(formula)
        return formula

    def finish(self) -> C2NormalForm:
        return C2NormalForm(
            qf_formula=_combine_qf_parts(tuple(self.qf_parts)),
            forall_exists=tuple(self.forall_exists),
            exists=tuple(self.exists),
            forall_counts=tuple(self.forall_counts),
            counts=tuple(self.counts),
            count_definitions=tuple(self.count_definitions),
            requires_nonempty_domain=self.requires_nonempty_domain,
        )

    def add_universal(self, formula: object | None) -> None:
        while isinstance(formula, Quantifier) and formula.kind == QuantifierKind.FORALL:
            formula = formula.body
        if not isinstance(formula, Formula):
            if formula is None:
                return
            raise NormalizeError("Universal definition must have a typed QF body")
        self.qf_parts.append(formula)

    def _abstract_count(self, formula: CountingQuantifier) -> object:
        self._ensure_base_count_comparator(formula)
        counted_var = formula.variable
        rest_vars = _sorted_vars(free_vars(formula.body) - {counted_var})
        if len(rest_vars) > 1:
            raise NormalizeError(
                "C2 counting subformulas may have at most one free variable "
                "besides the counted variable."
            )
        marker = self._fresh_atom("@c2_count", rest_vars)
        if rest_vars:
            section = self._forall_count_section(
                outer_var=rest_vars[0],
                count_formula=formula,
            )
        else:
            section = self._count_section(
                count_formula=formula,
            )
        self.count_definitions.append(
            CountDefinition(marker=marker, section=section)
        )
        return marker

    def _abstract_quantifier(self, formula: Quantifier) -> object:
        self.requires_nonempty_domain = True
        var = _single_quantified_variable(formula)
        body = self.rewrite_embedded(formula.body)
        rest_vars = _sorted_vars(free_vars(body) - {var})
        if len(rest_vars) > 1:
            raise NormalizeError(
                "C2 quantified subformulas may have at most one free variable "
                "besides the quantified variable."
            )
        marker = self._fresh_atom("@c2_quant", rest_vars)
        if formula.kind == QuantifierKind.EXISTS:
            self._define_exists_marker(marker, rest_vars, var, body)
        elif formula.kind == QuantifierKind.FORALL:
            self._define_forall_marker(marker, rest_vars, var, body)
        else:
            raise NormalizeError(f"Unsupported quantifier kind: {formula.kind!r}")
        return marker

    def _define_exists_marker(
        self,
        marker: object,
        rest_vars: tuple[object, ...],
        quantified_var: object,
        body: object,
    ) -> None:
        self.add_universal(
            self._forall_all(
                (*rest_vars, quantified_var),
                self.ctx.implies(body, marker),
            )
        )
        witness_body = self.ctx.disjunction(self.ctx.neg(marker), body)
        self._add_witness(rest_vars, quantified_var, witness_body)

    def _define_forall_marker(
        self,
        marker: object,
        rest_vars: tuple[object, ...],
        quantified_var: object,
        body: object,
    ) -> None:
        self.add_universal(
            self._forall_all(
                (*rest_vars, quantified_var),
                self.ctx.implies(marker, body),
            )
        )
        witness_body = self.ctx.disjunction(marker, _negate_quantifier_free(body))
        self._add_witness(rest_vars, quantified_var, witness_body)

    def _add_witness(
        self,
        rest_vars: tuple[object, ...],
        quantified_var: object,
        body: object,
    ) -> None:
        existential = self.ctx.exists(quantified_var, body)
        if rest_vars:
            self.forall_exists.append(self.ctx.forall(rest_vars[0], existential))
        else:
            self.exists.append(existential)

    def _count_section(
        self,
        *,
        count_formula: CountingQuantifier,
    ) -> CountSection:
        counted_var = count_formula.variable
        body = self._atomic_body_for_count(
            count_formula.body,
            variables=(counted_var,),
        )
        return CountSection(
            comparator=count_formula.comparator,
            count=count_formula.count,
            body=body,
            counted_var=counted_var,
        )

    def _forall_count_section(
        self,
        *,
        outer_var: object,
        count_formula: CountingQuantifier,
    ) -> ForallCountSection:
        self._ensure_base_count_comparator(count_formula)
        counted_var = count_formula.variable
        body = self._atomic_body_for_count(
            count_formula.body,
            variables=(outer_var, counted_var),
        )
        return ForallCountSection(
            comparator=count_formula.comparator,
            count=count_formula.count,
            body=body,
            outer_var=outer_var,
            counted_var=counted_var,
        )

    def _atomic_body_for_count(
        self,
        body: object,
        *,
        variables: tuple[object, ...],
    ) -> object:
        rewritten = self.rewrite_embedded(body)
        extra_free_vars = free_vars(rewritten) - set(variables)
        if extra_free_vars:
            raise NormalizeError(
                "Counting body contains variables outside the count scope: "
                f"{', '.join(str(var) for var in extra_free_vars)}"
            )
        if _is_canonical_atom(rewritten, variables):
            return rewritten
        relation_atom = self._fresh_atom("@c2_rel", variables)
        self.add_universal(
            self._forall_all(variables, self.ctx.iff(relation_atom, rewritten))
        )
        return relation_atom

    def _forall_all(self, variables: tuple[object, ...], body: object) -> object:
        if not variables:
            return body
        return self.ctx.forall(variables[0] if len(variables) == 1 else variables, body)

    def _fresh_atom(self, prefix: str, variables: tuple[object, ...]) -> object:
        while True:
            name = f"{prefix}_{self.fresh_index}"
            self.fresh_index += 1
            if name not in self.used_predicate_names:
                self.used_predicate_names.add(name)
                break
        predicate = self.ctx.predicate(name, len(variables))
        return self.ctx.atom(predicate, *variables)

    def _ensure_base_count_comparator(self, formula: CountingQuantifier) -> None:
        if formula.comparator not in {"=", "!=", "<", "<=", ">", ">=", "mod"}:
            raise NormalizeError(
                f"Unsupported count comparator: {formula.comparator!r}"
            )


def _alpha_rename_shadowed_binders(formula: Formula) -> Formula:
    renamer = _AlphaRenamer(formula)
    return renamer.rename(formula, env={}, bound=frozenset())


@dataclass
class _AlphaRenamer:
    root: object
    fresh_index: int = 0

    def __post_init__(self) -> None:
        self.ctx = context_for(self.root)
        self.used_variable_names = _all_variable_names(self.root)

    def rename(
        self,
        formula: object,
        *,
        env: dict[object, object],
        bound: frozenset[object],
    ) -> object:
        if isinstance(formula, Atom):
            return self.ctx.atom(
                formula.predicate,
                *tuple(self._rename_term(term, env) for term in formula.terms),
            )
        if isinstance(formula, Eq):
            return self.ctx.eq(
                self._rename_term(formula.left, env),
                self._rename_term(formula.right, env),
            )
        if isinstance(formula, Not):
            return self.ctx.neg(self.rename(formula.body, env=env, bound=bound))
        if isinstance(formula, And):
            return self.ctx.conjunction(
                *(self.rename(arg, env=env, bound=bound) for arg in formula.args)
            )
        if isinstance(formula, Or):
            return self.ctx.disjunction(
                *(self.rename(arg, env=env, bound=bound) for arg in formula.args)
            )
        if isinstance(formula, Implies):
            return self.ctx.implies(
                self.rename(formula.left, env=env, bound=bound),
                self.rename(formula.right, env=env, bound=bound),
            )
        if isinstance(formula, Iff):
            return self.ctx.iff(
                self.rename(formula.left, env=env, bound=bound),
                self.rename(formula.right, env=env, bound=bound),
            )
        if isinstance(formula, Quantifier):
            var = _single_quantified_variable(formula)
            new_var, body_env = self._binder(var, env=env, bound=bound)
            return self.ctx._quantifier(
                formula.kind,
                new_var,
                self.rename(formula.body, env=body_env, bound=bound | {var}),
            )
        if isinstance(formula, CountingQuantifier):
            var = formula.variable
            new_var, body_env = self._binder(var, env=env, bound=bound)
            return self.ctx.count(
                new_var,
                formula.comparator,
                formula.count,
                self.rename(formula.body, env=body_env, bound=bound | {var}),
            )
        return formula

    def _binder(
        self,
        var: object,
        *,
        env: dict[object, object],
        bound: frozenset[object],
    ) -> tuple[object, dict[object, object]]:
        if var not in bound:
            body_env = dict(env)
            body_env.pop(var, None)
            return var, body_env
        while True:
            name = f"{var}_c2_{self.fresh_index}"
            self.fresh_index += 1
            if name not in self.used_variable_names:
                self.used_variable_names.add(name)
                break
        renamed = self.ctx.variable(name)
        body_env = dict(env)
        body_env[var] = renamed
        return renamed, body_env

    def _rename_term(self, term: object, env: dict[object, object]) -> object:
        if term in env:
            return env[term]
        return term


def _simplify_trivial_counts(formula: Formula) -> Formula:
    if isinstance(formula, Not):
        return context_for(formula).neg(_simplify_trivial_counts(formula.body))
    if isinstance(formula, And):
        return context_for(formula).conjunction(
            *(_simplify_trivial_counts(arg) for arg in formula.args)
        )
    if isinstance(formula, Or):
        return context_for(formula).disjunction(
            *(_simplify_trivial_counts(arg) for arg in formula.args)
        )
    if isinstance(formula, Implies):
        return context_for(formula).implies(
            _simplify_trivial_counts(formula.left),
            _simplify_trivial_counts(formula.right),
        )
    if isinstance(formula, Iff):
        return context_for(formula).iff(
            _simplify_trivial_counts(formula.left),
            _simplify_trivial_counts(formula.right),
        )
    if isinstance(formula, Quantifier):
        _single_quantified_variable(formula)
        return context_for(formula)._quantifier(
            formula.kind,
            formula.variables,
            _simplify_trivial_counts(formula.body),
        )
    if isinstance(formula, CountingQuantifier):
        ctx = context_for(formula)
        body = _simplify_trivial_counts(formula.body)
        if _is_always_true_count(formula):
            return ctx.true()
        if _is_always_false_count(formula):
            return ctx.false()
        if _is_zero_count(formula) and formula.comparator in {"=", "<="}:
            return ctx.forall(
                formula.variable,
                ctx.neg(body),
            )
        if _is_positive_existence_count(formula):
            return ctx.exists(formula.variable, body)
        return ctx.count(formula.variable, formula.comparator, formula.count, body)
    return formula


def _negate_quantifier_free(formula: Formula) -> Formula:
    return context_for(formula).neg(formula)


def _is_zero_count(formula: CountingQuantifier) -> bool:
    return formula.count == 0 and not isinstance(formula.count, bool)


def _is_positive_existence_count(formula: CountingQuantifier) -> bool:
    return (
        formula.comparator == ">"
        and formula.count == 0
        and not isinstance(formula.count, bool)
    ) or (
        formula.comparator == ">="
        and formula.count == 1
        and not isinstance(formula.count, bool)
    ) or (
        formula.comparator == "!="
        and formula.count == 0
        and not isinstance(formula.count, bool)
    )


def _is_always_true_count(formula: CountingQuantifier) -> bool:
    return (
        formula.comparator == ">="
        and formula.count == 0
        and not isinstance(formula.count, bool)
    )


def _is_always_false_count(formula: CountingQuantifier) -> bool:
    return (
        formula.comparator == "<"
        and formula.count == 0
        and not isinstance(formula.count, bool)
    )


def _is_direct_forall_count(formula: object) -> bool:
    if not isinstance(formula, Quantifier) or formula.kind != QuantifierKind.FORALL:
        return False
    _single_quantified_variable(formula)
    return isinstance(formula.body, CountingQuantifier)


def _is_direct_forall_exists(formula: object) -> bool:
    if not isinstance(formula, Quantifier) or formula.kind != QuantifierKind.FORALL:
        return False
    _single_quantified_variable(formula)
    return (
        isinstance(formula.body, Quantifier)
        and formula.body.kind == QuantifierKind.EXISTS
    )


def _direct_forall_body(formula: object) -> tuple[object, object]:
    if not isinstance(formula, Quantifier):
        raise TypeError("Expected a quantified formula.")
    return _single_quantified_variable(formula), formula.body


def _single_quantified_variable(formula: Quantifier) -> Variable:
    if len(formula.variables) != 1:
        raise NormalizeError(
            "Multi-variable quantifier lists are not supported by the C2 "
            "normalizer yet."
        )
    variable = formula.variables[0]
    if not isinstance(variable, Variable):
        raise NormalizeError("Quantifier binders must be typed Variables.")
    return variable


def _sorted_vars(vars_: object) -> tuple[object, ...]:
    return tuple(sorted(vars_, key=str))


def _is_canonical_atom(formula: object, variables: tuple[object, ...]) -> bool:
    return isinstance(formula, Atom) and formula.terms == variables


def _combine_qf_parts(parts: tuple[Formula, ...]) -> Formula | None:
    if not parts:
        return None
    return reduce(context_for(*parts).conjunction, parts)


def _validate_c2_source(formula: Formula) -> None:
    """Reject syntax outside the two-variable, unary/binary solver fragment."""

    from wfomc.fol import Atom

    def visit(node: object) -> None:
        if isinstance(node, Formula):
            scoped_variables = free_vars(node)
            if len(scoped_variables) > 2:
                names = ", ".join(sorted(str(var) for var in scoped_variables))
                raise NormalizeError(
                    f"C2 subformula uses more than two variables: {names}"
                )
        if isinstance(node, Atom) and node.predicate.arity > 2:
            raise NormalizeError(
                f"C2 solver normal form supports predicate arity at most 2: "
                f"{node.predicate.name}/{node.predicate.arity}"
            )
        if isinstance(node, Quantifier):
            _single_quantified_variable(node)
        elif isinstance(node, CountingQuantifier) and not isinstance(
            node.variable, Variable
        ):
            raise NormalizeError("Counting binders must be typed Variables.")
        for child in formula_children(node):
            visit(child)

    visit(formula)
    unbound = free_vars(formula)
    if unbound:
        names = ", ".join(sorted(str(variable) for variable in unbound))
        raise NormalizeError(
            f"C2 normalizer requires a closed sentence; unbound variables: {names}"
        )


def _all_variable_names(formula: object) -> set[str]:
    from wfomc.fol import Variable

    names: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, Atom):
            names.update(term.name for term in node.terms if isinstance(term, Variable))
        elif isinstance(node, Eq):
            names.update(
                term.name
                for term in (node.left, node.right)
                if isinstance(term, Variable)
            )
        if isinstance(node, Quantifier):
            names.update(
                variable.name
                for variable in node.variables
                if isinstance(variable, Variable)
            )
        elif isinstance(node, CountingQuantifier) and isinstance(
            node.variable, Variable
        ):
            names.add(node.variable.name)
        for child in formula_children(node):
            visit(child)

    visit(formula)
    return names


__all__ = ["NormalizeError", "normalize"]
