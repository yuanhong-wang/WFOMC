"""Problem parser boundary tests; formula grammar lives in test_fol_syntax."""

from pathlib import Path

from wfomc.cardinality_constraints import CardinalityConstraints, Comparator
from wfomc.fol import Constant, Formula, FormulaKind, Predicate, Variable
from wfomc.parser import parse_problem, parse_problem_file
from wfomc.problem import Problem


ROOT = Path(__file__).resolve().parents[2]


def test_parse_minimal_problem_uses_typed_defaults():
    parsed = parse_problem(r"\forall X: P(X)")

    assert isinstance(parsed, Problem)
    assert isinstance(parsed.sentence, Formula)
    assert parsed.sentence.op is FormulaKind.FORALL
    assert parsed.domain == frozenset()
    assert parsed.weights == {}
    assert parsed.cardinality_constraints == CardinalityConstraints()
    assert parsed.evidence.unary.is_empty


def test_parse_full_wfomcs_preserves_typed_signature_and_evidence():
    parsed = parse_problem(
        r"""
\forall X: P(X)
domain = {a, b}
2 3 P
P(a)
"""
    )
    quantified_variable, body = parsed.sentence.args
    predicate, term = body.args
    literal = parsed.evidence.unary.literals[0]

    assert isinstance(quantified_variable, Variable)
    assert isinstance(predicate, Predicate)
    assert isinstance(term, Variable)
    assert all(isinstance(constant, Constant) for constant in parsed.domain)
    assert parsed.weights[predicate] == (2, 3)
    assert isinstance(literal.predicate, Predicate)
    assert str(literal.predicate(literal.constant)) == "P(a)"


def test_parse_cardinality_constraint_uses_exact_integer_terms():
    parsed = parse_problem(
        r"""
\forall X: (P(X) | Q(X))
domain = 3
1 1 P
1 1 Q
2 |P| - |Q| <= 2
"""
    )

    constraint = parsed.cardinality_constraints.constraints[0]
    assert [(term.predicate.name, term.coefficient) for term in constraint.terms] == [
        ("P", 2),
        ("Q", -1),
    ]
    assert constraint.comparator is Comparator.LE
    assert constraint.rhs == 2


def test_parse_wfomcs_file_records_source_and_unary_evidence():
    path = ROOT / "models" / "unary_evidence" / "evidence-only.wfomcs"

    parsed = parse_problem_file(path)

    assert str(parsed.sentence) == r"\forall X: T(X)"
    assert len(parsed.domain) == 3
    assert _evidence_strings(parsed) == {"P(domain0)"}
    assert parsed.options["source_path"] == str(path)


def test_parse_mln_file_preserves_weights_constraints_and_evidence():
    path = ROOT / "models" / "unary_evidence" / "molecule.mln"

    parsed = parse_problem_file(path)

    assert len(parsed.domain) == 5
    assert len(parsed.weights) == 1
    constraint = parsed.cardinality_constraints.constraints[0]
    assert constraint.comparator is Comparator.EQ
    assert constraint.rhs == 0
    assert {
        (term.predicate.name, term.coefficient) for term in constraint.terms
    } == {("AuxBond", 1), ("H", -1)}
    assert _evidence_strings(parsed) == {"C(atoms0)"}
    assert parsed.options["source_path"] == str(path)


def _evidence_strings(problem: Problem) -> set[str]:
    return {
        str(literal.predicate(literal.constant))
        if literal.positive
        else str(~literal.predicate(literal.constant))
        for literal in problem.evidence.unary.literals
    }
