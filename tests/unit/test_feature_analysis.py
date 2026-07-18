from __future__ import annotations

from pathlib import Path

from wfomc.engine.features import analyze_features
from wfomc.fol.normal_form import normalize
from wfomc.parser import parse_input, parse_problem
from wfomc.problem import Problem


ROOT = Path(__file__).parents[2]


def test_unary_evidence_constants_do_not_count_as_sentence_constants():
    problem = parse_input(str(ROOT / "models" / "unary_evidence" / "evidence-only.wfomcs"))

    features = analyze_features(problem)

    assert features.has_unary_evidence
    assert not features.has_named_constants
    assert features.named_constants == ()


def test_sentence_constants_are_reported_separately_from_unary_evidence():
    problem = parse_problem(r"""
\forall X: (R(X, domain0))

domain = {domain0, domain1}

P(domain0)
""")

    features = analyze_features(problem)

    assert features.has_unary_evidence
    assert features.has_named_constants
    assert features.named_constants == ("domain0",)


def test_pred_substring_in_regular_predicate_name_is_not_predk_feature():
    problem = parse_problem(r"""
\forall X: (BLOWPRED(X) | ~BLOWPRED(X))

domain = 2
""")

    features = analyze_features(problem)

    assert not features.has_linear_order
    assert not features.has_predk
    assert not features.has_circular_pred


def test_typed_order_predicates_drive_feature_analysis():
    from wfomc.fol import atom, conjunction, forall
    from wfomc.fol import Variable

    x = Variable("X")
    y = Variable("Y")
    sentence = conjunction(
        forall(x, forall(y, atom("LEQ", x, y))),
        forall(x, forall(y, atom("PRED", x, y))),
    )
    normal_form = normalize(sentence)

    features = analyze_features(Problem(sentence=sentence), normal_form)

    assert features.has_linear_order
    assert features.has_predk
    assert not features.has_circular_pred
    assert features.leq_predicate.name == "LEQ"
    assert dict(features.predecessor_predicates)[1].name == "PRED"


def test_typed_order_predicates_do_not_need_normal_form_for_feature_analysis():
    from wfomc.fol import atom, conjunction, forall
    from wfomc.fol import Variable

    x = Variable("X")
    y = Variable("Y")
    sentence = conjunction(
        forall(x, atom("LEQ", x, y)),
        forall(x, atom("CIRCULAR_PRED", x, y)),
    )

    features = analyze_features(Problem(sentence=sentence))

    assert features.has_linear_order
    assert features.has_predk
    assert features.has_circular_pred
    assert features.leq_predicate.name == "LEQ"
    assert dict(features.predecessor_predicates)[1].name == "CIRCULAR_PRED"
    assert features.circular_predecessor_predicate.name == "CIRCULAR_PRED"


def test_typed_order_predicate_names_still_require_binary_arity():
    from wfomc.fol import atom, conjunction, forall
    from wfomc.fol import Variable

    x = Variable("X")
    sentence = conjunction(
        forall(x, atom("LEQ", x)),
        forall(x, atom("PRED", x)),
    )

    features = analyze_features(Problem(sentence=sentence))

    assert not features.has_linear_order
    assert not features.has_predk
    assert not features.has_circular_pred


def test_typed_counting_does_not_need_normal_form_for_feature_analysis():
    from wfomc.fol import atom, count, forall
    from wfomc.fol import Variable

    x = Variable("X")
    y = Variable("Y")
    sentence = forall(x, count(y, "=", 2, atom("R", x, y)))

    features = analyze_features(Problem(sentence=sentence))

    assert features.has_c2_counting
    assert not features.has_mod_counting


def test_typed_mod_counting_does_not_need_string_fallback():
    from wfomc.fol import atom, count, forall
    from wfomc.fol import Variable

    x = Variable("X")
    y = Variable("Y")
    sentence = forall(x, count(y, "mod", (1, 3), atom("R", x, y)))

    features = analyze_features(Problem(sentence=sentence))

    assert features.has_c2_counting
    assert features.has_mod_counting


def test_binary_evidence_is_reported_as_a_feature():
    from wfomc.evidence import BinaryEvidence, Evidence, GroundBinaryLiteral
    from wfomc.fol import Constant, Predicate, true

    predicate = Predicate("R", 2)
    problem = Problem(
        sentence=true(),
        evidence=Evidence(
            binary=BinaryEvidence(
                (GroundBinaryLiteral(predicate, Constant("a"), Constant("b")),)
            )
        ),
    )

    features = analyze_features(problem)

    assert features.has_binary_evidence
