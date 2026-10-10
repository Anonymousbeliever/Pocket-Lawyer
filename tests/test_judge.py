"""
The tier-2 answer-quality judge.

The judge's own prompt cannot be unit tested - it needs the API. What CAN
be tested is everything around it, and that matters more than it sounds:
the surrounding logic decides whether a judge outage reads as a quality
regression, and whether a supported non-sequitur counts as a pass.
"""

import json
from types import SimpleNamespace

import pytest

from evaluation.judge import (
    CLAIM_CHECK,
    INSTRUCTIONS,
    JUDGE_SCHEMA,
    AnswerJudge,
    headline,
    load_cases,
    quality,
    score_case,
)
from evaluation.run import write_review
from evaluation.schema import Question, QuestionResult


def check(finding: str = "stated", reason: str = "") -> dict:
    return {"section_says": "words", "finding": finding, "reason": reason}


def verdict(**overrides) -> dict:
    base = {
        "claims": [check(), check()],
        "responsive": True,
        "defect": "none",
        "reason": "",
        "error": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------

@pytest.mark.parametrize("schema", [JUDGE_SCHEMA, CLAIM_CHECK])
def test_schema_satisfies_openai_strict_mode(schema):
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["additionalProperties"] is False


def test_the_judge_copies_the_law_before_deciding():
    """
    `section_says` comes first so the finding rests on words the judge
    actually read. Strict mode generates properties in schema order.
    """

    assert list(CLAIM_CHECK["properties"])[0] == "section_says"


def test_the_judge_is_never_asked_about_legal_correctness():
    """
    The restriction that makes this judge workable. An LLM judge was tried
    for reranking and failed precisely by reasoning about Kenyan law -
    returning bail-pending-appeal sections for a pre-trial question. This
    one is asked only what it can read.
    """

    lowered = INSTRUCTIONS.lower()

    assert "do not judge whether the law is correct" in lowered
    assert "you are a careful reader, not a" in lowered


# ---------------------------------------------------------
# WHAT COUNTS AS QUALITY
# ---------------------------------------------------------

def test_a_sound_answer_passes():
    assert quality(verdict()) is True


def test_a_supported_non_sequitur_still_fails():
    """
    The production case. Asked how to leave property to one's children,
    the system answered with a point about adopted children's inheritance
    rights - drawn accurately from a real cited chunk. Supported, and not
    an answer.
    """

    assert quality(
        verdict(responsive=False, defect="non_sequitur")
    ) is False


@pytest.mark.parametrize(
    "finding",
    ["adds_to_source", "wrong_actor", "overstates_certainty", "drops_condition"],
)
def test_one_claim_saying_more_than_its_law_fails_the_answer(finding):
    """
    The four failures of 2026-09-27, one per finding. Each answer was
    otherwise real law, cited correctly - and the one overstated claim is
    the one a person would act on.
    """

    assert quality(verdict(claims=[check(), check(finding)])) is False


def test_an_answer_level_defect_fails_even_when_every_claim_is_stated():
    assert quality(verdict(defect="unsupported_verdict")) is False


# ---------------------------------------------------------
# A JUDGE OUTAGE IS NOT A QUALITY REGRESSION
# ---------------------------------------------------------

def test_a_judge_error_is_not_scored_as_a_failure():
    """
    None, not False. Counting an API timeout as a bad answer would turn
    an outage into a metric collapse and send someone hunting a
    regression that never happened.
    """

    assert quality(verdict(error="APIError: timeout", responsive=None)) is None


def test_an_unjudged_answer_is_none():
    assert quality(None) is None
    assert quality({}) is None


def fake_judge(response: dict, model_name: str = "test") -> AnswerJudge:
    """
    An AnswerJudge whose API call returns `response`, with no key needed.
    Every request it would have sent is kept on `judge.sent`.
    """

    judge = object.__new__(AnswerJudge)
    judge.model_name = model_name
    judge.sent = []

    def create(**request):
        judge.sent.append(request)
        return SimpleNamespace(output_text=json.dumps(response))

    judge.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return judge


SOURCE = {"citation": "s.1", "unit_title": "Title", "content": "Text."}


def test_a_judge_that_skips_a_claim_cannot_pass_the_answer():
    """
    The claim it skipped is as likely as any other to be the bad one, so a
    partial check is an error, never a pass.
    """

    response = verdict(claims=[check()])
    del response["error"]

    result = fake_judge(response).judge(
        question="Q?",
        answer="A.",
        claims=["one", "two"],
        sources=[SOURCE],
    )

    assert "1 of 2" in result["error"]
    assert quality(result) is None


# ---------------------------------------------------------
# CHOOSING THE JUDGE MODEL
# ---------------------------------------------------------

@pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini", "O3-mini"])
def test_reasoning_models_are_sent_no_temperature(model):
    """They reject the parameter outright, so sending it fails the call."""

    judge = fake_judge(verdict(claims=[check()]), model_name=model)

    judge.judge(question="Q?", answer="A.", claims=["one"], sources=[SOURCE])

    assert "temperature" not in judge.sent[0]
    assert judge.sent[0]["model"] == model


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini"])
def test_other_models_are_judged_at_temperature_zero(model):
    judge = fake_judge(verdict(claims=[check()]), model_name=model)

    judge.judge(question="Q?", answer="A.", claims=["one"], sources=[SOURCE])

    assert judge.sent[0]["temperature"] == 0


def test_a_model_merely_containing_o_is_not_a_reasoning_model():
    from evaluation.judge import is_reasoning_model

    assert not is_reasoning_model("gpt-4o")
    assert not is_reasoning_model("omni-something")


def test_an_answer_with_no_claims_fails_without_calling_the_model():
    result = fake_judge({}).judge(
        question="Q?", answer="A.", claims=[], sources=[SOURCE]
    )

    assert quality(result) is False


# ---------------------------------------------------------
# THE HEADLINE THE REPORT SHOWS
# ---------------------------------------------------------

def test_the_headline_names_the_failing_claim():
    defect, reason = headline(
        verdict(claims=[check(), check("wrong_actor", "court, not police")])
    )

    assert defect == "wrong_actor"
    assert reason == "claim 2: court, not police"


def test_an_answer_level_defect_outranks_a_claim_defect():
    defect, _ = headline(
        verdict(
            defect="non_sequitur",
            responsive=False,
            claims=[check("drops_condition")],
        )
    )

    assert defect == "non_sequitur"


def test_an_unresponsive_answer_is_never_headlined_as_sound():
    """The model is told not to, but a "none" here must not read as a pass."""

    defect, _ = headline(verdict(responsive=False, defect="none"))

    assert defect == "unresponsive"


def test_a_sound_answer_headlines_none():
    assert headline(verdict())[0] == "none"


# ---------------------------------------------------------
# THE VALIDATION CASES
#
# The judge's prompt cannot be unit tested - it needs the API. These keep
# the cases it is measured against honest.
# ---------------------------------------------------------

def test_cases_include_both_bad_and_faithful_answers():
    """
    A judge tested only on bad answers passes by failing everything. The
    faithful answers are what catch that.
    """

    cases = load_cases()

    assert any(c["expect_quality"] is False for c in cases)
    assert any(c["expect_quality"] is True for c in cases)


def test_every_case_expectation_is_well_formed():
    for case in load_cases():
        assert case["cites"], case["id"]
        assert case["claims"], case["id"]

        for claim in case["claims"]:
            assert claim.get("expect") in (None, "flagged", "sound"), case["id"]

        # A faithful answer with a flagged claim would contradict itself.
        if case["expect_quality"]:
            assert all(c.get("expect") == "sound" for c in case["claims"])


def test_every_case_cites_real_chunks():
    from evaluation.schema import corpus_chunk_ids

    known = corpus_chunk_ids()

    if not known:
        pytest.skip("no ingested corpus on disk to validate against")

    for case in load_cases():
        assert set(case["cites"]) <= known, case["id"]


def test_no_case_is_copied_into_the_prompt():
    """
    A judge shown the test answers as examples would be measuring its
    memory, not its reading.
    """

    for case in load_cases():
        for claim in case["claims"]:
            assert claim["text"] not in INSTRUCTIONS, case["id"]


def case(expect_quality: bool, *expects: str | None) -> dict:
    return {
        "expect_quality": expect_quality,
        "claims": [{"text": "c", "expect": e} for e in expects],
    }


def test_a_missed_defect_is_a_disagreement():
    checks = score_case(
        case(False, "flagged"),
        verdict(claims=[check("stated")]),
    )

    assert [ok for ok, _ in checks] == [False, False]


def test_unasserted_claims_are_not_scored():
    checks = score_case(
        case(False, "flagged", None),
        verdict(claims=[check("wrong_actor"), check("stated")]),
    )

    assert len(checks) == 2
    assert all(ok for ok, _ in checks)


def test_a_passed_defect_shows_what_the_judge_read(capsys):
    """
    The first validation run hid this: reasons printed only for claims the
    judge flagged, so a missed defect could not be diagnosed.
    """

    from evaluation.judge import _print_judge_evidence

    _print_judge_evidence(
        {
            "expect_quality": False,
            "claims": [{"text": "The police must return it.",
                        "expect": "flagged"}],
        },
        verdict(
            claims=[
                {"section_says": "the court shall direct",
                 "finding": "stated",
                 "reason": "Matches the section."}
            ]
        ),
    )

    out = capsys.readouterr().out

    assert "claim 1 (DISAGREES) - judge: stated" in out
    assert "The police must return it." in out
    assert '"the court shall direct"' in out
    assert "Matches the section." in out
    assert "responsive=True" in out, "the answer-level miss must show too"


def test_agreed_stated_claims_are_not_printed(capsys):
    from evaluation.judge import _print_judge_evidence

    _print_judge_evidence(
        {"expect_quality": True,
         "claims": [{"text": "Fine.", "expect": "sound"}]},
        verdict(claims=[check()]),
    )

    assert capsys.readouterr().out == ""


def test_a_judge_error_never_counts_as_agreement():
    checks = score_case(
        case(False, "flagged"),
        {"error": "timeout", "claims": [], "responsive": None},
    )

    assert not any(ok for ok, _ in checks)


# ---------------------------------------------------------
# THE REVIEW SAMPLE
# ---------------------------------------------------------

def results(passes: int, failures: int) -> list[QuestionResult]:
    made = []

    for index in range(failures):
        made.append(
            QuestionResult(
                id=f"fail-{index}",
                answerable=True,
                answer_quality=False,
                judge_defect="non_sequitur",
                judge_reason="does not address the question",
            )
        )

    for index in range(passes):
        made.append(
            QuestionResult(
                id=f"pass-{index}",
                answerable=True,
                answer_quality=True,
                judge_defect="none",
            )
        )

    return made


def questions_for(items: list[QuestionResult]) -> list[Question]:
    return [
        Question(id=r.id, question=f"question {r.id}", answerable=True)
        for r in items
    ]


def test_review_mixes_failures_with_passes(tmp_path, monkeypatch):
    """
    A sample of nothing but rejections cannot reveal a FALSE rejection,
    which is the failure mode that would quietly make the system look
    worse than it is. Failures are capped at half.
    """

    import evaluation.run as run

    monkeypatch.setattr(
        run, "__file__", str(tmp_path / "run.py"), raising=False
    )

    items = results(passes=10, failures=10)
    path = write_review(questions_for(items), items, limit=6)

    text = path.read_text(encoding="utf-8")

    assert text.count("## fail-") == 3
    assert text.count("## pass-") == 3

    path.unlink()


def test_review_handles_fewer_judged_answers_than_asked_for(monkeypatch):
    items = results(passes=1, failures=1)
    path = write_review(questions_for(items), items, limit=20)

    text = path.read_text(encoding="utf-8")

    assert text.count("## ") == 2

    path.unlink()


def test_review_skips_unjudged_answers(monkeypatch):
    """Refusals and judge errors carry no verdict to review."""

    items = results(passes=1, failures=1)
    items.append(QuestionResult(id="refused", answerable=False))

    path = write_review(questions_for(items), items, limit=10)

    assert "## refused" not in path.read_text(encoding="utf-8")

    path.unlink()
