"""
The structured answer contract.

Polarity is fixed by construction here: the model fills a `verdict`
field and code writes the opening word, so the stated verdict and the
explanation cannot disagree.
"""

import json

import pytest

from backend.app.ai.answer import (
    ADVOCATE_NOTE,
    ANSWER_SCHEMA,
    GROUNDED_ITEM,
    INSUFFICIENT_ANSWER,
    MIN_QUOTE_CHARS,
    Grounded,
    LegalAnswer,
    cited_sources,
    quote_supports,
    render,
    verify_citations,
)


ARTICLE_49 = "constitution-of-kenya-2010@v2010-chapter-four-article-49"
ARTICLE_51 = "constitution-of-kenya-2010@v2010-chapter-four-article-51"

# Real text, so the quotes below are genuinely verifiable rather than
# matching a stub that would match anything.
ARTICLE_49_TEXT = (
    "(1) An arrested person has the right— (a) to be informed "
    "promptly, in a language that the person understands, of— (i) the "
    "reason for the arrest; (ii) the right to remain silent; and (iii) "
    "the consequences of not remaining silent."
)

ARTICLE_51_TEXT = (
    "A person who is detained, held in custody or imprisoned under the "
    "law, retains all the rights and fundamental freedoms in the Bill of "
    "Rights, except to the extent that any particular right is "
    "incompatible with the fact of being detained."
)

SUPPORTED_QUOTE = (
    "to be informed promptly, in a language that the person understands"
)


def sources() -> list[dict]:
    return [
        {
            "chunk_id": ARTICLE_49,
            "unit_title": "Rights of arrested persons",
            "content": ARTICLE_49_TEXT,
        },
        {
            "chunk_id": ARTICLE_51,
            "unit_title": "Rights of persons detained",
            "content": ARTICLE_51_TEXT,
        },
    ]


def grounded(
    text: str = (
        "Article 49(1)(a) gives an arrested person the right to be "
        "informed promptly of the reason for the arrest."
    ),
    chunk_id: str = ARTICLE_49,
    quote: str = SUPPORTED_QUOTE,
) -> Grounded:
    return Grounded(text=text, chunk_id=chunk_id, quote=quote)


def answer(**overrides) -> LegalAnswer:
    defaults = dict(
        sufficient=True,
        question_type="polar",
        verdict="no",
        statements=[grounded()],
        qualifications=[],
        missing="",
    )
    defaults.update(overrides)
    return LegalAnswer(**defaults)


# ---------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------

def test_schema_satisfies_openai_strict_mode():
    """Strict mode requires every property listed in `required`."""

    properties = set(ANSWER_SCHEMA["properties"])
    required = set(ANSWER_SCHEMA["required"])

    assert properties == required
    assert ANSWER_SCHEMA["additionalProperties"] is False


def test_verdict_enum_covers_non_polar_questions():
    verdicts = ANSWER_SCHEMA["properties"]["verdict"]["enum"]

    assert "not_applicable" in verdicts
    assert {"yes", "no", "it_depends"} <= set(verdicts)


# ---------------------------------------------------------
# POLARITY — THE REGRESSION
# ---------------------------------------------------------

def test_polar_no_answer_opens_with_no():
    """
    Regression: "Can police arrest me without telling me why?" was
    answered "Yes, the police cannot arrest you without telling you
    why." The opening word is now written by code from the verdict
    field, so it cannot contradict the explanation.
    """

    text = render(answer(verdict="no"))

    assert text.startswith("No.")
    assert not text.startswith("Yes")


def test_polar_yes_answer_opens_with_yes():
    text = render(answer(verdict="yes"))

    assert text.startswith("Yes.")


def test_it_depends_is_rendered():
    text = render(answer(verdict="it_depends"))

    assert text.startswith("It depends.")


def test_open_question_gets_no_polarity_prefix():
    text = render(
        answer(question_type="open", verdict="not_applicable")
    )

    assert not text.startswith(("Yes", "No", "It depends"))
    assert text.startswith("Article 49")


def test_qualifications_are_rendered():
    text = render(
        answer(
            qualifications=[
                grounded(text="Subject to the twenty-four hour limit.")
            ]
        )
    )

    assert "Important qualifications:" in text
    assert "- Subject to the twenty-four hour limit." in text


# ---------------------------------------------------------
# REFUSAL
# ---------------------------------------------------------

def test_insufficient_answer_renders_the_refusal():
    text = render(
        answer(
            sufficient=False,
            verdict="not_applicable",
            statements=[],
            missing="The sources do not address divorce.",
        )
    )

    assert text.startswith(INSUFFICIENT_ANSWER)
    assert "do not address divorce" in text


def test_every_answer_ends_with_the_advocate_note():
    """
    Information, not advice. No section says "consult an advocate", so the
    model cannot write it under grounding - code must.
    """

    for verdict, question_type in [("no", "polar"), ("not_applicable", "open")]:
        text = render(answer(verdict=verdict, question_type=question_type))

        assert text.endswith(ADVOCATE_NOTE)
        assert text.count(ADVOCATE_NOTE) == 1


def test_a_refusal_does_not_repeat_the_advocate_pointer():
    """The refusal already sends the user to an advocate."""

    text = render(answer(sufficient=False, statements=[]))

    assert ADVOCATE_NOTE not in text
    assert "advocate" in text


def test_the_judge_reads_the_answer_without_the_note():
    text = render(answer(), advocate_note=False)

    assert ADVOCATE_NOTE not in text
    assert text.startswith("No.")


def test_refusal_never_opens_with_a_verdict():
    """A false `sufficient` wins even if verdict was filled in."""

    text = render(answer(sufficient=False, verdict="no"))

    assert not text.startswith("No.")


# ---------------------------------------------------------
# GROUNDING
#
# The check that makes a claim verifiable rather than merely attributed.
# Its failure modes are cheap to provoke here and expensive to discover
# in production, so they are all covered.
# ---------------------------------------------------------

def test_the_grounded_item_schema_satisfies_strict_mode():
    assert set(GROUNDED_ITEM["properties"]) == set(GROUNDED_ITEM["required"])
    assert GROUNDED_ITEM["additionalProperties"] is False


def test_a_real_quote_is_accepted():
    assert quote_supports(SUPPORTED_QUOTE, ARTICLE_49_TEXT)


def test_a_fabricated_quote_is_rejected():
    """The failure this whole mechanism exists to catch."""

    assert not quote_supports(
        "must be told at the exact moment of arrest without any delay",
        ARTICLE_49_TEXT,
    )


def test_a_quote_below_the_length_floor_is_rejected():
    """
    Without a floor, "the" matches every passage and the check is
    theatre rather than verification.
    """

    assert len("the reason") < MIN_QUOTE_CHARS
    assert not quote_supports("the reason", ARTICLE_49_TEXT)


def test_whitespace_and_dash_variants_still_match():
    """
    A model reproducing a passage faithfully may re-wrap whitespace or
    straighten the em dashes Kenya Law's typesetting uses. Neither
    means the quote was invented.
    """

    assert quote_supports(
        "An arrested person has the right-  (a) to be informed promptly",
        ARTICLE_49_TEXT,
    )


def test_case_differences_still_match():
    assert quote_supports(SUPPORTED_QUOTE.upper(), ARTICLE_49_TEXT)


def test_a_grounded_statement_survives_verification():
    result = verify_citations(answer(), sources())

    assert result.sufficient is True
    assert len(result.statements) == 1
    assert result.ungrounded == []
    assert result.cited_chunk_ids == [ARTICLE_49]


def test_an_ungrounded_statement_is_dropped():
    """
    The drop itself. Half the claims failing also trips the refusal rule
    below, which is deliberate - see
    `test_an_answer_whose_claims_mostly_failed_becomes_a_refusal`.
    """

    result = verify_citations(
        answer(
            statements=[
                grounded(),
                grounded(
                    text="You must be told at the exact moment of arrest.",
                    quote="told at the exact moment of arrest, without delay",
                ),
            ]
        ),
        sources(),
    )

    assert len(result.statements) == 1
    assert len(result.ungrounded) == 1

    # An even split means half of what the model asserted was invented.
    # Accuracy outranks usefulness, so that refuses.
    assert result.sufficient is False


def test_a_quote_against_a_source_that_was_not_retrieved_is_dropped():
    """An invented citation and an invented claim get the same treatment."""

    result = verify_citations(
        answer(
            statements=[
                grounded(chunk_id="employment-act-2007-section-45")
            ]
        ),
        sources(),
    )

    assert result.statements == []
    assert len(result.ungrounded) == 1


# ---------------------------------------------------------
# ELIDED QUOTES
#
# Found in production, not in a fixture. Legal text is enumerated, so the
# words that state a rule are usually a stem plus ONE of its paragraphs,
# with other paragraphs in between. Requiring one unbroken span rejected
# every claim the model made about a real section and turned a genuine
# answer into a refusal.
# ---------------------------------------------------------

ENUMERATED = (
    "Where a judgment debtor fails to pay to the decree holder any sum "
    "specified in a decree or order, the Court\nmay—\n(a)\nwhere the "
    "judgment debtor's movable property is insufficient to satisfy the "
    "decree, order execution\nby attachment and sale of the judgment "
    "debtor's immovable property;\n(b)\norder the attachment of the salary "
    "of the judgement debtor; or"
)


def test_a_quote_may_elide_between_a_stem_and_its_paragraph():
    """The production failure this rule exists to fix."""

    assert quote_supports(
        "the Court may ... order the attachment of the salary of the "
        "judgement debtor",
        ENUMERATED,
    )


def test_fragments_must_appear_in_source_order():
    """Order is what stops a meaning being assembled backwards."""

    assert not quote_supports(
        "order the attachment of the salary of the judgement debtor ... "
        "Where a judgment debtor fails to pay",
        ENUMERATED,
    )


def test_stitching_single_words_is_rejected():
    """
    "the Court ... may ... not ... order execution" clears the total
    length floor while asserting the opposite of the source. The fragment
    count is what excludes it - a floor high enough to catch "not" would
    also catch the legitimate thirteen-character stem "the Court may".
    """

    assert not quote_supports(
        "the Court ... may ... not ... order execution", ENUMERATED
    )


def test_a_quote_may_omit_the_paragraph_marker():
    """
    Found in production, and the reason `writing-a-will` became a false
    refusal. The model quoted "No written will shall be valid unless-the
    testator has signed", where the source reads "...unless- (a) the
    testator has signed...". Every word verbatim and in order; only the
    marker absent.

    Markers are dropped from both sides. That is safe in a way general
    gap-skipping is not - "(a)" carries no meaning, so eliding it cannot
    change what the passage says, whereas skipping arbitrary text could
    drop a "not".
    """

    source = (
        "No written will shall be valid unless—\n(a)\nthe testator has "
        "signed or affixed his mark to the will, or it has been signed by "
        "some other person in the presence and by the direction of the "
        "testator;"
    )

    assert quote_supports(
        "No written will shall be valid unless—the testator has signed "
        "or affixed his mark to the will",
        source,
    )


def test_an_elided_fragment_must_still_be_real():
    assert not quote_supports(
        "the Court may ... imprison the judgment debtor for six months",
        ENUMERATED,
    )


# ---------------------------------------------------------
# QUOTES CUT SHORT WITH A FULL STOP
#
# Found in production: `product-env-factory-pollution` and
# `product-con-tribe-discrimination` were refused because four verbatim
# quotes ended in "." where the source continues. The texts and quotes
# below are copied exactly - the chunk content from the corpus, the quotes
# from the model's output.
# ---------------------------------------------------------

ARTICLE_70 = "constitution-of-kenya-2010@v2010-chapter-five-article-70"

ARTICLE_70_TEXT = (
    "(1) If a person alleges that a right to a clean and healthy environment "
    "recognised and protected under Article 42 has been, is being or is "
    "likely to be, denied, violated, infringed or threatened, the person may "
    "apply to a court for redress in addition to any other legal remedies "
    "that are available in respect to the same matter.\n(2) On application "
    "under clause (1), the court may make any order, or give any directions, "
    "it considers appropriate—\n(a) to prevent, stop or discontinue any act "
    "or omission that is harmful to the environment;\n(b) to compel any "
    "public officer to take measures to prevent or discontinue any act or "
    "omission that is harmful to the environment; or\n(c) to provide "
    "compensation for any victim of a violation of the right to a clean and "
    "healthy environment.\n(3) For the purposes of this Article, an "
    "applicant does not have to demonstrate that any person has incurred "
    "loss or suffered injury."
)

# The opening of section 5, verbatim.
SECTION_5_TEXT = (
    "(1) It shall be the duty of the Cabinet Secretary, labour officers and "
    "the Employment and Labour Relations Court—\n(a) to promote equality of "
    "opportunity in employment in order to eliminate discrimination in "
    "employment; and\n(b) to promote and guarantee equality of opportunity "
    "for a person who, is a migrant worker or a member of the family of the "
    "migrant worker, lawfully within Kenya.\n(2) An employer shall promote "
    "equal opportunity in employment and strive to eliminate discrimination "
    "in any employment policy or practice.\n(3) No employer shall "
    "discriminate directly or indirectly, against an employee or "
    "prospective employee or harass an employee or prospective employee—\n"
    "(a) on grounds of race, colour, sex, language, religion, political or "
    "other opinion, nationality, ethnic or social origin, disability, "
    "pregnancy, marital status or HIV status;\n(b) in respect of "
    "recruitment, training, promotion, terms and conditions of employment, "
    "termination of employment or other matters arising out of the "
    "employment."
)

REDRESS_QUOTE = (
    "If a person alleges that a right to a clean and healthy environment... "
    "has been, is being or is likely to be, denied, violated, infringed or "
    "threatened, the person may apply to a court for redress."
)

ORDER_QUOTE = (
    "On application under clause (1), the court may make any order, or give "
    "any directions, it considers appropriate—to prevent, stop or "
    "discontinue any act or omission that is harmful to the environment."
)


@pytest.mark.parametrize(
    "quote, source",
    [
        (REDRESS_QUOTE, ARTICLE_70_TEXT),
        (ORDER_QUOTE, ARTICLE_70_TEXT),
        (
            "No employer shall discriminate directly or indirectly, against "
            "an employee or prospective employee... on grounds of race, "
            "colour, sex, language, religion, political or other opinion, "
            "nationality, ethnic or social origin, disability, pregnancy, "
            "marital status or HIV status.",
            SECTION_5_TEXT,
        ),
        (
            "It shall be the duty of the Cabinet Secretary, labour officers "
            "and the Employment and Labour Relations Court... to promote "
            "equality of opportunity in employment in order to eliminate "
            "discrimination in employment.",
            SECTION_5_TEXT,
        ),
    ],
)
def test_a_quote_cut_short_with_a_full_stop_is_accepted(quote, source):
    assert quote_supports(quote, source)


def test_a_changed_word_still_fails_despite_the_full_stop():
    """Only the final character is forgiven - never the words before it."""

    assert not quote_supports(
        "the person must apply to a court for redress.", ARTICLE_70_TEXT
    )


def test_punctuation_inside_a_quote_must_still_match():
    assert not quote_supports(
        "the person may apply to a court; for redress in addition",
        ARTICLE_70_TEXT,
    )


def test_the_factory_pollution_answer_is_no_longer_refused():
    """The production refusal, end to end through verification."""

    parsed = LegalAnswer.from_json(
        json.dumps(
            {
                "sufficient": True,
                "question_type": "open",
                "verdict": "not_applicable",
                "statements": [
                    {
                        "text": "You can apply to a court for redress.",
                        "chunk_id": ARTICLE_70,
                        "quote": REDRESS_QUOTE,
                    },
                    {
                        "text": "The court can order the harm stopped.",
                        "chunk_id": ARTICLE_70,
                        "quote": ORDER_QUOTE,
                    },
                ],
                "qualifications": [],
                "missing": "",
            }
        )
    )

    result = verify_citations(
        parsed, [{"chunk_id": ARTICLE_70, "content": ARTICLE_70_TEXT}]
    )

    assert result.sufficient is True
    assert result.refused_by_grounding is False
    assert result.ungrounded == []


# ---------------------------------------------------------
# WHAT SURVIVES MUST STILL BE AN ANSWER
# ---------------------------------------------------------

def test_an_answer_whose_claims_mostly_failed_becomes_a_refusal():
    """
    Found in production. Asked how to leave property to one's children,
    the model made three claims, two were dropped, and the survivor was an
    incidental point about adopted children's inheritance rights - which
    was then rendered as the answer, with `sufficient` still true.

    A confident non-sequitur is harder for a reader to detect than the
    overstatement this check was built to catch, so losing most of the
    claims now refuses rather than keeping the remainder.
    """

    invented = grounded(quote="a quote that appears nowhere in the source")

    result = verify_citations(
        answer(statements=[grounded(), invented, grounded(quote="also invented entirely")]),
        sources(),
    )

    assert result.sufficient is False
    assert render(result).startswith(INSUFFICIENT_ANSWER)


def test_a_mostly_grounded_answer_survives():
    """Stripping one claim of three is not a reason to refuse."""

    result = verify_citations(
        answer(
            statements=[
                grounded(),
                grounded(quote="to remain silent; and (iii) the consequences"),
                grounded(quote="an invented quote that is not in the text"),
            ]
        ),
        sources(),
    )

    assert result.sufficient is True
    assert len(result.statements) == 2
    assert len(result.ungrounded) == 1


def test_an_answer_with_nothing_left_becomes_a_refusal():
    """
    The branch that could close `debt-inability`: if no claim can be
    grounded, the answer has nothing to stand on.
    """

    result = verify_citations(
        answer(
            statements=[grounded(quote="a quote that appears nowhere at all")]
        ),
        sources(),
    )

    assert result.sufficient is False
    assert result.verdict == "not_applicable"
    assert render(result).startswith(INSUFFICIENT_ANSWER)


def test_qualifications_are_grounded_too():
    """Exceptions and conditions are where overstatement hides."""

    result = verify_citations(
        answer(
            qualifications=[
                grounded(quote="an invented limit that is not in the text")
            ]
        ),
        sources(),
    )

    assert result.qualifications == []
    assert len(result.ungrounded) == 1


# ---------------------------------------------------------
# WHO REFUSED
#
# Verification can overwrite `sufficient`. On the first product-suite run six
# answerable questions were refused with the correct law in their sources,
# and nothing recorded whether the model declined or the grounding check
# overruled it. Those need opposite fixes, so the record must survive.
# ---------------------------------------------------------

def model_response(statements: list[dict], sufficient: bool = True) -> str:
    return json.dumps(
        {
            "sufficient": sufficient,
            "question_type": "polar",
            "verdict": "no" if sufficient else "not_applicable",
            "statements": statements,
            "qualifications": [],
            "missing": "" if sufficient else "The sources are silent.",
        }
    )


REAL = {"text": "Real.", "chunk_id": ARTICLE_49, "quote": SUPPORTED_QUOTE}
FAKE = {
    "text": "Invented.",
    "chunk_id": ARTICLE_49,
    "quote": "a quote that appears nowhere in the source",
}


def test_a_refusal_forced_by_grounding_is_recorded_as_such():
    parsed = LegalAnswer.from_json(model_response([REAL, FAKE]))

    result = verify_citations(parsed, sources())

    assert result.sufficient is False
    assert result.model_sufficient is True
    assert result.refused_by_grounding is True
    assert result.statements_dropped == 1


def test_the_models_own_refusal_is_not_blamed_on_grounding():
    parsed = LegalAnswer.from_json(model_response([], sufficient=False))

    result = verify_citations(parsed, sources())

    assert result.sufficient is False
    assert result.model_sufficient is False
    assert result.refused_by_grounding is False


def test_an_answer_that_survives_is_not_marked_refused():
    parsed = LegalAnswer.from_json(model_response([REAL, REAL, FAKE]))

    result = verify_citations(parsed, sources())

    assert result.sufficient is True
    assert result.refused_by_grounding is False
    assert result.statements_dropped == 1


def test_dropped_qualifications_are_not_counted_as_statements():
    """The refusal rule counts statements only, so the record must too."""

    result = verify_citations(
        answer(
            qualifications=[
                grounded(quote="an invented limit that is not in the text")
            ]
        ),
        sources(),
    )

    assert len(result.ungrounded) == 1
    assert result.statements_dropped == 0


def test_citations_are_derived_from_surviving_statements():
    """
    A citation cannot outlive the claim it supported, so the two can
    never disagree.
    """

    result = verify_citations(
        answer(
            statements=[
                grounded(),
                grounded(
                    chunk_id=ARTICLE_51,
                    quote="retains all the rights and fundamental freedoms",
                ),
            ]
        ),
        sources(),
    )

    assert result.cited_chunk_ids == [ARTICLE_49, ARTICLE_51]


def test_cited_sources_returns_only_what_was_cited():
    result = cited_sources(
        answer(
            statements=[
                grounded(
                    chunk_id=ARTICLE_51,
                    quote="retains all the rights and fundamental freedoms",
                )
            ]
        ),
        sources(),
    )

    assert len(result) == 1
    assert result[0]["chunk_id"] == ARTICLE_51


def test_cited_sources_preserves_citation_order():
    result = cited_sources(
        answer(
            statements=[
                grounded(
                    chunk_id=ARTICLE_51,
                    quote="retains all the rights and fundamental freedoms",
                ),
                grounded(),
            ]
        ),
        sources(),
    )

    assert [s["chunk_id"] for s in result] == [ARTICLE_51, ARTICLE_49]


def test_no_citations_yields_no_sources():
    assert cited_sources(answer(statements=[]), sources()) == []


# ---------------------------------------------------------
# PARSING
# ---------------------------------------------------------

def test_parses_a_well_formed_response():
    payload = json.dumps(
        {
            "sufficient": True,
            "question_type": "polar",
            "verdict": "no",
            "statements": [
                {
                    "text": "  Article 49 applies.  ",
                    "chunk_id": ARTICLE_49,
                    "quote": SUPPORTED_QUOTE,
                }
            ],
            "qualifications": [
                {"text": "A limit.", "chunk_id": ARTICLE_49, "quote": "x"}
            ],
            "missing": "",
        }
    )

    result = LegalAnswer.from_json(payload)

    assert result.verdict == "no"
    assert result.statements[0].text == "Article 49 applies."
    assert result.statements[0].quote == SUPPORTED_QUOTE
    assert result.qualifications[0].text == "A limit."


def test_invalid_json_is_rejected_clearly():
    with pytest.raises(RuntimeError, match="invalid JSON"):
        LegalAnswer.from_json("not json at all")


def test_missing_fields_are_reported():
    with pytest.raises(RuntimeError, match="missing required fields"):
        LegalAnswer.from_json(json.dumps({"sufficient": True}))

