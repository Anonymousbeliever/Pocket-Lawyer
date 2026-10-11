"""
The query rewriter.

The prompt cannot be unit tested - it needs the API. What can be, and
matters more, is everything that protects retrieval from a bad rewrite:
naming provisions is stopped in code, a failure degrades to no rewriting,
and a stored rewrite is reused rather than regenerated differently.
"""

import json
from types import SimpleNamespace

import pytest

from backend.app.ai.rewrite import INSTRUCTIONS, QueryRewriter, clean


QUESTION = "The police beat me at the station. Is that legal?"


# ---------------------------------------------------------
# NAMING A PROVISION IS STOPPED IN CODE
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "phrase",
    [
        "Article 29 freedom from torture",
        "art. 49 rights of arrested persons",
        "section 36A detention",
        "s. 22 search of a place",
        "Cap 75 procedure",
        "Part 4 search warrants",
        "rights under the Constitution",
        "Employment Act protection",
        "Penal Code assault",
    ],
)
def test_a_phrase_naming_a_provision_or_instrument_is_dropped(phrase):
    """
    A rewriter that names the wrong provision steers retrieval toward it
    with full confidence - the failure an LLM showed on bail pending
    appeal. Concepts only.
    """

    assert clean([phrase]) == []


@pytest.mark.parametrize(
    "phrase",
    [
        "any act or omission harmful to the environment",
        "constitutional right to privacy",
        "cruel, inhuman or degrading treatment",
    ],
)
def test_ordinary_legal_register_is_kept(phrase):
    """'act' and 'constitutional' are the register asked for, not names."""

    assert clean([phrase]) == [phrase]


def test_phrases_are_tidied_deduplicated_and_capped():
    assert clean(
        [
            '  "torture"  ',
            "Torture",
            "",
            "x" * 200,
            42,
            "freedom and security of the person",
            "rights of arrested persons",
            "one too many",
        ],
        count=3,
    ) == [
        "torture",
        "freedom and security of the person",
        "rights of arrested persons",
    ]


# ---------------------------------------------------------
# DEGRADE, NEVER FAIL
# ---------------------------------------------------------

def fake_client(phrases=None, fail=None):
    client = SimpleNamespace(sent=[])

    def create(**request):
        client.sent.append(dict(request))

        error = fail(request) if fail else None

        if error is not None:
            raise error

        return SimpleNamespace(
            output_text=json.dumps({"phrases": phrases or []})
        )

    client.responses = SimpleNamespace(create=create)
    return client


def rewriter(tmp_path=None, model="gpt-6-luna", **client_args):
    """Always with a fake client, so no test can reach the real API."""

    return QueryRewriter(
        model_name=model,
        count=3,
        cache_dir=tmp_path,
        client=fake_client(**client_args),
    )


def test_no_key_means_no_rewriting(monkeypatch):
    """
    No key: retrieval proceeds exactly as it does without rewriting.

    The key is removed for this test only. Without that, a machine with a
    key in .env would build a real client here and pay for an API call
    during pytest - which must stay free and keyless.
    """

    monkeypatch.setattr("backend.app.ai.rewrite.OPENAI_API_KEY", None)

    r = QueryRewriter(cache_dir=None)

    assert r.client is None
    assert r.rewrite(QUESTION) == []


def test_an_api_failure_degrades_to_nothing_and_is_not_stored(tmp_path):
    r = rewriter(tmp_path, fail=lambda request: RuntimeError("timeout"))

    assert r.rewrite(QUESTION) == []
    assert r.failures == 1
    assert "timeout" in r.last_error
    assert list(tmp_path.iterdir()) == [], "a failure must not become permanent"


def test_an_empty_question_is_not_sent():
    r = rewriter(phrases=["torture"])

    assert r.rewrite("   ") == []
    assert r.client.sent == []


def test_a_model_refusing_temperature_is_retried_without_it():
    from openai import BadRequestError

    def refuse(request):
        if "temperature" not in request:
            return None

        error = BadRequestError.__new__(BadRequestError)
        Exception.__init__(error, "Unsupported parameter: 'temperature'")
        error.param = "temperature"
        return error

    r = rewriter(phrases=["torture"], fail=refuse)

    assert r.rewrite(QUESTION) == ["torture"]
    assert [("temperature" in s) for s in r.client.sent] == [True, False]


def test_provisions_named_by_the_model_never_reach_retrieval():
    r = rewriter(phrases=["Article 29 torture", "cruel or degrading treatment"])

    assert r.rewrite(QUESTION) == ["cruel or degrading treatment"]


# ---------------------------------------------------------
# A STORED REWRITE IS REUSED, NEVER REGENERATED
# ---------------------------------------------------------

def test_a_stored_rewrite_is_reused_without_calling_the_model(tmp_path):
    first = rewriter(tmp_path, phrases=["torture"])
    assert first.rewrite(QUESTION) == ["torture"]

    # A different answer is waiting, and must never be asked for.
    second = rewriter(tmp_path, phrases=["something else entirely"])

    assert second.rewrite(QUESTION) == ["torture"]
    assert second.client.sent == []
    assert second.hits == 1


def test_the_record_can_be_read_by_a_person(tmp_path):
    r = rewriter(tmp_path, phrases=["torture"])
    r.rewrite(QUESTION)

    (path,) = tmp_path.iterdir()
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record == {
        "question": QUESTION,
        "model": "gpt-6-luna",
        "prompt": r.prompt_id,
        "phrases": ["torture"],
    }


# ---------------------------------------------------------
# NO RECALLED LAW: NUMBERS ARE DROPPED
#
# The first 165 stored rewrites included "not less than twenty-one working
# days after twelve consecutive months of service" for an annual-leave
# question - the answer, from the model's memory.
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "phrase",
    [
        "annual leave of 21 days",
        "not less than twenty-one working days of leave",
        "marriage of a person under eighteen years",
        "sick leave with half pay",
        "unlawful assembly of three or more persons",
        "fine not exceeding one thousand shillings",
    ],
)
def test_a_phrase_stating_a_number_is_dropped(phrase):
    assert clean([phrase]) == []


@pytest.mark.parametrize(
    "phrase",
    [
        "rights of everyone accused of an offence",
        "liability of someone who keeps an animal",
        "protection against being tried more than once",
        "first appearance before a court",
    ],
)
def test_words_merely_containing_a_number_are_kept(phrase):
    assert clean([phrase]) == [phrase]


def test_the_prompt_forbids_supplying_what_the_law_says():
    text = INSTRUCTIONS.format(count=3)

    assert "never what the law says about them" in text
    assert "no numbers" in text


def test_a_prompt_change_never_reuses_an_old_prompts_rewrite(tmp_path):
    """
    Stored rewrites are keyed on the exact instructions, so tightening the
    prompt cannot silently keep serving phrases the old prompt produced.
    """

    import backend.app.ai.rewrite as rewrite

    rewriter(tmp_path, phrases=["old phrase"]).rewrite(QUESTION)

    original = rewrite.INSTRUCTIONS

    try:
        rewrite.INSTRUCTIONS = original + "\nA tightened rule."

        tightened = rewriter(tmp_path, phrases=["new phrase"])

        assert tightened.rewrite(QUESTION) == ["new phrase"]

    finally:
        rewrite.INSTRUCTIONS = original


def test_a_different_model_does_not_reuse_another_models_rewrite(tmp_path):
    rewriter(tmp_path, model="gpt-6-luna", phrases=["torture"]).rewrite(QUESTION)

    other = rewriter(tmp_path, model="gpt-6.1-sol", phrases=["assault"])

    assert other.rewrite(QUESTION) == ["assault"]


def test_without_a_store_nothing_is_written(tmp_path):
    """Serving passes no store: users' questions are not written to disk."""

    r = QueryRewriter(count=3, cache_dir=None, client=fake_client(["torture"]))

    assert r.rewrite(QUESTION) == ["torture"]
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------
# THE PROMPT
# ---------------------------------------------------------

def test_the_prompt_formats_cleanly():
    text = INSTRUCTIONS.format(count=3)

    assert "at most 3 phrases" in text
    assert "{" not in text and "}" not in text


def test_the_example_is_unlike_every_test_question():
    """
    A rewriter shown a test question as its example would be measured on
    its memory, not its translation.
    """

    from evaluation.schema import load_questions

    for question in load_questions():
        assert "dog" not in question.question.lower(), question.id
        assert question.question not in INSTRUCTIONS, question.id
