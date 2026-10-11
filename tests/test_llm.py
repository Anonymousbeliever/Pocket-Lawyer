"""
The answering model's API call.

Only the call is tested here - the prompt needs the API. What matters is
the request that leaves: GPT-6 refuses `temperature`, and on this path a
refused request is not a failed metric but a failed answer for every user.
"""

import json
from types import SimpleNamespace

import pytest

from backend.app.ai.llm import LegalLLM, is_reasoning_model, rejects_temperature


# A valid refusal: no claims, so verification passes without any sources.
REFUSAL = json.dumps(
    {
        "sufficient": False,
        "question_type": "open",
        "verdict": "not_applicable",
        "statements": [],
        "qualifications": [],
        "missing": "The sources are silent.",
    }
)


def fake_llm(model_name: str = "gpt-4o-mini", fail=None) -> LegalLLM:
    """
    A LegalLLM whose API call returns REFUSAL, with no key needed. Every
    request it would have sent is kept on `llm.sent`; `fail(request)` may
    return an exception for the call to raise instead.
    """

    llm = object.__new__(LegalLLM)
    llm.model_name = model_name
    llm.send_temperature = not is_reasoning_model(model_name)
    llm.sent = []

    def create(**request):
        llm.sent.append(dict(request))

        error = fail(request) if fail else None

        if error is not None:
            raise error

        return SimpleNamespace(output_text=REFUSAL)

    llm.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return llm


def bad_request(param: str | None) -> Exception:
    """A real BadRequestError carrying the API's `param` field."""

    from openai import BadRequestError

    error = BadRequestError.__new__(BadRequestError)
    Exception.__init__(error, f"Unsupported parameter: '{param}'")
    error.param = param
    error.status_code = 400
    return error


def refuses_temperature(request: dict) -> Exception | None:
    """Fails exactly as GPT-6 did, and only while temperature is sent."""

    return bad_request("temperature") if "temperature" in request else None


def ask(llm: LegalLLM):
    return llm.generate(question="Can I appeal?", sources=[])


# ---------------------------------------------------------
# UNCHANGED FOR MODELS THAT ACCEPT TEMPERATURE
# ---------------------------------------------------------

def test_gpt_4o_mini_is_still_answered_at_temperature_zero():
    """
    Zero narrowed how often the model misquoted the statute between calls.
    The fix for GPT-6 must not take that away from models that accept it.
    """

    llm = fake_llm("gpt-4o-mini")

    ask(llm)

    assert len(llm.sent) == 1
    assert llm.sent[0]["temperature"] == 0


def test_reasoning_models_are_never_sent_temperature():
    llm = fake_llm("o3-mini")

    ask(llm)

    assert len(llm.sent) == 1
    assert "temperature" not in llm.sent[0]


# ---------------------------------------------------------
# GPT-6
# ---------------------------------------------------------

def test_a_model_that_refuses_temperature_still_answers():
    """
    GPT-6 Luna failed every call with "Unsupported parameter:
    'temperature'". Before this, setting OPENAI_MODEL to it would have
    failed every question a user asked.
    """

    llm = fake_llm("gpt-6-luna", fail=refuses_temperature)

    answer = ask(llm)

    assert answer.sufficient is False
    assert [("temperature" in r) for r in llm.sent] == [True, False]
    assert llm.send_temperature is False


def test_the_refusal_is_paid_once_not_per_question():
    """One LegalLLM serves every request, so it learns once per process."""

    llm = fake_llm("gpt-6-luna", fail=refuses_temperature)

    for _ in range(3):
        ask(llm)

    assert len(llm.sent) == 4


def test_the_retry_sends_everything_else_unchanged():
    llm = fake_llm("gpt-6-luna", fail=refuses_temperature)

    ask(llm)

    first, retry = llm.sent

    assert {k: v for k, v in first.items() if k != "temperature"} == retry


# ---------------------------------------------------------
# EVERY OTHER ERROR STILL SURFACES
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "error",
    [bad_request("text.format"), bad_request(None), RuntimeError("timeout")],
)
def test_any_other_error_propagates_without_a_retry(error):
    """A retry must not hide a real failure behind a second attempt."""

    llm = fake_llm("gpt-6-luna", fail=lambda request: error)

    with pytest.raises(type(error)):
        ask(llm)

    assert len(llm.sent) == 1
    assert llm.send_temperature, "a different error must not turn it off"


def test_only_the_temperature_refusal_is_recognised():
    assert rejects_temperature(bad_request("temperature"))
    assert not rejects_temperature(bad_request("max_output_tokens"))
    assert not rejects_temperature(RuntimeError("temperature"))
