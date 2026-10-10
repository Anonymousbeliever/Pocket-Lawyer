"""
Tier 2 answer-quality judging.

Every tier below this measures RETRIEVAL. `passed()` at tier 1 is
`rerank_hit` - "did the expected authority survive to the point where the
LLM would see it?" - and tier 2 adds only refusal correctness and whether
an expected chunk was cited. Before this judge, **nothing read the answer.**

WHY THE JUDGE CHECKS CLAIM BY CLAIM
-----------------------------------

The first version of this judge read the whole answer and asked one
question: "do its claims trace to the sources?" On 2026-09-27 it passed four
answers that were each wrong in a way that matters to the person asking:

    "Employees have the right to claim unpaid wages, including overtime,
     without fear of retaliation."        - neither cited section mentions
                                            overtime or retaliation
    "...they must return them..."         - the section gives that power to
                                            the court, not the police
    "...medical attention during illness" - the section says medicine, and
                                            "if possible" attendance during
                                            SERIOUS illness
    "Yes. ...insulting material about the president... can be considered
     illegal"                             - the offence turns on material
                                            likely to cause public alarm;
                                            the president is not mentioned

Every one of those claims had a real, verified quote. The failures were in
the sentence the model wrote AROUND the quote - an added subject, the wrong
actor, "must" for "may", a dropped condition. A judge reading the answer as
a whole saw real law, cited correctly, and passed it.

So the judge now takes the answer apart: each claim, one at a time, against
the full text of every section the answer cited, with a named finding for
each of the ways a claim can say more than its source. It must copy the
words that bear on the claim BEFORE deciding, so a verdict rests on text it
actually read.

Why the full sections and not only the quote: the quote is already verified
by `verify_citations`, and the condition a claim drops usually sits
elsewhere in the section - s.121's "unless the court sees fit" is in
subsection (3). The user also reads the claims beside one list of sources,
not one source per claim.

WHAT THIS JUDGE IS ASKED, AND WHAT IT IS NOT
--------------------------------------------

It compares words with words. It is NOT asked whether the law is correct,
and that restriction is the whole design. An LLM was tried for reranking
and failed exactly there: asked which provision answers a pre-trial bail
question, gpt-4o-mini returned two sections about bail *pending appeal* and
dropped Article 49. It has no reliable model of Kenyan law, and neither
does this one.

THE JUDGE IS NOT TRUSTED EITHER
-------------------------------

`python -m evaluation.judge` runs it against `judge_cases.yaml`: real
answers with the defects above, and faithful answers that must pass. Its
number means nothing until it agrees with those. The illustrations in the
prompt are invented and deliberately unlike those cases - a judge shown the
test answers as examples would be measuring its memory, not its reading.

This module is evaluation-only. It is never imported by `backend/`, so the
serving path stays free of a second model call.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from openai import OpenAI

from backend.app.core.config import (
    DOCUMENTS_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


CASES_PATH = Path(__file__).resolve().parent / "judge_cases.yaml"

# OpenAI's reasoning models - o1, o3-mini, o4-mini and the rest of the "o"
# series - reject `temperature` outright: the request fails rather than
# ignoring it. They run at their fixed default instead, so unlike a
# temperature-0 verdict, theirs can vary between runs.
_REASONING_MODEL = re.compile(r"^o\d")


def is_reasoning_model(model_name: str) -> bool:
    return bool(_REASONING_MODEL.match(model_name.strip().lower()))

# Ways a single claim can say more than the sections it rests on. "stated"
# is the only passing finding.
CLAIM_FINDINGS = [
    "stated",
    "adds_to_source",
    "wrong_actor",
    "overstates_certainty",
    "drops_condition",
]

ANSWER_DEFECTS = [
    "none",
    "non_sequitur",
    "wrong_topic",
    "incomplete",
    "unsupported_verdict",
]

CLAIM_CHECK = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "section_says": {
            "type": "string",
            "description": (
                "The words of the cited sections that bear on this claim, "
                "copied exactly. \"nothing\" if no words do."
            ),
        },
        "finding": {
            "type": "string",
            "enum": CLAIM_FINDINGS,
        },
        "reason": {
            "type": "string",
            "description": (
                "One sentence naming the exact difference between the claim "
                "and the section, or confirming there is none."
            ),
        },
    },
    "required": ["section_says", "finding", "reason"],
}

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "description": "One check per numbered claim, in order.",
            "items": CLAIM_CHECK,
        },
        "responsive": {
            "type": "boolean",
            "description": (
                "True only if the answer addresses the question that was "
                "actually asked."
            ),
        },
        "defect": {
            "type": "string",
            "enum": ANSWER_DEFECTS,
        },
        "reason": {
            "type": "string",
            "description": "One sentence about the answer as a whole.",
        },
    },
    "required": ["claims", "responsive", "defect", "reason"],
}


INSTRUCTIONS = """
You are checking an answer produced by a legal information system for
Kenya. You are a careful reader, not a lawyer. You compare words with
words.

The answer has been split into numbered CLAIMS, and you are shown the
full text of every section it cited. Bad answers rarely invent law
outright. They take a real section and say MORE than it says. Catching
that is your main job.

STEP 1 - CHECK EVERY CLAIM, ONE AT A TIME

For each claim, first copy into section_says the words of the sections
that bear on it, exactly as written. If no words do, write "nothing".
Only then choose a finding:

  stated                the sections say this. Plain wording is fine,
                        and so is applying a section to the person's own
                        facts WHEN the section's words cover those facts.
  adds_to_source        the claim includes a right, duty, protection,
                        remedy, subject or fact the sections do not
                        mention.
  wrong_actor           the sections give the power or duty to a
                        different person or body than the claim does.
  overstates_certainty  the sections say "may", "may recommend", "if
                        possible", "reasonable steps" or similar, and the
                        claim says "must", "is entitled", "will" or
                        states it as certain.
  drops_condition       the sections attach a condition, limit or
                        exception the claim leaves out, so the claim
                        covers more than the sections do.

Be strict. If you cannot copy words from the sections that state the
claim, it is not "stated".

Illustrations - invented, not Kenyan law:

  Section: "A tenant may apply to the tribunal for an order."
  Claim:   "You can apply to the tribunal, and the landlord cannot evict
            you while it decides."                 -> adds_to_source
  Section: "The Minister may approve the licence."
  Claim:   "The county can approve the licence."   -> wrong_actor
  Section: "The registrar may extend the period."
  Claim:   "The registrar must extend the period." -> overstates_certainty
  Section: "A driver over the age of seventy shall renew the licence
            every year."
  Claim:   "Drivers must renew their licence every year."
                                                    -> drops_condition
  Section: "An employer shall pay wages in the currency of Kenya."
  Claim:   "Your employer has to pay you in Kenyan shillings."
                                                    -> stated

STEP 2 - THE ANSWER AS A WHOLE

  responsive  does it address the question actually asked?
  defect      none
              non_sequitur        true, but does not answer the question
              wrong_topic         answers a different legal question
              incomplete          addresses the question but omits the
                                  central rule the sections contain
              unsupported_verdict opens with Yes, No or It depends, and
                                  the claims - as the sections actually
                                  state them - do not establish it

  If responsive is false, defect must not be "none".

  Worked example: asked "I want to leave my property to my children
  after I die. How do I legally do that?", an answer said "For adopted
  children, they will have the same inheritance rights as biological
  children." The claim IS stated by its section, but it does not answer
  what was asked: responsive = false, defect = non_sequitur.

Do NOT judge whether the law is correct, whether better law exists, or
whether the answer is good legal advice. Use no knowledge of Kenyan law.
The only question is whether the sections shown say what the claims say.

Return exactly one claim check per numbered claim, in order. Respond in
the required JSON only.
"""


class AnswerJudge:
    """Judges one answer at a time. Evaluation-only."""

    def __init__(self, model_name: str = OPENAI_MODEL):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Add it to your .env "
                "file, or run with --no-judge."
            )

        self.model_name = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def judge(
        self,
        question: str,
        answer: str,
        claims: list[str],
        sources: list[dict],
        max_source_chars: int = 8000,
    ) -> dict:
        """
        Returns the verdict, plus `error` when the judge itself failed.

        A judge failure must never read as an answer failure - that would
        turn an outage into a quality regression. `quality` is None in
        that case, and the report counts it apart.

        `max_source_chars` is deliberately generous. The first version cut
        sections at 1,200 characters, and s.49 alone is nearly 3,000: a
        condition past the cut would have been reported as missing from
        the law.
        """

        if not claims or not sources:
            return {
                "claims": [],
                "responsive": False,
                "defect": "incomplete",
                "reason": "The answer made no claim that cites a source.",
                "error": None,
            }

        numbered = "\n".join(
            f"CLAIM {index}: {text}"
            for index, text in enumerate(claims, start=1)
        )

        shown = "\n\n".join(
            f"SECTION {index}\n"
            f"{s.get('citation')} - {s.get('unit_title')}\n"
            f"{(s.get('content') or '')[:max_source_chars]}"
            for index, s in enumerate(sources, start=1)
        )

        request = {
            "model": self.model_name,
            "instructions": INSTRUCTIONS,
            "input": (
                f"QUESTION\n{question}\n\n"
                f"ANSWER GIVEN\n{answer}\n\n"
                f"CLAIMS TO CHECK\n{numbered}\n\n"
                f"SECTIONS THE ANSWER CITED\n\n{shown}"
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer_quality",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                }
            },
        }

        if not is_reasoning_model(self.model_name):
            request["temperature"] = 0

        try:
            response = self.client.responses.create(**request)

            verdict = json.loads(response.output_text.strip())

        except Exception as error:                        # noqa: BLE001
            return _judge_error(f"{type(error).__name__}: {error}")

        # A check that skipped a claim must not pass the answer: the claim
        # it skipped is exactly as likely as any other to be the bad one.
        if len(verdict.get("claims") or []) != len(claims):
            return _judge_error(
                f"judge checked {len(verdict.get('claims') or [])} of "
                f"{len(claims)} claims"
            )

        verdict["error"] = None

        return verdict


def _judge_error(message: str) -> dict:
    return {
        "claims": [],
        "responsive": None,
        "defect": None,
        "reason": "",
        "error": message,
    }


def quality(verdict: dict | None) -> bool | None:
    """
    One boolean from the verdict, or None when it could not be judged.

    Strict on purpose: an answer passes only if it is responsive, carries
    no answer-level defect, and EVERY claim is stated by its sections. One
    overstated claim fails the answer, because that one claim is what the
    person reading it may act on.
    """

    if not verdict or verdict.get("error"):
        return None

    if verdict.get("responsive") is None:
        return None

    claims_sound = all(
        check.get("finding") == "stated"
        for check in verdict.get("claims") or []
    )

    return bool(
        verdict.get("responsive")
        and verdict.get("defect") == "none"
        and claims_sound
    )


def headline(verdict: dict | None) -> tuple[str | None, str]:
    """
    The single defect and reason the report shows for a verdict.

    Answer-level problems win over claim-level ones: an answer to a
    different question is the bigger failure, whatever its claims say.
    """

    if not verdict or verdict.get("error"):
        return None, ""

    defect = verdict.get("defect")

    if verdict.get("responsive") is False or defect not in (None, "none"):
        named = defect if defect and defect != "none" else "unresponsive"
        return named, verdict.get("reason") or ""

    for number, check in enumerate(verdict.get("claims") or [], start=1):
        if check.get("finding") != "stated":
            return (
                check.get("finding"),
                f"claim {number}: {check.get('reason') or ''}",
            )

    return "none", verdict.get("reason") or ""


# ---------------------------------------------------------
# VALIDATION - does the judge agree with a careful reading?
# ---------------------------------------------------------

def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []

    if not isinstance(cases, list):
        raise ValueError(f"{path.name} must contain a list.")

    return cases


def load_sections(
    chunk_ids: set[str],
    documents_dir: Path = DOCUMENTS_DIR,
) -> dict[str, dict]:
    """The corpus chunks a set of cases cites, by id."""

    found: dict[str, dict] = {}

    for path in documents_dir.glob("*/chunks.json"):
        for chunk in json.loads(path.read_text(encoding="utf-8")):
            if chunk["chunk_id"] in chunk_ids:
                found[chunk["chunk_id"]] = chunk

    return found


def score_case(case: dict, verdict: dict) -> list[tuple[bool, str]]:
    """
    Every expectation in one case, as (agreed, description).

    Claims marked neither `flagged` nor `sound` are not asserted: where a
    careful reading is itself borderline, scoring the judge on it would
    measure a coin toss.
    """

    checks: list[tuple[bool, str]] = []

    judged = quality(verdict)
    expected = bool(case["expect_quality"])

    checks.append(
        (
            judged is expected,
            f"answer   expected {'PASS' if expected else 'FAIL'}, judge "
            f"{'ERROR' if judged is None else 'PASS' if judged else 'FAIL'}",
        )
    )

    findings = verdict.get("claims") or []

    for number, claim in enumerate(case["claims"], start=1):
        expect = claim.get("expect")

        if expect not in ("flagged", "sound"):
            continue

        finding = (
            findings[number - 1].get("finding")
            if number <= len(findings)
            else None
        )

        flagged = finding is not None and finding != "stated"

        agreed = finding is not None and flagged == (expect == "flagged")

        checks.append(
            (agreed, f"claim {number}  expected {expect}, judge {finding}")
        )

    return checks


def _print_judge_evidence(case: dict, verdict: dict) -> None:
    """
    The words the judge copied and its reason, for every claim it flagged
    and every claim where it disagreed with the case.

    The first validation run printed reasons only for claims the judge
    FLAGGED - so for the four defects it passed, there was no way to see
    what it had read. That is the difference between two opposite fixes:
    copied the right words and still said "stated" is the model's limit;
    copied the wrong words is a reading problem.
    """

    findings = verdict.get("claims") or []

    for number, claim in enumerate(case["claims"], start=1):
        if number > len(findings):
            break

        check = findings[number - 1]
        finding = check.get("finding")
        expect = claim.get("expect")

        disagreed = expect in ("flagged", "sound") and (
            (finding != "stated") != (expect == "flagged")
        )

        if finding == "stated" and not disagreed:
            continue

        marker = "DISAGREES" if disagreed else "flagged"

        print()
        print(f"       claim {number} ({marker}) - judge: {finding}")
        print(f"         claim     {claim['text']}")
        print(f"         copied    \"{check.get('section_says')}\"")
        print(f"         reason    {check.get('reason')}")

    judged = quality(verdict)

    answer_disagreed = (
        judged is not None and judged != bool(case["expect_quality"])
    )

    if answer_disagreed or verdict.get("defect") not in (None, "none"):
        print()
        print(
            f"       answer - responsive={verdict.get('responsive')} "
            f"defect={verdict.get('defect')}"
        )
        print(f"         reason    {verdict.get('reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the answer-quality judge against judge_cases.yaml and "
            "report how often it agrees with a careful reading."
        )
    )

    parser.add_argument(
        "--id",
        action="append",
        help="run only this case id (repeatable)",
    )

    parser.add_argument(
        "--model",
        default=OPENAI_MODEL,
        help=(
            f"the model to judge with (default {OPENAI_MODEL}). Changes the "
            "judge only - the answers being judged are fixed in the cases "
            "file, and serving is untouched"
        ),
    )

    args = parser.parse_args()

    cases = load_cases()

    if args.id:
        cases = [c for c in cases if c["id"] in set(args.id)]

    sections = load_sections(
        {cid for case in cases for cid in case["cites"]}
    )

    missing = {
        cid for case in cases for cid in case["cites"]
    } - set(sections)

    if missing:
        print(f"[FAIL] cited chunks not in the corpus: {sorted(missing)}")
        sys.exit(1)

    judge = AnswerJudge(model_name=args.model)

    agreed_total = 0
    checks_total = 0

    print("=" * 64)
    print(f"JUDGE VALIDATION  ({len(cases)} cases, model {args.model})")
    print("=" * 64)

    if is_reasoning_model(args.model):
        print()
        print(
            "Reasoning model: temperature not sent (it is rejected), so "
            "verdicts can vary between runs. Run it twice before trusting "
            "a single result."
        )

    for case in cases:
        verdict = judge.judge(
            question=case["question"],
            answer=case["answer"].strip(),
            claims=[claim["text"] for claim in case["claims"]],
            sources=[sections[cid] for cid in case["cites"]],
        )

        checks = score_case(case, verdict)

        agreed = sum(1 for ok, _ in checks if ok)
        agreed_total += agreed
        checks_total += len(checks)

        status = "AGREE" if agreed == len(checks) else "DISAGREE"

        print()
        print(f"[{status}] {case['id']}")

        if verdict.get("error"):
            print(f"       judge error: {verdict['error']}")

        for ok, description in checks:
            print(f"       {'ok ' if ok else 'XX '} {description}")

        _print_judge_evidence(case, verdict)

    print()
    print("=" * 64)
    print(f"Agreement: {agreed_total}/{checks_total} checks")
    print("=" * 64)

    sys.exit(0 if agreed_total == checks_total else 1)


if __name__ == "__main__":
    main()
