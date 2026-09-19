"""
What kind of input is this?

Until the serving layer existed there was no answer to that question anywhere
in the system. Every input - a greeting included - was embedded and searched
against Kenyan law, which is how "Who are you" came back as "I don't have
enough reliable information in my current legal sources."

The first version matched regexes against the raw string. It failed the way
hand-enumerated language always fails: "what is your name" slipped past a
pattern that covered "what's your name", and "morning" slipped past one that
required "good morning". Each miss cost a retrieval, a thirty-passage CPU
rerank and a paid LLM call to produce an unhelpful refusal.

So routing is semantic now. `hi`, `morning`, `gm`, `sup`, `habari` and `niaje`
mean one thing and share almost no characters - but BGE-M3 puts them in the
same neighbourhood, and the query vector is something the request already has
to compute.

**Nearest centroid, never a threshold.** This project has a recorded finding
that absolute similarity scores are not comparable across queries: a typo'd
legal question scored 0.0086 while a question the corpus could not answer
scored 0.0373. So the question asked here is not "is this similar enough to a
greeting" but "is this more like a greeting or more like a legal question" -
a relative comparison between classes, which is the property those scores
lacked.

**The failure directions are not equal.** A greeting sent to the pipeline costs
seconds and cents. A legal question sent to a canned reply is the product
failing at its job. Everything below resolves ties, near-ties and errors toward
LEGAL.
"""

import re
from collections.abc import Callable, Sequence
from enum import StrEnum

import numpy as np


class Intent(StrEnum):
    GREETING = "greeting"
    IDENTITY = "identity"
    GRATITUDE = "gratitude"
    LEGAL = "legal"


CONVERSATIONAL = (Intent.GREETING, Intent.IDENTITY, Intent.GRATITUDE)


# ---------------------------------------------------------
# EXACT-MATCH FAST PATH
#
# These are NOT the classifier any more. They are a shortcut for a handful of
# inputs so common and so unambiguous that spending an embedding on them is
# waste. The list is deliberately short and does NOT need to be complete -
# anything it misses falls through to the centroids, which is the whole point.
# Resist growing it.
# ---------------------------------------------------------

EXACT = {
    "hi": Intent.GREETING,
    "hey": Intent.GREETING,
    "hello": Intent.GREETING,
    "morning": Intent.GREETING,
    "good morning": Intent.GREETING,
    "good afternoon": Intent.GREETING,
    "good evening": Intent.GREETING,
    "who are you": Intent.IDENTITY,
    "what are you": Intent.IDENTITY,
    "what is your name": Intent.IDENTITY,
    "what can you do": Intent.IDENTITY,
    "thanks": Intent.GRATITUDE,
    "thank you": Intent.GRATITUDE,
}

# One pattern, for the shapes where a suffix is the only variation.
EXACT_PATTERN = re.compile(
    r"^(hello+|hey+|hi+|thanks? (a lot|so much|very much))$"
)


# ---------------------------------------------------------
# EXEMPLARS
#
# Hardcoded rather than read from evaluation/questions.yaml. That file has 71
# real legal questions and would make a tempting LEGAL exemplar set, but it
# would invert the dependency - backend/ importing from evaluation/ - and it
# would move the LEGAL centroid every time the question set grew. Routing must
# not change because someone added Employment Act questions.
#
# The legal exemplars deliberately span both indexed documents and several
# question shapes: rights, procedure, definitions, yes/no, and "how do I".
# ---------------------------------------------------------

EXEMPLARS: dict[Intent, list[str]] = {
    Intent.GREETING: [
        "hello",
        "hi there",
        "hey",
        "good morning",
        "good afternoon",
        "morning",
        "evening",
        "gm",
        "sup",
        "yo",
        "habari",
        "habari yako",
        "niaje",
        "sasa",
        "mambo",
        "vipi",
        "how are you",
        "how is it going",
    ],
    Intent.IDENTITY: [
        "who are you",
        "what are you",
        "what is your name",
        "what should I call you",
        "what do you do",
        "what can you help me with",
        "what are you for",
        "how do you work",
        "tell me about yourself",
        "introduce yourself",
        "are you a lawyer",
        "are you a real advocate",
        "are you a robot",
        "are you chatgpt",
        "who made you",
        "what is this app",
        "what is this service",
        "can you explain what you do",
    ],
    Intent.GRATITUDE: [
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you so much",
        "much appreciated",
        "asante",
        "asante sana",
        "that helps",
        "that was helpful",
        "great, thanks",
        "perfect thanks",
        "ok thanks",
        "cheers",
        "appreciate it",
    ],
    Intent.LEGAL: [
        "what are my rights if I am arrested",
        "can police arrest me without telling me why",
        "how long can the police hold me before taking me to court",
        "can I be released on bail before my trial",
        "do I have to answer police questions after being arrested",
        "when can the police arrest someone without a warrant",
        "can an ordinary person arrest someone who committed a crime",
        "can the police search me when they arrest me",
        "what must a charge sheet contain",
        "can I appeal if a magistrate convicts me",
        "can I be tried twice for the same offence",
        "what happens if the prosecution has no evidence against me",
        "can a victim tell the court how the crime affected them",
        "what rights do children have in Kenya",
        "what does the constitution say about torture",
        "am I allowed to protest peacefully",
        "can the government take my land",
        "what is freedom of expression",
        "can I hold dual citizenship",
        "how is the court system structured",
        "what are county governments responsible for",
        "how do I register to vote",
        "what is the meaning of consent",
        "who can arrest me",
        "what is bail",
        "explain article 49",
        "section 29 of the criminal procedure code",
        "is it legal to record a police officer",
    ],
}


# How far the winning conversational centroid must beat LEGAL before the
# pipeline is bypassed.
#
# MEASURED, not chosen. Run `python -m backend.app.ai.intent` to reproduce.
# Over 39 labelled inputs the two classes separate cleanly, with no overlap:
#
#     conversational margins   +0.094 .. +0.361
#     legal margins            -0.222 .. -0.006
#
# So anything in (-0.006, 0.094] separates them perfectly. 0.05 sits between
# the two clusters: 0.056 clear of the worst legal case, 0.044 clear of the
# weakest conversational one.
#
# An earlier 0.10 was above the conversational floor and pushed "who built
# this" (+0.094) into the pipeline - costly, not dangerous, which is the
# point of erring this way.
#
# Raise it if an unseen legal question ever scores positive; that is the
# failure that matters. Do NOT lower it to rescue a greeting: inside the
# margin the answer is LEGAL, because a misrouted greeting wastes a few
# seconds and cents while a misrouted legal question tells someone asking
# about their arrest "Hello! Ask me about Kenyan law."
CONVERSATIONAL_MARGIN = 0.05


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation."""

    cleaned = " ".join(text.lower().split())

    return cleaned.strip(" .!?,;:")


def exact_match(text: str) -> Intent | None:
    """The fast path. None means 'ask the centroids', never 'it is legal'."""

    cleaned = normalise(text)

    if not cleaned:
        return None

    if cleaned in EXACT:
        return EXACT[cleaned]

    if EXACT_PATTERN.match(cleaned):
        return (
            Intent.GRATITUDE
            if cleaned.startswith("thank")
            else Intent.GREETING
        )

    return None


class IntentClassifier:
    """
    Nearest-centroid routing over the query embedding.

    The embedder is injected rather than constructed here, for two reasons:
    the model is already loaded by the retriever and must not be loaded twice,
    and every test in this project runs without models - passing a stub keeps
    that true.
    """

    def __init__(
        self,
        embed: Callable[[str], Sequence[float]],
        exemplars: dict[Intent, list[str]] | None = None,
        margin: float = CONVERSATIONAL_MARGIN,
    ):
        self.embed = embed
        self.margin = margin
        self.centroids = self._build(exemplars or EXEMPLARS)

    def _build(self, exemplars: dict[Intent, list[str]]) -> dict[Intent, np.ndarray]:
        centroids: dict[Intent, np.ndarray] = {}

        for intent, phrases in exemplars.items():
            vectors = np.array(
                [self.embed(phrase) for phrase in phrases],
                dtype=np.float32,
            )

            centroids[intent] = _unit(vectors.mean(axis=0))

        return centroids

    def similarities(self, vector: Sequence[float]) -> dict[Intent, float]:
        """Cosine similarity to each centroid. Both sides are unit vectors."""

        query = _unit(np.asarray(vector, dtype=np.float32))

        return {
            intent: float(np.dot(query, centroid))
            for intent, centroid in self.centroids.items()
        }

    def classify(
        self,
        text: str,
        vector: Sequence[float] | None = None,
    ) -> Intent:
        """
        Route one input.

        The exact-match path runs first and costs nothing, so "hi" never pays
        for an embedding. Only what it does not recognise is embedded, and
        that happens HERE rather than in the caller - the classifier owns its
        embedder, so the route does not need to know that classification
        involves vectors at all.

        `vector` is accepted for callers that already have one, and for tests
        that want to control the geometry exactly.

        An embedding failure resolves to LEGAL. Routing is an optimisation
        over the pipeline; losing it must cost money, never correctness.
        """

        hit = exact_match(text)

        if hit is not None:
            return hit

        if vector is None:
            try:
                vector = self.embed(text)

            except Exception:
                return Intent.LEGAL

        scores = self.similarities(vector)

        legal = scores[Intent.LEGAL]

        best = max(CONVERSATIONAL, key=lambda intent: scores[intent])

        if scores[best] - legal < self.margin:
            return Intent.LEGAL

        return best


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


# ---------------------------------------------------------
# CANNED REPLIES
#
# These should sound like a person, not a terms-of-service page. The heavy
# disclaimer already lives where it earns its place: INSUFFICIENT_ANSWER tells
# the user to consult a qualified advocate at the moment the system genuinely
# cannot help them. Repeating it on "hello" adds nothing and makes the
# assistant feel defensive.
# ---------------------------------------------------------

REPLIES = {
    Intent.GREETING: (
        "Hello. Ask me anything about Kenyan law and I'll answer from the "
        "actual legal text - for example, what your rights are if you're "
        "arrested, or when the police can arrest someone without a warrant."
    ),
    Intent.IDENTITY: (
        "I'm Pocket Lawyer, an AI assistant for Kenyan law. I answer from the "
        "actual text of Kenyan legal sources and always show you exactly which "
        "provision an answer came from, so you can check it yourself. Right "
        "now I have the Constitution of Kenya, 2010 and the Criminal Procedure "
        "Code (Cap. 75). If the law I hold doesn't cover your question, I'll "
        "say so rather than guess. I'm not a substitute for an advocate on "
        "your specific situation."
    ),
    Intent.GRATITUDE: (
        "You're welcome. Ask me anything else about Kenyan law."
    ),
}


def reply_for(intent: Intent) -> str | None:
    """The canned reply for a conversational intent, or None for LEGAL."""

    return REPLIES.get(intent)


# ---------------------------------------------------------
# CLI HARNESS
#
# The margin cannot be set from unit tests: those use stub vectors and prove
# only that the arithmetic works. Whether BGE-M3 actually places "morning"
# near "hello" and "who can arrest me" near the legal centroid is a question
# about the real model, and this is where it gets answered.
# ---------------------------------------------------------

LABELLED: list[tuple[str, Intent]] = [
    # Conversational - the ones regexes missed, plus the shapes they never had
    ("morning", Intent.GREETING),
    ("gm", Intent.GREETING),
    ("sup", Intent.GREETING),
    ("hey there", Intent.GREETING),
    ("habari yako", Intent.GREETING),
    ("niaje", Intent.GREETING),
    ("how are you", Intent.GREETING),
    ("good day", Intent.GREETING),
    ("what is your name", Intent.IDENTITY),
    ("whats your name", Intent.IDENTITY),
    ("what should I call you", Intent.IDENTITY),
    ("are you an actual lawyer", Intent.IDENTITY),
    ("what is this thing", Intent.IDENTITY),
    ("so what do you actually do", Intent.IDENTITY),
    ("who built this", Intent.IDENTITY),
    ("asante sana", Intent.GRATITUDE),
    ("that was helpful", Intent.GRATITUDE),
    ("ok thanks", Intent.GRATITUDE),
    ("appreciate it", Intent.GRATITUDE),
    # Legal - including the ones most likely to be mistaken for chat
    ("what is the meaning of consent", Intent.LEGAL),
    ("who can arrest me", Intent.LEGAL),
    ("what is bail", Intent.LEGAL),
    ("what are my rights", Intent.LEGAL),
    ("what does the law say", Intent.LEGAL),
    ("explain article 49", Intent.LEGAL),
    ("tell me about my rights on arrest", Intent.LEGAL),
    ("hello, can police arrest me without a warrant", Intent.LEGAL),
    ("hi, how long can I be held before court", Intent.LEGAL),
    ("thanks - and what happens at a bail hearing", Intent.LEGAL),
    ("what is a charge sheet", Intent.LEGAL),
    ("how do I appeal a conviction", Intent.LEGAL),
    ("can I be tried twice for the same offence", Intent.LEGAL),
    ("what happens if I cannot pay my debts", Intent.LEGAL),
    ("tell me the criminal code on robbery", Intent.LEGAL),
    ("is it legal to record a police officer", Intent.LEGAL),
    ("section 29", Intent.LEGAL),
    ("what rights do children have", Intent.LEGAL),
    ("can the police search my house", Intent.LEGAL),
    ("do I need a lawyer", Intent.LEGAL),
]


def main():
    from backend.app.ai.retriever import LegalRetriever

    print("=" * 78)
    print("POCKET LAWYER — INTENT ROUTING")
    print("=" * 78)
    print()

    retriever = LegalRetriever()

    print()
    print("Building centroids...")

    classifier = IntentClassifier(embed=retriever.embed_query)

    print(f"[PASS] {len(classifier.centroids)} centroids")
    print(f"       margin = {classifier.margin}")
    print()

    header = f"{'input':<48} {'expected':<10} {'got':<10} {'margin':>7}"
    print(header)
    print("-" * len(header))

    failures: list[tuple[str, Intent, Intent, float]] = []
    margins: list[float] = []
    routed = 0

    for text, expected in LABELLED:
        vector = retriever.embed_query(text)

        scores = classifier.similarities(vector)
        got = classifier.classify(text, vector)

        # The decision that costs something is chat-or-law, not which
        # canned reply gets picked.
        if (expected is Intent.LEGAL) == (got is Intent.LEGAL):
            routed += 1

        best = max(CONVERSATIONAL, key=lambda i: scores[i])
        margin = scores[best] - scores[Intent.LEGAL]

        margins.append(margin)

        flag = "" if got is expected else "   <-- WRONG"

        print(f"{text[:47]:<48} {expected:<10} {got:<10} {margin:>7.3f}{flag}")

        if got is not expected:
            failures.append((text, expected, got, margin))

    print()
    print("=" * 78)

    passed = len(LABELLED) - len(failures)

    # Three outcomes, not two, and conflating them overstates the damage.
    # "how are you" landing on IDENTITY instead of GREETING is not a routing
    # error at all: the decision that costs something is chat-or-law, and
    # that one was right. It just picks a different canned reply.
    dangerous = [f for f in failures if f[1] is Intent.LEGAL]
    costly = [
        f for f in failures
        if f[1] is not Intent.LEGAL and f[2] is Intent.LEGAL
    ]
    cosmetic = [
        f for f in failures
        if f[1] is not Intent.LEGAL and f[2] is not Intent.LEGAL
    ]

    print(f"{passed}/{len(LABELLED)} exact intent")
    print(f"{routed}/{len(LABELLED)} routed to the right side of the branch")
    print()
    print(f"  legal misrouted to chat     : {len(dangerous)}   <- the one that matters")
    print(f"  chat misrouted to the law   : {len(costly)}   <- wasteful, safe")
    print(f"  wrong reply, right branch   : {len(cosmetic)}   <- cosmetic")

    if dangerous:
        print()
        print("BLOCKER - these would be answered with a canned reply:")
        for text, _, got, margin in dangerous:
            print(f"  {text!r} -> {got} (margin {margin:+.3f})")

    if cosmetic:
        print()
        print("Cosmetic - bypassed the pipeline correctly, different reply:")
        for text, expected, got, _ in cosmetic:
            print(f"  {text!r} {expected} -> {got}")

    print()
    print("Margin spread, for setting CONVERSATIONAL_MARGIN:")

    conversational = [
        m for m, (_, e) in zip(margins, LABELLED) if e is not Intent.LEGAL
    ]
    legal = [m for m, (_, e) in zip(margins, LABELLED) if e is Intent.LEGAL]

    if conversational:
        print(
            f"  conversational: min {min(conversational):+.3f}  "
            f"max {max(conversational):+.3f}"
        )

    if legal:
        print(f"  legal         : min {min(legal):+.3f}  max {max(legal):+.3f}")

    print()
    print("  Set the margin above the legal max and below the")
    print("  conversational min. If those overlap, the classes are not")
    print("  separable on this exemplar set - add exemplars, do not shave")
    print("  the margin.")
    print("=" * 78)


if __name__ == "__main__":
    main()
