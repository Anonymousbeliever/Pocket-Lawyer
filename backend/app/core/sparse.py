"""
BGE-M3's lexical (sparse) weights, from the model already in memory.

Dense retrieval encodes *meaning*, and that is what makes it good at
paraphrase: "can cops arrest me" reaches "every arrested person has the
right to be informed". The cost is that exact terms get averaged into the
general sense of a passage, so section numbers, chapter citations and rare
legal terms lose their sharpness.

Measured here, concretely. Asked "What is the sentence for murder in
Kenya?", section 204 - *"Any person convicted of murder shall be sentenced
to death"*, eleven words answering the question almost verbatim - came back
at **rank 53**, beaten by "Conspiracy to murder", which is longer and shares
more vocabulary. Content-only cosine: 0.6992 for conspiracy against 0.6026
for the provision that actually answers it.

Sparse retrieval fails in the opposite direction: it matches exact terms and
cannot paraphrase at all. Fusing the two covers both, which is what legal
questions need - "what are my rights if arrested" is paraphrase, while
"section 295" and "malice aforethought" are anchors.

**Why this file exists rather than a dependency.** BGE-M3 produces lexical
weights natively, but `sentence_transformers` does not expose them: asking
its `SparseEncoder` for BAAI/bge-m3 silently converts the dense model into a
generic 4096-dimension projection, which is not term-based and would not do
anchor matching. FlagEmbedding exposes the real thing but would load a
second 2.2 GB copy of the model. The actual head is `Linear(1024, 1)` - one
weight row and a bias - so it runs on the token embeddings of the model that
is already loaded, for no extra memory and no new package.
"""

from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from backend.app.core.config import EMBEDDING_MODEL


SPARSE_HEAD_FILE = "sparse_linear.pt"


@dataclass(frozen=True)
class SparseVector:
    """Qdrant's sparse form: parallel indices and values."""

    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices


class LexicalEncoder:
    """
    Token-level lexical weights from a loaded BGE-M3.

    Takes the `SentenceTransformer` rather than constructing one, for the
    same reason the intent classifier takes an embedder: the model is
    already in memory and must not be loaded twice.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        model_name: str = EMBEDDING_MODEL,
        max_length: int = 512,
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        self.transformer = model[0].auto_model
        self.max_length = max_length

        hidden = self.transformer.config.hidden_size

        self.head = torch.nn.Linear(hidden, 1)
        self.head.load_state_dict(
            torch.load(
                hf_hub_download(model_name, SPARSE_HEAD_FILE),
                map_location="cpu",
                weights_only=True,
            )
        )
        self.head.eval()

        # Special tokens carry weight but no meaning. Left in, every
        # passage would share a high-weight match on CLS and SEP.
        self.special_ids = set(self.tokenizer.all_special_ids)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        hidden = self.transformer(**batch).last_hidden_state

        # ReLU keeps this genuinely sparse: a term either carries weight or
        # is absent, never negative.
        weights = torch.relu(self.head(hidden)).squeeze(-1)

        return [
            self._pool(ids.tolist(), row.tolist(), mask.tolist())
            for ids, row, mask in zip(
                batch["input_ids"], weights, batch["attention_mask"]
            )
        ]

    def _pool(
        self,
        ids: list[int],
        weights: list[float],
        mask: list[int],
    ) -> SparseVector:
        """
        One weight per distinct token, taking the maximum.

        A term repeated in a long passage should not out-score the same
        term used once in a short one - that is the length bias sparse
        retrieval is supposed to avoid.
        """

        best: dict[int, float] = {}

        for token_id, weight, attended in zip(ids, weights, mask):
            if not attended or token_id in self.special_ids or weight <= 0:
                continue

            if weight > best.get(token_id, 0.0):
                best[token_id] = weight

        items = sorted(best.items())

        return SparseVector(
            indices=[token for token, _ in items],
            values=[round(weight, 6) for _, weight in items],
        )

    def encode_one(self, text: str) -> SparseVector:
        vectors = self.encode([text])

        return vectors[0] if vectors else SparseVector([], [])


# ---------------------------------------------------------
# CLI HARNESS
#
# Every AI module here doubles as its own harness. This one shows the terms
# a passage and a query actually share, which is the whole mechanism.
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("POCKET LAWYER — LEXICAL (SPARSE) WEIGHTS")
    print("=" * 70)
    print()

    model = SentenceTransformer(EMBEDDING_MODEL)
    encoder = LexicalEncoder(model)

    print("[PASS] sparse head loaded")
    print()

    query = "What is the sentence for murder in Kenya?"

    passages = {
        "s.204 Punishment of murder": (
            "Any person convicted of murder shall be sentenced to death."
        ),
        "s.224 Conspiracy to murder": (
            "Any person who conspires with any other person to kill any "
            "person, whether that person is in Kenya or elsewhere, is "
            "guilty of a felony."
        ),
    }

    query_vector = encoder.encode_one(query)
    query_terms = dict(zip(query_vector.indices, query_vector.values))

    print(f"query: {query!r}")
    print(f"  {len(query_vector)} terms")
    print()

    for label, text in passages.items():
        vector = encoder.encode_one(text)
        terms = dict(zip(vector.indices, vector.values))

        shared = set(terms) & set(query_terms)
        score = sum(terms[t] * query_terms[t] for t in shared)

        print(f"{label}")
        print(f"  sparse dot product : {score:.4f}")
        print(
            "  shared terms       : "
            + ", ".join(
                sorted(
                    self_decoded
                    for self_decoded in (
                        encoder.tokenizer.decode([t]).strip() for t in shared
                    )
                    if self_decoded
                )
            )
        )
        print()

    print("=" * 70)
    print("Dense put s.204 at rank 53 for this query. If the dot product")
    print("above favours s.204, fusion is what rescues it.")
    print("=" * 70)


if __name__ == "__main__":
    main()
