"""
Lexical (sparse) weights.

The model-dependent half - whether BGE-M3 actually assigns "murder" and
"sentence" high weights - is measured by the CLI harness,
`python -m backend.app.core.sparse`. These cover the pooling, which is
where the quiet mistakes live: a special token left in gives every passage
a high-weight match on CLS, and summing instead of taking the max
reintroduces exactly the length bias sparse retrieval is meant to avoid.
"""

import pytest

from backend.app.core.sparse import LexicalEncoder, SparseVector


CLS, SEP, PAD = 0, 2, 1


def pool(ids, weights, mask=None, special=(CLS, SEP, PAD)):
    """Exercise `_pool` without constructing a model."""

    encoder = object.__new__(LexicalEncoder)
    encoder.special_ids = set(special)

    return encoder._pool(ids, weights, mask or [1] * len(ids))


# ---------------------------------------------------------
# POOLING
# ---------------------------------------------------------

def test_one_weight_per_distinct_token():
    vector = pool([10, 11, 12], [0.3, 0.2, 0.1])

    assert vector.indices == [10, 11, 12]
    assert vector.values == [0.3, 0.2, 0.1]


def test_repeats_take_the_maximum_not_the_sum():
    """
    Summing would let a long passage that repeats a term out-score a short
    one that answers the question - the length bias this is meant to fix.
    """

    vector = pool([10, 10, 10], [0.1, 0.9, 0.4])

    assert vector.indices == [10]
    assert vector.values == [0.9]


def test_special_tokens_are_excluded():
    """
    CLS and SEP carry weight but no meaning. Left in, every passage shares
    a high-weight match with every query.
    """

    vector = pool([CLS, 10, SEP], [0.99, 0.3, 0.99])

    assert vector.indices == [10]


def test_padding_is_excluded():
    vector = pool([10, PAD, PAD], [0.4, 0.5, 0.5], mask=[1, 0, 0])

    assert vector.indices == [10]


def test_zero_weight_terms_are_dropped():
    """ReLU output: a term either carries weight or is absent."""

    vector = pool([10, 11], [0.0, 0.4])

    assert vector.indices == [11]


def test_indices_are_sorted():
    """Qdrant expects sparse indices in ascending order."""

    vector = pool([99, 10, 50], [0.1, 0.2, 0.3])

    assert vector.indices == sorted(vector.indices)
    assert vector.values == [0.2, 0.3, 0.1]


def test_a_passage_of_only_special_tokens_is_empty():
    """
    Degrades rather than producing a meaningless vector - the retriever
    falls back to dense when this happens.
    """

    vector = pool([CLS, SEP], [0.9, 0.9])

    assert vector.is_empty
    assert len(vector) == 0


def test_indices_and_values_stay_parallel():
    vector = pool([30, 10, 20, 10], [0.1, 0.5, 0.2, 0.9])

    assert len(vector.indices) == len(vector.values)
    assert dict(zip(vector.indices, vector.values)) == {
        10: 0.9,
        20: 0.2,
        30: 0.1,
    }


# ---------------------------------------------------------
# THE VECTOR TYPE
# ---------------------------------------------------------

def test_empty_vector_reports_itself_empty():
    assert SparseVector([], []).is_empty
    assert not SparseVector([1], [0.5]).is_empty


def test_length_is_the_term_count():
    assert len(SparseVector([1, 2, 3], [0.1, 0.2, 0.3])) == 3


# ---------------------------------------------------------
# THE MECHANISM IT EXISTS FOR
# ---------------------------------------------------------

def test_shared_terms_produce_a_score_where_dense_failed():
    """
    The measured failure: "What is the sentence for murder in Kenya?" put
    section 204 at rank 53, behind "Conspiracy to murder", which is longer
    and shares more vocabulary.

    Using the real weights from the harness - murder and sentence overlap
    strongly between query and s.204, while the conspiracy passage shares
    only murder.
    """

    murder, sentence, death, kenya, conspire = 10, 11, 12, 13, 14

    query = pool([kenya, murder, sentence], [0.309, 0.277, 0.227])

    s204 = pool([murder, death, sentence], [0.329, 0.297, 0.267])
    s224 = pool([murder, conspire], [0.300, 0.280])

    def dot(a: SparseVector, b: SparseVector) -> float:
        left = dict(zip(a.indices, a.values))
        right = dict(zip(b.indices, b.values))

        return sum(left[t] * right[t] for t in set(left) & set(right))

    assert dot(query, s204) > dot(query, s224)


@pytest.mark.parametrize("ids,weights", [([], []), ([CLS], [0.5])])
def test_degenerate_input_does_not_raise(ids, weights):
    assert pool(ids, weights).is_empty
