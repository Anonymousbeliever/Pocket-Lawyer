"""
How a stored chunk is presented to the embedding and reranking models.

The legal text alone is not always enough to find or judge a chunk. An
article's topic often lives in its title and nowhere in its body -
Article 16 is titled "Dual citizenship" and its text reads "A citizen
by birth does not lose citizenship by acquiring the citizenship of
another country", never using the word "dual". Embedding the body
alone made that article effectively unfindable.

The same header helps the reranker for the opposite reason. Schedule
chunks are long, heterogeneous lists that a cross-encoder scores well
against almost anything; labelling one "SIXTH SCHEDULE" gives it
something to judge against when the question is about torture.

This is used ONLY for embedding and reranking. The raw `content` is
what reaches the LLM, so the header never pollutes the answer context.
"""


def contextual_text(chunk: dict) -> str:
    """Prepend document, citation and unit title to a chunk's text."""

    content = chunk.get("content") or ""

    lines: list[str] = []

    document = chunk.get("title")

    if document:
        lines.append(str(document))

    heading = _heading(chunk)

    if heading:
        lines.append(heading)

    if not lines:
        return content

    return "\n".join(lines) + "\n\n" + content


def _heading(chunk: dict) -> str:
    citation = str(chunk.get("citation") or "").strip()
    unit_title = str(chunk.get("unit_title") or "").strip()

    if not unit_title:
        return citation

    if not citation:
        return unit_title

    # Schedules carry the same string as both citation and title;
    # "SIXTH SCHEDULE: SIXTH SCHEDULE" helps nobody.
    if unit_title.lower() in citation.lower():
        return citation

    return f"{citation}: {unit_title}"
