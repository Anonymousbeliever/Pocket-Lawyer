import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


INPUT_PATH = Path(
    "data/processed/constitution_chunks.json"
)

OUTPUT_PATH = Path(
    "data/processed/constitution_embeddings.json"
)

MODEL_NAME = "BAAI/bge-m3"


def main() -> None:

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input file not found: {INPUT_PATH}"
        )

    print("=" * 60)
    print("POCKET LAWYER — CONSTITUTION EMBEDDINGS")
    print("=" * 60)
    print()

    print(f"Model: {MODEL_NAME}")
    print("Loading embedding model...")
    print()

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded.")
    print()

    chunks = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(f"Chunks to embed: {len(chunks)}")
    print()

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=4,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print()
    print("Embeddings generated.")
    print()

    output = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        item = {
            **chunk,
            "embedding": embedding.tolist(),
            "embedding_model": MODEL_NAME,
        }

        output.append(item)

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dimensions = len(
        output[0]["embedding"]
    )

    print("=" * 60)
    print("EMBEDDING VALIDATION")
    print("=" * 60)
    print()

    print(
        f"Embeddings generated: {len(output)}"
    )

    print(
        f"Vector dimensions: {dimensions}"
    )

    print(
        f"Embedding model: {MODEL_NAME}"
    )

    print()

    if len(output) != len(chunks):
        raise SystemExit(
            "Embedding count does not match chunk count."
        )

    if any(
        len(item["embedding"]) != dimensions
        for item in output
    ):
        raise SystemExit(
            "Inconsistent embedding dimensions detected."
        )

    print("[PASS] Every chunk has an embedding")
    print("[PASS] Embedding count matches chunk count")
    print("[PASS] Vector dimensions are consistent")
    print()
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Embedding generation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()