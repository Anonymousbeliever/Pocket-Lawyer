import hashlib
import json
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data/raw/constitution/Constitution of Kenya.pdf"
EXTRACTED_PATH = PROJECT_ROOT / "data/extracted/constitution.txt"
CLEANED_PATH = PROJECT_ROOT / "data/cleaned/constitution_cleaned.txt"

OUTPUT_PATH = PROJECT_ROOT / "data/metadata/constitution.json"


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def build_metadata() -> dict:
    return {
        "document_id": "constitution-of-kenya-2010",
        "title": "Constitution of Kenya, 2010",
        "document_type": "constitution",
        "jurisdiction": "Kenya",
        "language": "English",

        "source": {
            "name": "Kenya Law",
            "url": None
        },

        "dates": {
            "publication_date": None,
            "effective_date": "2010-08-27"
        },

        "version": "2010",

        "files": {
            "raw": str(RAW_PATH.relative_to(PROJECT_ROOT)),
            "extracted": str(EXTRACTED_PATH.relative_to(PROJECT_ROOT)),
            "cleaned": str(CLEANED_PATH.relative_to(PROJECT_ROOT))
        },

        "checksums": {
            "raw_sha256": calculate_sha256(RAW_PATH),
            "extracted_sha256": calculate_sha256(EXTRACTED_PATH),
            "cleaned_sha256": calculate_sha256(CLEANED_PATH)
        },

        "processing": {
            "status": "validated",
            "validator": "constitution_validator",
            "validated_at": date.today().isoformat()
        }
    }


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw document not found: {RAW_PATH}"
        )

    if not EXTRACTED_PATH.exists():
        raise FileNotFoundError(
            f"Extracted document not found: {EXTRACTED_PATH}"
        )

    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned document not found: {CLEANED_PATH}"
        )

    metadata = build_metadata()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"Metadata successfully written to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()