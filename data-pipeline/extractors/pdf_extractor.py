from pathlib import Path

import pymupdf


def extract_text(pdf_path: str, output_path: str) -> None:
    pdf = Path(pdf_path)
    output = Path(output_path)

    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    document = pymupdf.open(pdf)
    page_count = len(document)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append(
            f"\n--- PAGE {page_number} ---\n\n{text}"
        )

    document.close()

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        "\n".join(pages),
        encoding="utf-8",
    )

    print(f"Extracted {page_count} pages → {output}")


if __name__ == "__main__":
    extract_text(
        "data/raw/constitution/Constitution of Kenya.pdf",
        "data/extracted/constitution.txt",
    )