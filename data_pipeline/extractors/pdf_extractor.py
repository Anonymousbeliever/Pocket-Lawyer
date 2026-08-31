"""PDF text extraction. Document-type agnostic."""

from pathlib import Path

import pymupdf


def extract_text(pdf_path: str | Path, output_path: str | Path) -> int:
    """Extract every page of a PDF to a text file. Returns page count."""

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

    output.write_text("\n".join(pages), encoding="utf-8")

    print(f"       {page_count} pages -> {output.name}")

    return page_count
