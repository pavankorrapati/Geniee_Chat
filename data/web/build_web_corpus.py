"""
GENIEE WEB CORPUS BUILDER
=========================

Combines cleaned Python documentation into a separate corpus for
continued pretraining.

Input:
    data/web/cleaned/*.txt

Output:
    data/raw/corpus_web.txt

IMPORTANT:
- Existing corpus files are NOT overwritten.
- Existing checkpoints are NOT modified.
"""

from __future__ import annotations

from pathlib import Path
import re


# ----------------------------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEB_DIR = PROJECT_ROOT / "data" / "web"
CLEANED_DIR = WEB_DIR / "cleaned"

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DIR / "corpus_web.txt"


# ----------------------------------------------------------------------
# TEXT CLEANING
# ----------------------------------------------------------------------

def normalize_document(text: str) -> str:
    """Final corpus-level normalization."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize blank lines.
    text = re.sub(
        r"\n[ \t]*\n[ \t]*\n+",
        "\n\n",
        text,
    )

    return text.strip()


def create_document_block(
    source_name: str,
    text: str,
) -> str:
    """
    Wrap each document in a clear boundary.

    The source markers help the model understand where one
    documentation page starts and ends.
    """

    return (
        "\n"
        + "=" * 72
        + "\n"
        + f"SOURCE: Python 3.14 Documentation - {source_name}\n"
        + "=" * 72
        + "\n\n"
        + text
        + "\n\n"
    )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("GENIEE PYTHON 3.14 WEB CORPUS BUILDER")
    print("=" * 72)

    print(f"Cleaned input : {CLEANED_DIR}")
    print(f"Corpus output : {OUTPUT_FILE}")
    print()

    if not CLEANED_DIR.exists():
        print("ERROR:")
        print("Cleaned directory does not exist.")
        print()
        print("Run:")
        print("  python data\\web\\download_web_data.py")
        print("  python data\\web\\clean_web_data.py")
        return

    cleaned_files = sorted(CLEANED_DIR.glob("*.txt"))

    if not cleaned_files:
        print("ERROR: No cleaned text files found.")
        print("Run clean_web_data.py first.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    documents: list[str] = []

    total_chars = 0

    print("Reading documents...")

    for index, text_file in enumerate(cleaned_files, start=1):

        try:
            text = text_file.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            text = normalize_document(text)

            if not text:
                print(
                    f"  [{index}/{len(cleaned_files)}] "
                    f"{text_file.name}: EMPTY - skipped"
                )
                continue

            block = create_document_block(
                text_file.name,
                text,
            )

            documents.append(block)

            total_chars += len(text)

            print(
                f"  [{index}/{len(cleaned_files)}] "
                f"{text_file.name}: "
                f"{len(text):,} chars"
            )

        except Exception as exc:
            print(
                f"  [{index}/{len(cleaned_files)}] "
                f"{text_file.name}: FAILED - {exc}"
            )

    if not documents:
        print()
        print("ERROR: No usable documents were found.")
        return

    corpus = (
        "# GENIEE PYTHON 3.14 WEB CORPUS\n\n"
        "This corpus contains cleaned text extracted from "
        "the official Python 3.14 documentation.\n\n"
        + "".join(documents)
    )

    # Safety check: this script should only write corpus_web.txt.
    if OUTPUT_FILE.name != "corpus_web.txt":
        raise RuntimeError(
            "Safety check failed: unexpected output filename."
        )

    OUTPUT_FILE.write_text(
        corpus,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("CORPUS BUILD COMPLETE")
    print("=" * 72)

    print(f"Documents       : {len(documents)}")
    print(f"Source chars    : {total_chars:,}")
    print(f"Corpus chars    : {len(corpus):,}")
    print(f"Output file     : {OUTPUT_FILE}")

    print()
    print("IMPORTANT:")
    print("  Existing corpus files were NOT modified.")
    print("  Existing checkpoints were NOT modified.")
    print()
    print("Next step:")
    print("  Use corpus_web.txt for a separate continued-pretraining run.")
    print()
    print("Recommended checkpoint:")
    print("  checkpoints\\geniee_pretrain_web_best.pt")


if __name__ == "__main__":
    main()