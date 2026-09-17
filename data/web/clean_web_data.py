"""
GENIEE WEB DATA CLEANER
=======================

Reads downloaded Python documentation HTML files and converts them
into clean text suitable for corpus construction.

Input:
    data/web/downloaded/*.html

Output:
    data/web/cleaned/*.txt

IMPORTANT:
- Does NOT modify any checkpoint.
- Does NOT modify the existing training corpus.
"""

from __future__ import annotations

from pathlib import Path
import html
import re


# ----------------------------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEB_DIR = PROJECT_ROOT / "data" / "web"
DOWNLOAD_DIR = WEB_DIR / "downloaded"
CLEANED_DIR = WEB_DIR / "cleaned"


# ----------------------------------------------------------------------
# HTML CLEANING
# ----------------------------------------------------------------------

def remove_html_comments(text: str) -> str:
    """Remove HTML comments."""

    return re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL,
    )


def remove_tag_block(text: str, tag: str) -> str:
    """Remove complete HTML blocks such as script/style/nav."""

    pattern = rf"<{tag}\b[^>]*>.*?</{tag}>"

    return re.sub(
        pattern,
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def remove_html_tags(text: str) -> str:
    """Remove remaining HTML tags."""

    return re.sub(
        r"<[^>]+>",
        " ",
        text,
    )


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving useful paragraph structure."""

    # Decode HTML entities.
    text = html.unescape(text)

    # Normalize Windows/newline variants.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

    # Remove spaces around newlines.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    return text.strip()


def clean_html(html_text: str) -> str:
    """Convert HTML into readable plain text."""

    text = html_text

    # Remove comments.
    text = remove_html_comments(text)

    # Remove elements that should never enter the language corpus.
    for tag in [
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
    ]:
        text = remove_tag_block(text, tag)

    # Convert some structural HTML elements to line breaks.
    text = re.sub(
        r"<(?:br|/p|/div|/section|/article|/li|/h1|/h2|/h3|/h4|/h5|/h6)\b[^>]*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining tags.
    text = remove_html_tags(text)

    # Decode entities and normalize whitespace.
    text = normalize_text(text)

    return text


# ----------------------------------------------------------------------
# FILE PROCESSING
# ----------------------------------------------------------------------

def clean_file(input_path: Path, output_path: Path) -> bool:
    """Clean one HTML file."""

    print(f"\nCleaning:")
    print(f"  {input_path.name}")

    try:
        html_text = input_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        cleaned = clean_html(html_text)

        if not cleaned:
            print("  WARNING: produced empty text")
            return False

        output_path.write_text(
            cleaned,
            encoding="utf-8",
        )

        print(
            f"  OK - "
            f"{len(html_text):,} HTML chars -> "
            f"{len(cleaned):,} text chars"
        )

        return True

    except Exception as exc:
        print(
            f"  FAILED - "
            f"{type(exc).__name__}: {exc}"
        )
        return False


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("GENIEE PYTHON 3.14 WEB DATA CLEANER")
    print("=" * 72)

    print(f"Input  : {DOWNLOAD_DIR}")
    print(f"Output : {CLEANED_DIR}")
    print()

    if not DOWNLOAD_DIR.exists():
        print("ERROR:")
        print(f"Download directory does not exist:")
        print(f"  {DOWNLOAD_DIR}")
        print()
        print("Run this first:")
        print("  python data\\web\\download_web_data.py")
        return

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    html_files = sorted(DOWNLOAD_DIR.glob("*.html"))

    if not html_files:
        print("ERROR: No HTML files found.")
        print("Run download_web_data.py first.")
        return

    successful = 0
    failed = 0

    for html_file in html_files:

        output_file = CLEANED_DIR / f"{html_file.stem}.txt"

        if clean_file(html_file, output_file):
            successful += 1
        else:
            failed += 1

    print()
    print("=" * 72)
    print("CLEANING COMPLETE")
    print("=" * 72)

    print(f"Input files  : {len(html_files)}")
    print(f"Successful   : {successful}")
    print(f"Failed       : {failed}")
    print(f"Output       : {CLEANED_DIR}")

    print()
    print("Checkpoint status:")
    print("  NO CHECKPOINT WAS MODIFIED.")


if __name__ == "__main__":
    main()