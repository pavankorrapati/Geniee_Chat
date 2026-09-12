from pathlib import Path
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "corpus_v2.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "corpus_clean.txt"


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def clean_corpus(text: str) -> str:
    """
    Clean the Geniee pretraining corpus.

    Removes:
        - Long separator lines such as ========================
        - Decorative section separators
        - Excessive blank lines
        - Trailing whitespace
        - Repeated blank lines

    Keeps:
        - Normal headings
        - Paragraphs
        - Code examples
        - Technical terminology
        - Natural punctuation
    """

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        # ----------------------------------------------------
        # Remove whitespace at beginning/end
        # ----------------------------------------------------
        line = line.strip()

        # ----------------------------------------------------
        # Remove empty lines temporarily
        # ----------------------------------------------------
        if not line:
            cleaned_lines.append("")
            continue

        # ----------------------------------------------------
        # Remove decorative separator lines
        #
        # Examples:
        # ====================================================
        # ----------------------------------------------------
        # ****************************************************
        # ----------------------------------------------------
        # We only remove lines that are primarily decoration.
        # ----------------------------------------------------

        if re.fullmatch(r"[=\-_*#]{10,}", line):
            continue

        # ----------------------------------------------------
        # Remove lines containing only repeated "=" characters
        # even if there are spaces between them.
        # ----------------------------------------------------

        normalized = re.sub(r"\s+", "", line)

        if (
            len(normalized) >= 10
            and set(normalized) == {"="}
        ):
            continue

        # ----------------------------------------------------
        # Remove excessive spaces inside a line
        # ----------------------------------------------------

        line = re.sub(r"[ \t]+", " ", line)

        cleaned_lines.append(line)

    # ========================================================
    # Remove excessive blank lines
    # ========================================================

    result_lines = []

    blank_count = 0

    for line in cleaned_lines:

        if line == "":
            blank_count += 1

            # Keep at most ONE blank line
            if blank_count > 1:
                continue

        else:
            blank_count = 0

        result_lines.append(line)

    # ========================================================
    # Final cleanup
    # ========================================================

    text = "\n".join(result_lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Ensure exactly one newline at EOF
    text += "\n"

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GENIEE CORPUS CLEANING")
    print("=" * 70)

    print()
    print(f"Input  : {INPUT_FILE}")
    print(f"Output : {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input corpus not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Read corpus
    # --------------------------------------------------------

    original_text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    original_characters = len(original_text)
    original_lines = len(original_text.splitlines())

    # Count separator lines before cleaning
    original_separator_lines = sum(
        1
        for line in original_text.splitlines()
        if re.fullmatch(r"[=\-_*#]{10,}", line.strip())
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    cleaned_text = clean_corpus(original_text)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    cleaned_characters = len(cleaned_text)
    cleaned_lines = len(cleaned_text.splitlines())

    cleaned_separator_lines = sum(
        1
        for line in cleaned_text.splitlines()
        if re.fullmatch(r"[=\-_*#]{10,}", line.strip())
    )

    removed_characters = (
        original_characters - cleaned_characters
    )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        cleaned_text,
        encoding="utf-8"
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print(f"Original characters : {original_characters:,}")
    print(f"Clean characters    : {cleaned_characters:,}")
    print(f"Removed characters  : {removed_characters:,}")

    print()

    print(f"Original lines      : {original_lines:,}")
    print(f"Clean lines         : {cleaned_lines:,}")

    print()

    print(
        f"Separator lines before : "
        f"{original_separator_lines:,}"
    )

    print(
        f"Separator lines after  : "
        f"{cleaned_separator_lines:,}"
    )

    print()

    print(f"Clean corpus : {OUTPUT_FILE}")

    print()
    print("=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()