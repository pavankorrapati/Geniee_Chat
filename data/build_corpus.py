from pathlib import Path
import sys


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PATHS
# ==========================================================

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CORPUS_DIR = RAW_DIR / "corpus"

OUTPUT_FILE = RAW_DIR / "corpus_v2.txt"


# ==========================================================
# CONFIGURATION
# ==========================================================

MIN_FILE_SIZE = 100


# ==========================================================
# HELPERS
# ==========================================================

def clean_text(text: str) -> str:
    """
    Basic corpus cleaning.

    Removes excessive blank lines and trailing whitespace.
    """

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        lines.append(line)

    return "\n".join(lines)


def load_text_file(path: Path) -> str:

    try:

        text = path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    return clean_text(text)


# ==========================================================
# BUILD CORPUS
# ==========================================================

def build_corpus():

    print("=" * 70)
    print("GENIEE CORPUS BUILDER")
    print("=" * 70)

    print()
    print("Corpus directory:")
    print(CORPUS_DIR)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    # ------------------------------------------------------
    # Create directory
    # ------------------------------------------------------

    CORPUS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Find files
    # ------------------------------------------------------

    files = sorted(
        CORPUS_DIR.glob("*.txt")
    )

    if not files:

        print()
        print("ERROR: No .txt files found.")
        print()
        print("Add training files to:")
        print(CORPUS_DIR)
        print()

        return

    print()
    print("Files discovered:", len(files))
    print()

    corpus_sections = []

    total_characters = 0

    # ------------------------------------------------------
    # Process each file
    # ------------------------------------------------------

    for index, path in enumerate(files, start=1):

        text = load_text_file(path)

        character_count = len(text)

        if character_count < MIN_FILE_SIZE:

            print(
                f"[SKIP] {path.name:<35} "
                f"{character_count:>8} chars"
            )

            continue

        print(
            f"[{index:02d}] {path.name:<35} "
            f"{character_count:>8} chars"
        )

        # --------------------------------------------------
        # Section boundary
        # --------------------------------------------------

        section = (
            "\n"
            "============================================================\n"
            f"TOPIC: {path.stem.replace('_', ' ').title()}\n"
            "============================================================\n"
            "\n"
            + text
            + "\n"
        )

        corpus_sections.append(section)

        total_characters += character_count

    # ------------------------------------------------------
    # Existing baseline corpus
    # ------------------------------------------------------

    baseline = RAW_DIR / "corpus.txt"

    if baseline.exists():

        print()
        print(
            "Adding baseline corpus:",
            baseline.name
        )

        baseline_text = load_text_file(
            baseline
        )

        if baseline_text:

            section = (
                "\n"
                "============================================================\n"
                "TOPIC: General Geniee Knowledge\n"
                "============================================================\n"
                "\n"
                + baseline_text
                + "\n"
            )

            corpus_sections.insert(
                0,
                section
            )

            total_characters += len(
                baseline_text
            )

    # ------------------------------------------------------
    # Write corpus
    # ------------------------------------------------------

    final_text = "\n".join(
        corpus_sections
    )

    OUTPUT_FILE.write_text(
        final_text,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("CORPUS BUILD COMPLETE")
    print("=" * 70)

    print()
    print("Source files      :", len(files))
    print("Output characters :", f"{len(final_text):,}")
    print(
        "Output file       :",
        OUTPUT_FILE
    )

    print()
    print("Next step:")
    print(
        "python data\\prepare_pretraining_corpus.py"
    )

    print()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    build_corpus()