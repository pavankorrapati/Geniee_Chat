# from __future__ import annotations

# from pathlib import Path
# import random


# # ============================================================
# # PATHS
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAW_CORPUS = (
#     PROJECT_ROOT
#     / "data"
#     / "raw"
#     / "corpus_web.txt"
# )

# OUTPUT_DIR = (
#     PROJECT_ROOT
#     / "data"
#     / "processed"
#     / "splits"
# )

# TRAIN_FILE = OUTPUT_DIR / "train.txt"
# VALIDATION_FILE = OUTPUT_DIR / "validation.txt"


# # ============================================================
# # CONFIGURATION
# # ============================================================

# VALIDATION_RATIO = 0.10
# RANDOM_SEED = 42


# # ============================================================
# # FUNCTIONS
# # ============================================================

# def load_corpus() -> str:

#     if not RAW_CORPUS.exists():
#         raise FileNotFoundError(
#             f"Raw corpus not found: {RAW_CORPUS}"
#         )

#     text = RAW_CORPUS.read_text(
#         encoding="utf-8"
#     ).strip()

#     if not text:
#         raise ValueError(
#             f"Raw corpus is empty: {RAW_CORPUS}"
#         )

#     return text


# def split_corpus(text: str):

#     # Split by paragraphs.
#     paragraphs = [
#         paragraph.strip()
#         for paragraph in text.split("\n\n")
#         if paragraph.strip()
#     ]

#     if len(paragraphs) < 2:
#         raise ValueError(
#             "Corpus must contain at least two paragraphs."
#         )

#     random.seed(RANDOM_SEED)

#     random.shuffle(paragraphs)

#     validation_count = max(
#         1,
#         int(len(paragraphs) * VALIDATION_RATIO),
#     )

#     validation_paragraphs = paragraphs[
#         :validation_count
#     ]

#     train_paragraphs = paragraphs[
#         validation_count:
#     ]

#     return (
#         train_paragraphs,
#         validation_paragraphs,
#     )


# def save_split(
#     paragraphs: list[str],
#     path: Path,
# ):

#     path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     text = "\n\n".join(paragraphs)

#     path.write_text(
#         text + "\n",
#         encoding="utf-8",
#     )


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print()
#     print("=" * 70)
#     print("GENIEE PRETRAINING CORPUS PREPARATION")
#     print("=" * 70)

#     print()
#     print(f"Raw corpus : {RAW_CORPUS}")
#     print(f"Output dir : {OUTPUT_DIR}")
#     print(f"Validation: {VALIDATION_RATIO * 100:.0f}%")
#     print(f"Seed      : {RANDOM_SEED}")

#     # --------------------------------------------------------
#     # Load
#     # --------------------------------------------------------

#     text = load_corpus()

#     print()
#     print(f"Raw characters: {len(text):,}")

#     # --------------------------------------------------------
#     # Split
#     # --------------------------------------------------------

#     train_paragraphs, validation_paragraphs = (
#         split_corpus(text)
#     )

#     # --------------------------------------------------------
#     # Save
#     # --------------------------------------------------------

#     save_split(
#         train_paragraphs,
#         TRAIN_FILE,
#     )

#     save_split(
#         validation_paragraphs,
#         VALIDATION_FILE,
#     )

#     # --------------------------------------------------------
#     # Statistics
#     # --------------------------------------------------------

#     train_text = TRAIN_FILE.read_text(
#         encoding="utf-8"
#     )

#     validation_text = VALIDATION_FILE.read_text(
#         encoding="utf-8"
#     )

#     print()
#     print("-" * 70)
#     print("RESULT")
#     print("-" * 70)

#     print()
#     print(
#         f"Train paragraphs     : {len(train_paragraphs)}"
#     )

#     print(
#         f"Validation paragraphs: "
#         f"{len(validation_paragraphs)}"
#     )

#     print(
#         f"Train characters     : {len(train_text):,}"
#     )

#     print(
#         f"Validation characters: "
#         f"{len(validation_text):,}"
#     )

#     print()
#     print(f"Train file      : {TRAIN_FILE}")
#     print(f"Validation file : {VALIDATION_FILE}")

#     print()
#     print("=" * 70)
#     print("CORPUS PREPARATION COMPLETE")
#     print("=" * 70)
#     print()


# if __name__ == "__main__":
#     main()
from __future__ import annotations

from pathlib import Path
import random


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

TRAIN_FILE = OUTPUT_DIR / "train.txt"
VALIDATION_FILE = OUTPUT_DIR / "validation.txt"


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

# Automatically include every .txt file under data/raw
RAW_FILE_PATTERN = "*.txt"


# ============================================================
# FUNCTIONS
# ============================================================

def discover_raw_files() -> list[Path]:
    """
    Discover every TXT file inside data/raw.

    Files are processed in sorted order so that the input
    order is deterministic before the random paragraph shuffle.
    """

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw corpus directory not found: {RAW_DIR}"
        )

    files = sorted(
        RAW_DIR.glob(RAW_FILE_PATTERN)
    )

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in raw corpus directory: {RAW_DIR}"
        )

    return files


def load_corpora(
    raw_files: list[Path],
) -> tuple[str, dict[str, int]]:
    """
    Load and combine all discovered raw TXT files.

    Returns:
        combined_text
        source_statistics
    """

    combined_parts: list[str] = []
    source_statistics: dict[str, int] = {}

    for path in raw_files:

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            print(
                f"WARNING: Skipping empty file: {path.name}"
            )
            continue

        character_count = len(text)

        source_statistics[
            path.name
        ] = character_count

        combined_parts.append(text)

    if not combined_parts:
        raise ValueError(
            f"All TXT files in {RAW_DIR} are empty."
        )

    # Separate documents with two newlines so that
    # paragraphs from different source files remain distinct.
    combined_text = "\n\n".join(
        combined_parts
    )

    return (
        combined_text,
        source_statistics,
    )


def split_corpus(text: str):

    # --------------------------------------------------------
    # Split by paragraphs
    # --------------------------------------------------------

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    if len(paragraphs) < 2:
        raise ValueError(
            "Combined corpus must contain at least two paragraphs."
        )

    # --------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(
        paragraphs
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    validation_count = max(
        1,
        int(
            len(paragraphs)
            * VALIDATION_RATIO
        ),
    )

    validation_paragraphs = paragraphs[
        :validation_count
    ]

    train_paragraphs = paragraphs[
        validation_count:
    ]

    return (
        train_paragraphs,
        validation_paragraphs,
    )


def save_split(
    paragraphs: list[str],
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = "\n\n".join(
        paragraphs
    )

    path.write_text(
        text + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GENIEE PRETRAINING CORPUS PREPARATION")
    print("=" * 70)

    print()
    print(f"Raw directory : {RAW_DIR}")
    print(f"Output dir    : {OUTPUT_DIR}")
    print(
        f"Validation    : "
        f"{VALIDATION_RATIO * 100:.0f}%"
    )
    print(
        f"Seed          : {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Discover raw files
    # --------------------------------------------------------

    raw_files = discover_raw_files()

    print()
    print("-" * 70)
    print("RAW FILES DISCOVERED")
    print("-" * 70)

    print()

    for index, path in enumerate(
        raw_files,
        start=1,
    ):
        print(
            f"{index:3}. {path.name}"
        )

    print()
    print(
        f"Total raw TXT files: "
        f"{len(raw_files)}"
    )

    # --------------------------------------------------------
    # Load and combine
    # --------------------------------------------------------

    combined_text, source_statistics = (
        load_corpora(
            raw_files
        )
    )

    print()
    print("-" * 70)
    print("SOURCE STATISTICS")
    print("-" * 70)

    print()

    total_characters = 0

    for filename, character_count in (
        source_statistics.items()
    ):

        print(
            f"{filename:<40} "
            f"{character_count:>10,} characters"
        )

        total_characters += character_count

    print()
    print(
        f"{'TOTAL':<40} "
        f"{total_characters:>10,} characters"
    )

    print()
    print(
        f"Combined corpus characters: "
        f"{len(combined_text):,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_paragraphs, validation_paragraphs = (
        split_corpus(
            combined_text
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_split(
        train_paragraphs,
        TRAIN_FILE,
    )

    save_split(
        validation_paragraphs,
        VALIDATION_FILE,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    train_text = TRAIN_FILE.read_text(
        encoding="utf-8"
    )

    validation_text = (
        VALIDATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print()
    print(
        f"Train paragraphs     : "
        f"{len(train_paragraphs):,}"
    )

    print(
        f"Validation paragraphs: "
        f"{len(validation_paragraphs):,}"
    )

    print(
        f"Train characters     : "
        f"{len(train_text):,}"
    )

    print(
        f"Validation characters: "
        f"{len(validation_text):,}"
    )

    print()
    print(
        f"Train file      : "
        f"{TRAIN_FILE}"
    )

    print(
        f"Validation file : "
        f"{VALIDATION_FILE}"
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("RAW CORPUS VERIFICATION")
    print("-" * 70)

    print()

    print(
        f"✓ Raw directory exists"
    )

    print(
        f"✓ {len(raw_files)} TXT file(s) discovered"
    )

    print(
        f"✓ All non-empty TXT files included"
    )

    print(
        f"✓ Combined corpus created"
    )

    print(
        f"✓ Train/validation split created"
    )

    print()
    print("=" * 70)
    print("CORPUS PREPARATION COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()