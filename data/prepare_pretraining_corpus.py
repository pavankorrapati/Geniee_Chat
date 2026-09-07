from __future__ import annotations

from pathlib import Path
import random


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_CORPUS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "corpus_v2.txt"
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


# ============================================================
# FUNCTIONS
# ============================================================

def load_corpus() -> str:

    if not RAW_CORPUS.exists():
        raise FileNotFoundError(
            f"Raw corpus not found: {RAW_CORPUS}"
        )

    text = RAW_CORPUS.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            f"Raw corpus is empty: {RAW_CORPUS}"
        )

    return text


def split_corpus(text: str):

    # Split by paragraphs.
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    if len(paragraphs) < 2:
        raise ValueError(
            "Corpus must contain at least two paragraphs."
        )

    random.seed(RANDOM_SEED)

    random.shuffle(paragraphs)

    validation_count = max(
        1,
        int(len(paragraphs) * VALIDATION_RATIO),
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

    text = "\n\n".join(paragraphs)

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
    print(f"Raw corpus : {RAW_CORPUS}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Validation: {VALIDATION_RATIO * 100:.0f}%")
    print(f"Seed      : {RANDOM_SEED}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    text = load_corpus()

    print()
    print(f"Raw characters: {len(text):,}")

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_paragraphs, validation_paragraphs = (
        split_corpus(text)
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

    validation_text = VALIDATION_FILE.read_text(
        encoding="utf-8"
    )

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print()
    print(
        f"Train paragraphs     : {len(train_paragraphs)}"
    )

    print(
        f"Validation paragraphs: "
        f"{len(validation_paragraphs)}"
    )

    print(
        f"Train characters     : {len(train_text):,}"
    )

    print(
        f"Validation characters: "
        f"{len(validation_text):,}"
    )

    print()
    print(f"Train file      : {TRAIN_FILE}")
    print(f"Validation file : {VALIDATION_FILE}")

    print()
    print("=" * 70)
    print("CORPUS PREPARATION COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()