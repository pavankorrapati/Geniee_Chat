from pathlib import Path
import sys


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ==========================================================
# Geniee imports
# ==========================================================

from data.corpus import GenieeCorpus


# ==========================================================
# Configuration
# ==========================================================

CORPUS_DIR = (
    PROJECT_ROOT / "corpus"
)


# ==========================================================
# Main
# ==========================================================

def main():

    print()
    print("=" * 60)
    print("GENIEE CORPUS TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Create corpus
    # --------------------------------------------------

    corpus = GenieeCorpus(
        corpus_dir=CORPUS_DIR,
        validation_split=0.1,
        seed=42
    )

    # --------------------------------------------------
    # Find files
    # --------------------------------------------------

    files = corpus.get_text_files()

    print()
    print(
        f"Corpus files found: {len(files)}"
    )

    print()

    for file_path in files:

        relative_path = (
            file_path.relative_to(
                PROJECT_ROOT
            )
        )

        print(
            f"✓ {relative_path}"
        )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    stats = corpus.statistics()

    print()
    print("-" * 60)
    print("CORPUS STATISTICS")
    print("-" * 60)

    print(
        f"Files       : {stats['files']}"
    )

    print(
        f"Characters  : "
        f"{stats['characters']:,}"
    )

    print(
        f"Words       : "
        f"{stats['words']:,}"
    )

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------

    train_text, validation_text = (
        corpus.train_validation_split()
    )

    print()
    print("-" * 60)
    print("TRAIN / VALIDATION")
    print("-" * 60)

    print(
        f"Train characters      : "
        f"{len(train_text):,}"
    )

    print(
        f"Validation characters : "
        f"{len(validation_text):,}"
    )

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------

    print()
    print("-" * 60)
    print("TRAIN PREVIEW")
    print("-" * 60)

    print(
        train_text[:500]
    )

    print()
    print("-" * 60)
    print("VALIDATION PREVIEW")
    print("-" * 60)

    print(
        validation_text[:500]
    )

    # --------------------------------------------------
    # Success
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("CORPUS TEST SUCCESSFUL")
    print("=" * 60)


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()