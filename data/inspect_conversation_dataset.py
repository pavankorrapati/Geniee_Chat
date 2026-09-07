from pathlib import Path
import json
import sys


# ==========================================================
# PROJECT ROOT
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
# IMPORT
# ==========================================================

from data.conversation_dataset import (
    GenieeConversationDataset
)


# ==========================================================
# PATHS
# ==========================================================

CONVERSATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "conversations.jsonl"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)


# ==========================================================
# COUNT JSONL
# ==========================================================

def count_records(path):

    if not path.exists():

        return 0

    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            if line.strip():

                count += 1

    return count


# ==========================================================
# LOAD FIRST RECORD
# ==========================================================

def first_record(path):

    if not path.exists():

        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:

                continue

            return json.loads(line)

    return None


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()

    print("=" * 70)

    print(
        "GENIEE V2 CONVERSATION DATASET INSPECTION"
    )

    print("=" * 70)

    # ======================================================
    # SOURCE DATASET
    # ======================================================

    dataset = (
        GenieeConversationDataset(
            dataset_path=CONVERSATION_FILE,
            seed=42,
            validation_split=0.10,
            test_split=0.10,
        )
    )

    records = dataset.load()

    stats = dataset.statistics()

    print()

    print("SOURCE DATASET")

    print("-" * 70)

    print(
        f"Conversations             : "
        f"{stats['records']}"
    )

    print(
        f"Duplicates removed        : "
        f"{stats['duplicates_removed']}"
    )

    print(
        f"Total messages            : "
        f"{stats['messages']}"
    )

    print(
        f"System messages           : "
        f"{stats['system_messages']}"
    )

    print(
        f"User messages             : "
        f"{stats['user_messages']}"
    )

    print(
        f"Assistant messages        : "
        f"{stats['assistant_messages']}"
    )

    print(
        f"Avg messages/conversation : "
        f"{stats['avg_messages_per_conversation']:.2f}"
    )

    print(
        f"Estimated words           : "
        f"{stats['estimated_words']}"
    )

    print(
        f"Avg words/conversation    : "
        f"{stats['estimated_words_per_conversation']:.2f}"
    )

    # ======================================================
    # CATEGORIES
    # ======================================================

    print()

    print("CATEGORIES")

    print("-" * 70)

    if stats["categories"]:

        for category, count in sorted(
            stats["categories"].items()
        ):

            print(
                f"{category:<25} {count}"
            )

    else:

        print(
            "No category metadata."
        )

    # ======================================================
    # SOURCES
    # ======================================================

    print()

    print("SOURCE FILES")

    print("-" * 70)

    for source, count in sorted(
        stats["sources"].items()
    ):

        print(
            f"{source:<45} {count}"
        )

    # ======================================================
    # BUILD SPLITS
    # ======================================================

    paths = dataset.build_splits(
        SPLIT_DIR
    )

    print()

    print("TRAIN / VALIDATION / TEST")

    print("-" * 70)

    print(
        f"Train       : "
        f"{len(dataset.train_records)}"
    )

    print(
        f"Validation  : "
        f"{len(dataset.validation_records)}"
    )

    print(
        f"Test        : "
        f"{len(dataset.test_records)}"
    )

    print()

    for name, path in paths.items():

        print(
            f"{name:<12}: {path}"
        )

    # ======================================================
    # SHOW EXAMPLES
    # ======================================================

    print()

    print(
        "CHAT TEMPLATE EXAMPLES"
    )

    print("=" * 70)

    for index, record in enumerate(
        dataset.train_records[:3],
        start=1,
    ):

        print()

        print(
            f"--- Example {index} ---"
        )

        print(
            dataset.render_record(
                record
            )
        )

    # ======================================================
    # GENERATION PROMPT
    # ======================================================

    print()

    print(
        "GENERATION PROMPT EXAMPLE"
    )

    print("=" * 70)

    if dataset.train_records:

        print(
            dataset.render_record(
                dataset.train_records[0],
                add_generation_prompt=True,
            )
        )

    # ======================================================
    # TEST SELENIUM
    # ======================================================

    print()

    print(
        "SELENIUM CHECK"
    )

    print("-" * 70)

    selenium_records = []

    for record in records:

        text = json.dumps(
            record,
            ensure_ascii=False,
        )

        if "Selenium" in text:

            selenium_records.append(
                record
            )

    print(
        f"Selenium conversations found: "
        f"{len(selenium_records)}"
    )

    for record in selenium_records[
        :5
    ]:

        print()

        print(
            dataset.render_record(
                record
            )
        )

    print()

    print("=" * 70)

    print(
        "INSPECTION COMPLETE"
    )

    print("=" * 70)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    main()