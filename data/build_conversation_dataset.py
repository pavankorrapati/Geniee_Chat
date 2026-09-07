from pathlib import Path
import json
import re
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
# PATHS
# ==========================================================

CORPUS_DIR = (
    PROJECT_ROOT / "corpus"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CONVERSATION_FILE = (
    OUTPUT_DIR
    / "conversations.jsonl"
)


# ==========================================================
# SETTINGS
# ==========================================================

ADD_SYSTEM_MESSAGE = True

SYSTEM_MESSAGE = (
    "You are Geniee, a helpful AI assistant. "
    "Answer clearly, accurately, and directly."
)


# ==========================================================
# QA EXTRACTION
# ==========================================================

QA_PATTERN = re.compile(
    r"(?ms)"
    r"^User:\s*(.*?)"
    r"^\s*Assistant:\s*(.*?)"
    r"(?=^\s*User:|\Z)"
)


# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_text(text):

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:

            lines.append(line)

    return "\n".join(lines)


# ==========================================================
# EXTRACT QA
# ==========================================================

def extract_qa_examples(text):

    matches = QA_PATTERN.findall(
        text
    )

    examples = []

    for question, answer in matches:

        question = question.strip()

        answer = answer.strip()

        if not question:
            continue

        if not answer:
            continue

        examples.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    return examples


# ==========================================================
# CREATE RECORD
# ==========================================================

def create_record(
    question,
    answer,
    source,
    category,
):

    messages = []

    if ADD_SYSTEM_MESSAGE:

        messages.append(
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return {
        "messages": messages,
        "metadata": {
            "source": source,
            "category": category,
        },
    }


# ==========================================================
# PROCESS CORPUS
# ==========================================================

def build_dataset():

    if not CORPUS_DIR.exists():

        raise FileNotFoundError(
            f"Corpus directory not found: "
            f"{CORPUS_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total_files = 0

    qa_files = 0

    plain_text_files = 0

    for path in sorted(
        CORPUS_DIR.rglob("*.txt")
    ):

        total_files += 1

        try:

            text = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            print(
                f"Skipping non UTF-8 file: "
                f"{path}"
            )

            continue

        text = normalize_text(
            text
        )

        if not text:

            continue

        relative_path = (
            path.relative_to(
                CORPUS_DIR
            )
        )

        if len(relative_path.parts) > 1:

            category = (
                relative_path.parts[0]
            )

        else:

            category = "root"

        qa_examples = extract_qa_examples(
            text
        )

        if qa_examples:

            qa_files += 1

            for example in qa_examples:

                record = create_record(
                    question=example[
                        "question"
                    ],
                    answer=example[
                        "answer"
                    ],
                    source=str(
                        relative_path
                    ),
                    category=category,
                )

                records.append(
                    record
                )
        else:

            plain_text_files += 1

            print(
                f"Skipping plain document for "
                f"instruction tuning: {relative_path}"
            )

            continue

        # else:

        #     plain_text_files += 1

        #     # --------------------------------------------------
        #     # Plain documents are converted into a simple
        #     # assistant-style training conversation.
        #     # --------------------------------------------------

        #     record = {

        #         "messages": [

        #             {
        #                 "role": "user",
        #                 "content": (
        #                     "Explain the following "
        #                     "information:\n\n"
        #                     + text
        #                 ),
        #             },

        #             {
        #                 "role": "assistant",
        #                 "content": text,
        #             },

        #         ],

        #         "metadata": {
        #             "source": str(
        #                 relative_path
        #             ),
        #             "category": category,
        #             "source_type": "document",
        #         },
        #     }

        #     if ADD_SYSTEM_MESSAGE:

        #         record["messages"].insert(
        #             0,
        #             {
        #                 "role": "system",
        #                 "content":
        #                     SYSTEM_MESSAGE,
        #             },
        #         )

        #     records.append(
        #         record
        #     )

    if not records:

        raise ValueError(
            "No training conversations "
            "were generated."
        )

    # ======================================================
    # REMOVE EXACT DUPLICATES
    # ======================================================

    unique_records = []

    seen = set()

    duplicates = 0

    for record in records:

        key = json.dumps(
            record["messages"],
            ensure_ascii=False,
            sort_keys=True,
        )

        if key in seen:

            duplicates += 1

            continue

        seen.add(key)

        unique_records.append(
            record
        )

    records = unique_records

    # ======================================================
    # WRITE
    # ======================================================

    with CONVERSATION_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in records:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            handle.write("\n")

    # ======================================================
    # SUMMARY
    # ======================================================

    print()

    print("=" * 70)

    print(
        "GENIEE V2 CONVERSATION DATASET BUILDER"
    )

    print("=" * 70)

    print(
        f"Corpus directory     : {CORPUS_DIR}"
    )

    print(
        f"Files discovered     : {total_files}"
    )

    print(
        f"Q&A files            : {qa_files}"
    )

    print(
        f"Plain text files     : {plain_text_files}"
    )

    print(
        f"Duplicate records    : {duplicates}"
    )

    print(
        f"Final conversations  : {len(records)}"
    )

    print(
        f"Output               : "
        f"{CONVERSATION_FILE}"
    )

    print("=" * 70)

    # ======================================================
    # EXAMPLES
    # ======================================================

    print()

    print(
        "FIRST 3 CONVERSATIONS"
    )

    print("=" * 70)

    for index, record in enumerate(
        records[:3],
        start=1,
    ):

        print()

        print(
            f"--- Conversation {index} ---"
        )

        for message in record[
            "messages"
        ]:

            print(
                f"[{message['role']}]"
            )

            print(
                message["content"]
            )

    print()

    print("=" * 70)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    build_dataset()