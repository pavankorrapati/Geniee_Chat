from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class GenieeConversationDataset:
    """
    Geniee V2 structured conversation dataset.

    Responsibilities:
        - Load JSONL conversations
        - Validate conversations
        - Remove exact duplicates
        - Split into train / validation / test
        - Calculate dataset statistics
        - Render conversations using Geniee chat format

    JSONL format:

    {
        "messages": [
            {
                "role": "system",
                "content": "You are Geniee, a helpful AI assistant."
            },
            {
                "role": "user",
                "content": "What is Selenium?"
            },
            {
                "role": "assistant",
                "content": "Selenium is a framework..."
            }
        ],
        "metadata": {
            "source": "selenium_qa.txt",
            "category": "instructions"
        }
    }
    """

    VALID_ROLES = {
        "system",
        "user",
        "assistant",
    }

    def __init__(
        self,
        dataset_path,
        seed=42,
        validation_split=0.10,
        test_split=0.10,
    ):
        self.dataset_path = Path(dataset_path)

        self.seed = seed

        self.validation_split = validation_split
        self.test_split = test_split

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}"
            )

        if validation_split < 0:
            raise ValueError(
                "validation_split must be >= 0."
            )

        if test_split < 0:
            raise ValueError(
                "test_split must be >= 0."
            )

        if validation_split + test_split >= 1.0:
            raise ValueError(
                "validation_split + test_split "
                "must be less than 1.0."
            )

        self.records = []

        self.train_records = []
        self.validation_records = []
        self.test_records = []

        self.duplicate_count = 0

    # ==========================================================
    # TEXT NORMALIZATION
    # ==========================================================

    @staticmethod
    def normalize_content(content):
        if not isinstance(content, str):
            raise ValueError(
                "Message content must be a string."
            )

        content = content.strip()

        return content

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def validate_record(
        cls,
        record,
        line_number=0,
    ):
        if not isinstance(record, dict):
            raise ValueError(
                f"Line {line_number}: "
                "record must be a JSON object."
            )

        messages = record.get("messages")

        if not isinstance(messages, list):
            raise ValueError(
                f"Line {line_number}: "
                "'messages' must be a list."
            )

        if not messages:
            raise ValueError(
                f"Line {line_number}: "
                "'messages' cannot be empty."
            )

        cleaned_messages = []

        for index, message in enumerate(messages):

            if not isinstance(message, dict):
                raise ValueError(
                    f"Line {line_number}, "
                    f"message {index}: "
                    "message must be an object."
                )

            role = message.get("role")
            content = message.get("content")

            if role not in cls.VALID_ROLES:
                raise ValueError(
                    f"Line {line_number}, "
                    f"message {index}: "
                    f"invalid role '{role}'."
                )

            content = cls.normalize_content(
                content
            )

            if not content:
                raise ValueError(
                    f"Line {line_number}, "
                    f"message {index}: "
                    "content cannot be empty."
                )

            cleaned_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        has_user = any(
            message["role"] == "user"
            for message in cleaned_messages
        )

        has_assistant = any(
            message["role"] == "assistant"
            for message in cleaned_messages
        )

        if not has_user:
            raise ValueError(
                f"Line {line_number}: "
                "conversation must contain "
                "a user message."
            )

        if not has_assistant:
            raise ValueError(
                f"Line {line_number}: "
                "conversation must contain "
                "an assistant message."
            )

        metadata = record.get(
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Line {line_number}: "
                "metadata must be an object."
            )

        return {
            "messages": cleaned_messages,
            "metadata": metadata,
        }

    # ==========================================================
    # DUPLICATE DETECTION
    # ==========================================================

    @staticmethod
    def conversation_key(record):

        return json.dumps(
            record["messages"],
            ensure_ascii=False,
            sort_keys=True,
        )

    # ==========================================================
    # LOAD JSONL
    # ==========================================================

    def load(self):

        records = []

        seen = set()

        duplicates = 0

        with self.dataset_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line_number, line in enumerate(
                handle,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    raw = json.loads(line)

                except json.JSONDecodeError as exc:

                    raise ValueError(
                        f"Line {line_number}: "
                        f"Invalid JSON: {exc}"
                    ) from exc

                record = self.validate_record(
                    raw,
                    line_number,
                )

                key = self.conversation_key(
                    record
                )

                if key in seen:

                    duplicates += 1

                    continue

                seen.add(key)

                records.append(record)

        if not records:

            raise ValueError(
                f"No valid conversations found in "
                f"{self.dataset_path}"
            )

        self.records = records

        self.duplicate_count = duplicates

        return records

    # ==========================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ==========================================================

    def split(self):

        if not self.records:
            self.load()

        records = list(self.records)

        random_generator = random.Random(
            self.seed
        )

        random_generator.shuffle(records)

        total = len(records)

        validation_count = int(
            total * self.validation_split
        )

        test_count = int(
            total * self.test_split
        )

        if (
            self.validation_split > 0
            and validation_count == 0
            and total >= 3
        ):
            validation_count = 1

        if (
            self.test_split > 0
            and test_count == 0
            and total >= 3
        ):
            test_count = 1

        if (
            validation_count
            + test_count
            >= total
        ):
            raise ValueError(
                "Not enough records to create "
                "training split."
            )

        self.test_records = records[
            :test_count
        ]

        self.validation_records = records[
            test_count:
            test_count + validation_count
        ]

        self.train_records = records[
            test_count + validation_count:
        ]

        return (
            self.train_records,
            self.validation_records,
            self.test_records,
        )

    # ==========================================================
    # CHAT TEMPLATE
    # ==========================================================

    @staticmethod
    def render_chat(
        messages,
        add_generation_prompt=False,
    ):

        parts = []

        for message in messages:

            role = message["role"]

            content = message["content"]

            parts.append(
                f"<|{role}|>"
            )

            parts.append(content)

        if add_generation_prompt:

            parts.append(
                "<|assistant|>"
            )

        return "\n".join(parts)

    # ==========================================================
    # RENDER RECORD
    # ==========================================================

    @classmethod
    def render_record(
        cls,
        record,
        add_generation_prompt=False,
    ):

        return cls.render_chat(
            record["messages"],
            add_generation_prompt,
        )

    # ==========================================================
    # WRITE JSONL
    # ==========================================================

    @staticmethod
    def write_jsonl(
        records,
        output_path,
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
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

    # ==========================================================
    # STATISTICS
    # ==========================================================

    def statistics(self):

        if not self.records:
            self.load()

        role_counts = Counter()

        category_counts = Counter()

        source_counts = Counter()

        conversation_lengths = []

        word_counts = []

        for record in self.records:

            messages = record[
                "messages"
            ]

            conversation_lengths.append(
                len(messages)
            )

            for message in messages:

                role = message["role"]

                role_counts[role] += 1

                word_counts.append(
                    len(
                        message["content"].split()
                    )
                )

            metadata = record.get(
                "metadata",
                {},
            )

            category = metadata.get(
                "category"
            )

            if category:

                category_counts[
                    str(category)
                ] += 1

            source = metadata.get(
                "source"
            )

            if source:

                source_counts[
                    str(source)
                ] += 1

        total_words = sum(
            word_counts
        )

        return {

            "records": len(
                self.records
            ),

            "duplicates_removed":
                self.duplicate_count,

            "messages":
                sum(role_counts.values()),

            "system_messages":
                role_counts["system"],

            "user_messages":
                role_counts["user"],

            "assistant_messages":
                role_counts["assistant"],

            "avg_messages_per_conversation":
                sum(conversation_lengths)
                / len(conversation_lengths),

            "estimated_words":
                total_words,

            "estimated_words_per_conversation":
                total_words
                / len(self.records),

            "max_words_per_message":
                max(word_counts)
                if word_counts
                else 0,

            "categories":
                dict(category_counts),

            "sources":
                dict(source_counts),
        }

    # ==========================================================
    # BUILD SPLITS
    # ==========================================================

    def build_splits(
        self,
        output_dir,
    ):

        if not self.records:
            self.load()

        self.split()

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = (
            output_dir
            / "train.jsonl"
        )

        validation_path = (
            output_dir
            / "validation.jsonl"
        )

        test_path = (
            output_dir
            / "test.jsonl"
        )

        self.write_jsonl(
            self.train_records,
            train_path,
        )

        self.write_jsonl(
            self.validation_records,
            validation_path,
        )

        self.write_jsonl(
            self.test_records,
            test_path,
        )

        return {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
        }