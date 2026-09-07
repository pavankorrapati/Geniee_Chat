from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ==========================================================
# Geniee Chat Format
# ==========================================================

SYSTEM_PREFIX = "<|system|>\n"
USER_PREFIX = "<|user|>\n"
ASSISTANT_PREFIX = "<|assistant|>\n"


class GenieeInstructionDataset(Dataset):
    """
    Dataset for supervised instruction tuning.

    Important:
        Only assistant responses contribute to the loss.

    Example:

        <|system|>
        You are Geniee...

        <|user|>
        What is Selenium?

        <|assistant|>
        Selenium is a browser automation framework.

    Training behavior:

        system tokens     -> loss = 0
        user tokens       -> loss = 0
        assistant tokens  -> loss = 1
        padding            -> loss = 0

    Returned tensors:

        input_ids
        target_ids
        loss_mask
        attention_mask
    """

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_seq_len,
        pad_id=None,
    ):
        super().__init__()

        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if pad_id is None:
            pad_id = tokenizer.pad_id

        self.pad_id = pad_id

        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Instruction dataset not found: "
                f"{self.jsonl_path}"
            )

        if self.max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be greater than 0."
            )

        self.records = []
        self.samples = []

        self._load_records()
        self._build_samples()

        if not self.samples:
            raise ValueError(
                "No valid instruction samples were created."
            )

    # ======================================================
    # Load JSONL
    # ======================================================

    def _load_records(self):

        with self.jsonl_path.open(
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

                    record = json.loads(line)

                except json.JSONDecodeError as exc:

                    raise ValueError(
                        f"Invalid JSON at "
                        f"{self.jsonl_path}:"
                        f"{line_number}\n"
                        f"{exc}"
                    ) from exc

                self._validate_record(
                    record,
                    line_number,
                )

                self.records.append(record)

        if not self.records:
            raise ValueError(
                f"No records found in "
                f"{self.jsonl_path}"
            )

    # ======================================================
    # Validate record
    # ======================================================

    @staticmethod
    def _validate_record(
        record,
        line_number,
    ):

        if not isinstance(record, dict):

            raise ValueError(
                f"Line {line_number}: "
                "record must be an object."
            )

        messages = record.get(
            "messages"
        )

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

        roles = []

        for message_index, message in enumerate(
            messages
        ):

            if not isinstance(
                message,
                dict,
            ):

                raise ValueError(
                    f"Line {line_number}, "
                    f"message {message_index}: "
                    "message must be an object."
                )

            role = message.get("role")

            content = message.get("content")

            if role not in {
                "system",
                "user",
                "assistant",
            }:

                raise ValueError(
                    f"Line {line_number}, "
                    f"message {message_index}: "
                    f"invalid role: {role}"
                )

            if not isinstance(
                content,
                str,
            ):

                raise ValueError(
                    f"Line {line_number}, "
                    f"message {message_index}: "
                    "content must be a string."
                )

            if not content.strip():

                raise ValueError(
                    f"Line {line_number}, "
                    f"message {message_index}: "
                    "content cannot be empty."
                )

            roles.append(role)

        if "user" not in roles:

            raise ValueError(
                f"Line {line_number}: "
                "conversation has no user message."
            )

        if "assistant" not in roles:

            raise ValueError(
                f"Line {line_number}: "
                "conversation has no assistant message."
            )

    # ======================================================
    # Tokenize a conversation
    # ======================================================

    def _tokenize_conversation(
        self,
        record,
    ):
        """
        Returns:

            token_ids
            loss_flags

        loss_flags has the same length as token_ids.

        1 = token contributes to assistant loss
        0 = token does not contribute to loss
        """

        token_ids = []
        loss_flags = []

        messages = record["messages"]

        for message_index, message in enumerate(
            messages
        ):

            role = message["role"]

            content = message["content"].strip()

            # ------------------------------------------------
            # Determine prefix
            # ------------------------------------------------

            if role == "system":

                prefix = SYSTEM_PREFIX

            elif role == "user":

                prefix = USER_PREFIX

            elif role == "assistant":

                prefix = ASSISTANT_PREFIX

            else:

                raise ValueError(
                    f"Unsupported role: {role}"
                )

            # ------------------------------------------------
            # Prefix tokens
            # ------------------------------------------------

            prefix_ids = self.tokenizer.encode(
                prefix,
                add_bos=False,
                add_eos=False,
            )

            token_ids.extend(prefix_ids)

            # ------------------------------------------------
            # Prefix loss
            #
            # Assistant prefix is included in training
            # because the model should learn how to enter
            # assistant mode.
            # ------------------------------------------------

            if role == "assistant":

                loss_flags.extend(
                    [1] * len(prefix_ids)
                )

            else:

                loss_flags.extend(
                    [0] * len(prefix_ids)
                )

            # ------------------------------------------------
            # Content tokens
            # ------------------------------------------------

            content_ids = self.tokenizer.encode(
                content,
                add_bos=False,
                add_eos=False,
            )

            token_ids.extend(content_ids)

            if role == "assistant":

                loss_flags.extend(
                    [1] * len(content_ids)
                )

            else:

                loss_flags.extend(
                    [0] * len(content_ids)
                )

            # ------------------------------------------------
            # Newline between messages
            # ------------------------------------------------

            newline_ids = self.tokenizer.encode(
                "\n",
                add_bos=False,
                add_eos=False,
            )

            token_ids.extend(newline_ids)

            if role == "assistant":

                loss_flags.extend(
                    [1] * len(newline_ids)
                )

            else:

                loss_flags.extend(
                    [0] * len(newline_ids)
                )

        # ----------------------------------------------------
        # Add EOS
        # ----------------------------------------------------

        eos_ids = self.tokenizer.encode(
            "",
            add_bos=False,
            add_eos=True,
        )

        token_ids.extend(eos_ids)

        # EOS is part of assistant learning.
        loss_flags.extend(
            [1] * len(eos_ids)
        )

        return token_ids, loss_flags

    # ======================================================
    # Build training samples
    # ======================================================

    def _build_samples(self):

        for record_index, record in enumerate(
            self.records
        ):

            token_ids, loss_flags = (
                self._tokenize_conversation(
                    record
                )
            )

            if len(token_ids) < 2:

                continue

            # ------------------------------------------------
            # Add BOS
            # ------------------------------------------------

            bos_ids = self.tokenizer.encode(
                "",
                add_bos=True,
                add_eos=False,
            )

            token_ids = (
                bos_ids
                + token_ids
            )

            loss_flags = (
                [0] * len(bos_ids)
                + loss_flags
            )

            # ------------------------------------------------
            # Truncate
            #
            # Keep the beginning of the conversation.
            #
            # For the current V2 dataset most examples are
            # short enough to fit into 128 tokens.
            # ------------------------------------------------

            if len(token_ids) > self.max_seq_len + 1:

                token_ids = token_ids[
                    : self.max_seq_len + 1
                ]

                loss_flags = loss_flags[
                    : self.max_seq_len + 1
                ]

            # ------------------------------------------------
            # Input / target shift
            # ------------------------------------------------

            input_ids = token_ids[:-1]

            target_ids = token_ids[1:]

            target_loss_flags = (
                loss_flags[1:]
            )

            # ------------------------------------------------
            # Padding
            # ------------------------------------------------

            current_length = len(input_ids)

            padding_length = (
                self.max_seq_len
                - current_length
            )

            if padding_length > 0:

                input_ids = (
                    input_ids
                    + [self.pad_id]
                    * padding_length
                )

                target_ids = (
                    target_ids
                    + [self.pad_id]
                    * padding_length
                )

                target_loss_flags = (
                    target_loss_flags
                    + [0]
                    * padding_length
                )

            else:

                input_ids = input_ids[
                    : self.max_seq_len
                ]

                target_ids = target_ids[
                    : self.max_seq_len
                ]

                target_loss_flags = (
                    target_loss_flags[
                        : self.max_seq_len
                    ]
                )

            # ------------------------------------------------
            # Attention mask
            # ------------------------------------------------

            attention_mask = [
                0 if token_id == self.pad_id
                else 1
                for token_id in input_ids
            ]

            # ------------------------------------------------
            # Convert to tensors
            # ------------------------------------------------

            input_tensor = torch.tensor(
                input_ids,
                dtype=torch.long,
            )

            target_tensor = torch.tensor(
                target_ids,
                dtype=torch.long,
            )

            loss_mask_tensor = torch.tensor(
                target_loss_flags,
                dtype=torch.float32,
            )

            attention_mask_tensor = torch.tensor(
                attention_mask,
                dtype=torch.long,
            )

            self.samples.append(
                {
                    "input_ids": input_tensor,
                    "target_ids": target_tensor,
                    "loss_mask": loss_mask_tensor,
                    "attention_mask": (
                        attention_mask_tensor
                    ),
                    "record_index": record_index,
                }
            )

    # ======================================================
    # Dataset API
    # ======================================================

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        if index < 0 or index >= len(
            self.samples
        ):

            raise IndexError(
                "Instruction dataset index "
                "out of range."
            )

        sample = self.samples[index]

        return {
            "input_ids": sample[
                "input_ids"
            ],

            "target_ids": sample[
                "target_ids"
            ],

            "loss_mask": sample[
                "loss_mask"
            ],

            "attention_mask": sample[
                "attention_mask"
            ],
        }

    # ======================================================
    # Statistics
    # ======================================================

    def statistics(self):

        total_tokens = 0
        assistant_tokens = 0
        padding_tokens = 0

        for sample in self.samples:

            input_ids = sample[
                "input_ids"
            ]

            loss_mask = sample[
                "loss_mask"
            ]

            attention_mask = sample[
                "attention_mask"
            ]

            total_tokens += (
                attention_mask.sum().item()
            )

            assistant_tokens += (
                loss_mask.sum().item()
            )

            padding_tokens += (
                (
                    attention_mask == 0
                ).sum().item()
            )

        return {
            "records": len(self.records),

            "samples": len(self.samples),

            "total_sequence_tokens":
                int(total_tokens),

            "assistant_loss_tokens":
                int(assistant_tokens),

            "padding_tokens":
                int(padding_tokens),

            "assistant_loss_percentage":
                (
                    assistant_tokens
                    / total_tokens
                    * 100
                    if total_tokens > 0
                    else 0
                ),

            "padding_percentage":
                (
                    padding_tokens
                    / (
                        len(self.samples)
                        * self.max_seq_len
                    )
                    * 100
                    if self.samples
                    else 0
                ),

            "max_seq_len":
                self.max_seq_len,

            "pad_id":
                self.pad_id,
        }