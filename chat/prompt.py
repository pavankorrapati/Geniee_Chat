# from __future__ import annotations

# SYSTEM_PREFIX = "<|system|>\\n"
# USER_PREFIX = "<|user|>\\n"
# ASSISTANT_PREFIX = "<|assistant|>\\n"

# DEFAULT_SYSTEM_PROMPT = (
#     "You are Geniee, a helpful AI assistant. "
#     "Answer clearly, accurately, and directly. "
#     "If the question is unclear, say what is missing instead of inventing facts."
# )


# def build_prompt(system_message: str, messages: list[dict]) -> str:
#     """Build the exact role format used during SFT."""
#     parts = [SYSTEM_PREFIX, system_message.strip(), "\\n"]
#     for message in messages:
#         role = message["role"]
#         content = message["content"].strip()
#         if role == "user":
#             parts.extend([USER_PREFIX, content, "\\n"])
#         elif role == "assistant":
#             parts.extend([ASSISTANT_PREFIX, content, "\\n"])
#         else:
#             raise ValueError(f"Unsupported chat role: {role}")
#     parts.append(ASSISTANT_PREFIX)
#     return "".join(parts)


# def build_single_turn_prompt(user_message: str, system_message: str = DEFAULT_SYSTEM_PROMPT) -> str:
#     return build_prompt(system_message, [{"role": "user", "content": user_message}])


# from future import annotations
from __future__ import annotations

# ==========================================================

# GENIEE CHAT PROMPT FORMAT

# ==========================================================

#

# IMPORTANT:

# These prefixes MUST match the prefixes used by

# data/instruction_dataset.py during SFT training.

#

# The "\n" below is a REAL newline character in Python.

#

# Training format:

#

# <|system|>

# system message

# <|user|>

# user message

# <|assistant|>

#

# ==========================================================

SYSTEM_PREFIX = "<|system|>\n"

USER_PREFIX = "<|user|>\n"

ASSISTANT_PREFIX = "<|assistant|>\n"

# ==========================================================

# DEFAULT SYSTEM PROMPT

# ==========================================================

DEFAULT_SYSTEM_PROMPT = (
"You are Geniee, a helpful AI assistant. "
"Answer clearly, accurately, and directly. "
"If the question is unclear, say what is missing "
"instead of inventing facts."
)

# ==========================================================

# BUILD PROMPT

# ==========================================================

def build_prompt(
    system_message: str,
    messages: list[dict],
    ) -> str:
    """
    Build a Geniee conversation prompt.

    This function MUST use the same format that was used
    during supervised fine-tuning.

    Example input:

        system_message = "You are Geniee."

        messages = [
            {
                "role": "user",
                "content": "What is Selenium?"
            }
        ]

    Result:

        <|system|>
        You are Geniee.
        <|user|>
        What is Selenium?
        <|assistant|>

    The final <|assistant|> prefix tells the model that it
    should generate the assistant response.
    """

    if not isinstance(
        system_message,
        str,
    ):
        raise TypeError(
            "system_message must be a string."
        )

    if not isinstance(
        messages,
        list,
    ):
        raise TypeError(
            "messages must be a list."
        )

    # ------------------------------------------------------
    # System message
    # ------------------------------------------------------

    parts = [
        SYSTEM_PREFIX,
        system_message.strip(),
        "\n",
    ]

    # ------------------------------------------------------
    # Conversation messages
    # ------------------------------------------------------

    for message in messages:

        if not isinstance(
            message,
            dict,
        ):
            raise TypeError(
                "Each message must be a dictionary."
            )

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )

        if not isinstance(
            role,
            str,
        ):
            raise ValueError(
                "Message role must be a string."
            )

        if not isinstance(
            content,
            str,
        ):
            raise ValueError(
                "Message content must be a string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "Message content cannot be empty."
            )

        # --------------------------------------------------
        # User message
        # --------------------------------------------------

        if role == "user":

            parts.extend(
                [
                    USER_PREFIX,
                    content,
                    "\n",
                ]
            )

        # --------------------------------------------------
        # Assistant message
        # --------------------------------------------------

        elif role == "assistant":

            parts.extend(
                [
                    ASSISTANT_PREFIX,
                    content,
                    "\n",
                ]
            )

        # --------------------------------------------------
        # Unsupported role
        # --------------------------------------------------

        else:

            raise ValueError(
                f"Unsupported chat role: {role}"
            )

    # ------------------------------------------------------
    # Final assistant prefix
    #
    # Generation starts after this prefix.
    # ------------------------------------------------------

    parts.append(
        ASSISTANT_PREFIX
    )

    return "".join(parts)

# ==========================================================

# SINGLE-TURN PROMPT

# ==========================================================

def build_single_turn_prompt(
    user_message: str,
    system_message: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
    """
    Build a prompt for a single user question.

    ```
    Example:

        prompt = build_single_turn_prompt(
            "What is Selenium?"
        )

    Result:

        <|system|>
        You are Geniee...
        <|user|>
        What is Selenium?
        <|assistant|>
    """

    if not isinstance(
        user_message,
        str,
    ):
        raise TypeError(
            "user_message must be a string."
        )

    return build_prompt(
        system_message,
        [
            {
                "role": "user",
                "content": user_message,
            }
        ],
    )

# ==========================================================

# PROMPT VALIDATION

# ==========================================================

def validate_prompt_format(
    prompt: str,
    ) -> bool:
    """
    Validate that a generated prompt contains the expected
    Geniee role structure.

    ```
    Returns:
        True  -> valid prompt
        False -> invalid prompt
    """

    if not isinstance(
        prompt,
        str,
    ):
        return False

    required_prefixes = [
        SYSTEM_PREFIX,
        USER_PREFIX,
        ASSISTANT_PREFIX,
    ]

    for prefix in required_prefixes:

        if prefix not in prompt:

            return False

    # The prompt used for generation should finish at
    # the assistant prefix.
    if not prompt.endswith(
        ASSISTANT_PREFIX
    ):
        return False

    return True

# ==========================================================

# DEBUG / SELF TEST

# ==========================================================

if __name__ == "__main__":
    
    test_prompt = build_single_turn_prompt(
        "What is Selenium?"
    )

    print("=" * 70)
    print("GENIEE PROMPT FORMAT TEST")
    print("=" * 70)

    print()
    print("Generated prompt:")
    print("-" * 70)
    print(test_prompt)
    print("-" * 70)

    print()
    print(
        "Contains real newline characters:",
        "\n" in test_prompt,
    )

    print(
        "Contains literal \\\\n sequence:",
        "\\n" in test_prompt,
    )

    print(
        "Prompt format valid:",
        validate_prompt_format(test_prompt),
    )

    print("=" * 70)
