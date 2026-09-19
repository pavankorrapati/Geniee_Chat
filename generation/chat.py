from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat.conversation import GenieeConversation
from chat.prompt import DEFAULT_SYSTEM_PROMPT
from generation.generate import load_geniee
from python_debugger import diagnose_traceback
from rag.retriever import LocalRetriever, load_corpus_documents
from rag.web_retriever import WebRetriever


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "geniee_sft_best.pt"
)

INSTRUCTION_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

CORPUS_DIR = PROJECT_ROOT / "corpus"

WEB_CORPUS_DIR = (
    PROJECT_ROOT
    / "data"
    / "web"
    / "cleaned"
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

UNKNOWN_ANSWER = (
    "I do not have enough reliable information to answer that accurately yet."
)


_DYNAMIC_TERMS = {
    "today",
    "todays",
    "latest",
    "current",
    "currently",
    "now",
    "news",
    "weather",
    "price",
    "rate",
    "cost",
    "stock",
    "market",
    "score",
    "result",
    "results",
    "release",
    "released",
    "version",
    "update",
}


# ---------------------------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------------------------

def _question_terms(text: str) -> set[str]:
    """
    Extract meaningful words from a question.

    Stop words are removed so lexical matching focuses on
    the important terms in the user's question.
    """

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "can",
        "do",
        "does",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "why",
        "with",
        "you",
    }

    return {
        word
        for word in re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
        if word not in stop_words
    }


def _normalize_question(text: str) -> str:
    """
    Normalize a question for trained-Q&A matching.
    """

    normalized = text.casefold()

    # Remove parenthesized text.
    normalized = re.sub(
        r"\([^)]*\)",
        " ",
        normalized,
    )

    normalized = normalized.replace("-", " ")

    normalized = re.sub(
        r"[^a-z0-9\s]",
        " ",
        normalized,
    )

    replacements = {
        "sqli": "sql injection",
        "xss": "cross site scripting",
        "bruteforce": "brute force",
        "seleniumj": "selenium",
        "cheif": "chief",
    }

    for source, target in replacements.items():
        normalized = normalized.replace(
            source,
            target,
        )

    exception_aliases = {
        "why do i get a keyerror": "what is a keyerror",
        "why do i get keyerror": "what is a keyerror",
        "why do i get a nameerror": "what is a nameerror",
        "why do i get a typeerror": "what is a typeerror",
        "why do i get an indexerror": "what is an indexerror",
        "why do i get an attributeerror": "what is an attributeerror",
        "why do i get a valueerror": "what is a valueerror",
    }

    for source, target in exception_aliases.items():
        normalized = normalized.replace(
            source,
            target,
        )

    return " ".join(normalized.split())


# ---------------------------------------------------------------------------
# TRAINED INSTRUCTION DATA
# ---------------------------------------------------------------------------

def _load_instruction_answers() -> list[
    tuple[str, str, set[str]]
]:
    """
    Load instruction/Q&A records from the processed JSONL splits.

    Returns:
        [
            (
                normalized_question,
                assistant_answer,
                question_terms,
            ),
            ...
        ]
    """

    answers: list[
        tuple[str, str, set[str]]
    ] = []

    if not INSTRUCTION_SPLIT_DIR.exists():
        return answers

    for split_path in sorted(
        INSTRUCTION_SPLIT_DIR.glob("*.jsonl")
    ):
        try:
            lines = split_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except OSError:
            continue

        for line in lines:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = record.get(
                "messages",
                [],
            )

            user = next(
                (
                    message.get("content")
                    for message in messages
                    if message.get("role") == "user"
                ),
                None,
            )

            assistant = next(
                (
                    message.get("content")
                    for message in messages
                    if message.get("role") == "assistant"
                ),
                None,
            )

            if (
                isinstance(user, str)
                and isinstance(assistant, str)
            ):
                user = user.strip()
                assistant = assistant.strip()

                if user and assistant:
                    answers.append(
                        (
                            _normalize_question(user),
                            assistant,
                            _question_terms(user),
                        )
                    )

    return answers


# ---------------------------------------------------------------------------
# LOCAL CORPUS
# ---------------------------------------------------------------------------

def _load_local_documents() -> list[
    tuple[str, str]
]:
    """
    Load the local Geniee corpus and previously downloaded
    web documents.

    Important:
        These web documents are local files.
        They are NOT live Internet requests.
    """

    documents: list[
        tuple[str, str]
    ] = []

    if CORPUS_DIR.exists():
        documents.extend(
            load_corpus_documents(
                CORPUS_DIR
            )
        )

    if WEB_CORPUS_DIR.exists():
        documents.extend(
            (
                f"downloaded_web/{name}",
                text,
            )
            for name, text in load_corpus_documents(
                WEB_CORPUS_DIR
            )
        )

    return documents


# ---------------------------------------------------------------------------
# GENIEE CHAT
# ---------------------------------------------------------------------------

class GenieeChat:
    """
    Main Geniee orchestration class.

    Response priority:

        1. Python traceback debugger
        2. Exact / strong trained-Q&A match
        3. Local retrieval / RAG
        4. Live WebRetriever when required and enabled
        5. Local 27M model fallback

    Exposed state:

        last_source
        last_sources
        last_usage

    These fields are consumed by the FastAPI layer and UI.
    """

    def __init__(
        self,
        generator,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        knowledge_base=None,
        retriever=None,
        web_retriever=None,
    ):
        self.generator = generator

        self.conversation = GenieeConversation(
            system_prompt=system_prompt,
            max_turns=6,
        )

        self.knowledge_base = (
            knowledge_base
            if knowledge_base is not None
            else _load_instruction_answers()
        )

        self.retriever = (
            retriever
            if retriever is not None
            else LocalRetriever(
                _load_local_documents()
            )
        )

        self.web_retriever = (
            web_retriever
            if web_retriever is not None
            else WebRetriever()
        )

        self.last_source = "model"

        self.last_sources: list[str] = []

        self.last_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "context_limit": self.context_limit,
        }

    # -----------------------------------------------------------------------
    # MODEL CONTEXT
    # -----------------------------------------------------------------------

    @property
    def context_limit(self) -> int:
        return int(
            self.generator.model.config.max_seq_len
        )

    # -----------------------------------------------------------------------
    # CONVERSATION RESTORE
    # -----------------------------------------------------------------------

    def restore_messages(
        self,
        messages: list[dict],
    ) -> None:
        """
        Restore user/assistant messages persisted by ChatHistoryStore.

        Conversation memory remains bounded by max_turns.
        """

        self.conversation.clear()

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if (
                not isinstance(content, str)
                or not content.strip()
            ):
                continue

            if role == "user":
                self.conversation.add_user(
                    content
                )

            elif role == "assistant":
                self.conversation.add_assistant(
                    content
                )

    # -----------------------------------------------------------------------
    # TOKEN COUNTING
    # -----------------------------------------------------------------------

    def _count_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count tokens using the actual Geniee tokenizer.
        """

        if not text:
            return 0

        return len(
            self.generator.tokenizer.encode(
                text,
                add_bos=True,
                add_eos=False,
            )
        )

    def _set_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """
        Store token usage for the API/UI.
        """

        self.last_usage = {
            "prompt_tokens": int(
                prompt_tokens
            ),
            "completion_tokens": int(
                completion_tokens
            ),
            "total_tokens": int(
                prompt_tokens
                + completion_tokens
            ),
            "context_limit": self.context_limit,
        }

    def _usage_for_direct_answer(
        self,
        answer: str,
    ) -> None:
        self._set_usage(
            self._count_tokens(
                self.conversation.prompt()
            ),
            self._count_tokens(answer),
        )

    # -----------------------------------------------------------------------
    # TRAINED Q&A RETRIEVAL
    # -----------------------------------------------------------------------

    def _retrieve_answer(
        self,
        user_text: str,
    ) -> str | None:
        """
        Find an answer from the processed instruction/Q&A dataset.
        """

        normalized_query = _normalize_question(
            user_text
        )

        query_terms = _question_terms(
            user_text
        )

        if not query_terms:
            return None

        # Exact normalized match.
        exact = next(
            (
                answer
                for question, answer, _
                in self.knowledge_base
                if question == normalized_query
            ),
            None,
        )

        if exact:
            return exact

        # Conservative lexical similarity.
        best_answer = None
        best_score = 0.0

        for (
            _,
            answer,
            question_terms,
        ) in self.knowledge_base:

            if not question_terms:
                continue

            overlap = len(
                query_terms
                & question_terms
            )

            if not overlap:
                continue

            score = (
                overlap
                / len(
                    query_terms
                    | question_terms
                )
            )

            if score > best_score:
                best_score = score
                best_answer = answer

        return (
            best_answer
            if best_score >= 0.50
            else None
        )

    # -----------------------------------------------------------------------
    # LOCAL RETRIEVAL RELEVANCE
    # -----------------------------------------------------------------------

    def _evaluate_local_relevance(
        self,
        user_text: str,
        references: list[
            tuple[str, str]
        ],
    ) -> float:
        if not references:
            return 0.0

        query_terms = _question_terms(
            user_text
        )

        if not query_terms:
            return 0.0

        reference_terms = _question_terms(
            " ".join(
                text
                for _, text in references
            )
        )

        return (
            len(
                query_terms
                & reference_terms
            )
            / len(query_terms)
        )

    # -----------------------------------------------------------------------
    # WEB REQUIREMENT
    # -----------------------------------------------------------------------

    def _requires_external_context(
        self,
        user_text: str,
    ) -> bool:
        """
        Detect questions that are likely to require current information.
        """

        tokens = set(
            re.findall(
                r"[a-z0-9]+",
                user_text.lower(),
            )
        )

        return bool(
            tokens & _DYNAMIC_TERMS
        )

    # -----------------------------------------------------------------------
    # RESPONSE FORMATTING
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_answer(
        text: str,
    ) -> str:
        """
        Preserve useful whitespace/newlines while removing
        accidental excessive whitespace.
        """

        text = (
            text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text

    @staticmethod
    def clean_response(
        text: str,
    ) -> str:
        """
        Remove internal Geniee control markers from generated text.
        """

        if not text:
            return ""

        for marker in (
            "<|user|>",
            "<|system|>",
            "<|assistant|>",
        ):
            if marker in text:
                text = text.split(
                    marker,
                    1,
                )[0]

        text = (
            text
            .replace(
                "<|endoftext|>",
                "",
            )
            .replace(
                "⁇",
                "",
            )
            .strip()
        )

        return GenieeChat._format_answer(
            text
        )

    # -----------------------------------------------------------------------
    # GENERATION QUALITY CHECK
    # -----------------------------------------------------------------------

    @staticmethod
    def _looks_unreliable(
        text: str,
        question: str = "",
    ) -> bool:
        """
        Detect obvious low-quality output from the small local model.
        """

        words = re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

        if len(words) < 5:
            return True

        unique_ratio = (
            len(set(words))
            / max(1, len(words))
        )

        if unique_ratio < 0.35:
            return True

        # Known signs of corrupted tiny-model generation.
        if any(
            fragment in text.lower()
            for fragment in (
                "cannesses",
                "runcted",
                "destin the expected",
            )
        ):
            return True

        question_terms = _question_terms(
            question
        )

        response_terms = _question_terms(
            text
        )

        return (
            bool(question_terms)
            and not bool(
                question_terms
                & response_terms
            )
        )

    # -----------------------------------------------------------------------
    # PROMPT BUILDING
    # -----------------------------------------------------------------------

    def _build_prompt(
        self,
        references: list[
            tuple[str, str]
        ] | None = None,
    ) -> str:
        """
        Build the model prompt with optional local references.
        """

        prompt = self.conversation.prompt()

        if not references:
            return prompt

        # Keep reference injection deliberately small because
        # the current Geniee checkpoint has a 256-token context window.
        blocks = []

        for name, text in references[:2]:
            cleaned = " ".join(
                text.split()
            )

            blocks.append(
                f"[{name}] {cleaned[:500]}"
            )

        reference_text = (
            "\n\nRelevant trusted local reference:\n"
            + "\n".join(blocks)
            + "\n\nUse the reference only when it answers "
              "the user's question."
        )

        marker = "<|assistant|>\n"

        if marker in prompt:
            head, tail = prompt.rsplit(
                marker,
                1,
            )

            return (
                head
                + reference_text
                + "\n"
                + marker
                + tail
            )

        return (
            prompt
            + reference_text
        )

    # -----------------------------------------------------------------------
    # CONTEXT FITTING
    # -----------------------------------------------------------------------

    def _fit_prompt_to_context(
        self,
        max_new_tokens: int,
        references: list[
            tuple[str, str]
        ] | None = None,
    ) -> tuple[str, int, int]:
        """
        Fit conversation + retrieval context + requested generation
        inside the actual model context window.

        Priority:

            1. Keep requested output length if possible.
            2. Remove old conversation turns.
            3. Reduce retrieval references.
            4. Reduce reference text.
            5. Reduce generation length only as the final fallback.
        """

        desired_new_tokens = max(
            1,
            int(max_new_tokens),
        )

        # Never request more than the context window can possibly support.
        desired_new_tokens = min(
            desired_new_tokens,
            max(
                1,
                self.context_limit - 8,
            ),
        )

        working_refs = list(
            references or []
        )

        while True:

            prompt = self._build_prompt(
                working_refs
            )

            prompt_tokens = self._count_tokens(
                prompt
            )

            # Normal successful case.
            if (
                prompt_tokens
                + desired_new_tokens
                <= self.context_limit
            ):
                return (
                    prompt,
                    prompt_tokens,
                    desired_new_tokens,
                )

            messages = (
                self.conversation
                .memory
                .messages
            )

            # ---------------------------------------------------------------
            # Remove oldest completed turn first.
            # ---------------------------------------------------------------
            if len(messages) > 2:
                del messages[:2]
                continue

            # ---------------------------------------------------------------
            # Then remove additional retrieval references.
            # ---------------------------------------------------------------
            if len(working_refs) > 1:
                working_refs.pop()
                continue

            # ---------------------------------------------------------------
            # Then shrink the remaining reference.
            # ---------------------------------------------------------------
            if working_refs:
                name, text = working_refs[0]

                if len(text) > 180:
                    working_refs[0] = (
                        name,
                        text[
                            :max(
                                180,
                                len(text) // 2,
                            )
                        ],
                    )
                    continue

                working_refs.clear()
                continue

            # ---------------------------------------------------------------
            # Final fallback: reduce generation length.
            # ---------------------------------------------------------------
            available = (
                self.context_limit
                - prompt_tokens
            )

            if available <= 0:
                # The latest user turn + system prompt alone
                # exceeded the model context.
                latest_user = next(
                    (
                        msg["content"]
                        for msg in reversed(messages)
                        if msg.get("role") == "user"
                    ),
                    "",
                )

                self.conversation.clear()

                if latest_user:
                    self.conversation.add_user(
                        latest_user
                    )

                prompt = self._build_prompt()

                prompt_tokens = self._count_tokens(
                    prompt
                )

                available = max(
                    1,
                    self.context_limit
                    - prompt_tokens,
                )

            return (
                prompt,
                prompt_tokens,
                max(
                    1,
                    min(
                        desired_new_tokens,
                        available,
                    ),
                ),
            )

    # -----------------------------------------------------------------------
    # LIVE WEB ANSWER
    # -----------------------------------------------------------------------

    def _web_answer(
        self,
        user_text: str,
    ) -> tuple[
        str | None,
        list[str],
    ]:
        """
        Retrieve live web evidence.

        The 27M model is currently too small to reliably summarize
        arbitrary live pages, so web results are returned extractively
        instead of asking the model to rewrite them.

        This prevents garbled web answers.
        """

        try:
            evidence = (
                self.web_retriever.retrieve(
                    user_text
                )
            )
        except Exception:
            evidence = []

        if not evidence:
            return None, []

        snippets: list[str] = []
        urls: list[str] = []
        seen_snippets: set[str] = set()

        for sentence, url in evidence:

            sentence = self._format_answer(
                sentence
            )

            key = sentence.casefold()

            if (
                sentence
                and key not in seen_snippets
            ):
                seen_snippets.add(key)
                snippets.append(sentence)

            if (
                url
                and url not in urls
            ):
                urls.append(url)

            if len(snippets) >= 2:
                break

        if not snippets:
            return None, urls

        return (
            " ".join(snippets),
            urls,
        )

    # -----------------------------------------------------------------------
    # MAIN RESPONSE
    # -----------------------------------------------------------------------

    def respond(
        self,
        user_text: str,
        max_new_tokens: int = 100,
        enable_web_search: bool = True,
    ) -> str:
        """
        Generate a Geniee response.

        Parameters
        ----------
        user_text:
            User's message.

        max_new_tokens:
            Maximum number of output tokens requested by the caller/UI.

        enable_web_search:
            Whether live web retrieval is allowed.

        Important:
            max_new_tokens is NOT hardcoded internally.
            The value flows from the API/UI into the model generation call.
        """

        user_text = user_text.strip()

        if not user_text:
            return ""

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than 0"
            )

        # Reset response metadata.
        self.last_source = "model"
        self.last_sources = []

        self._set_usage(
            0,
            0,
        )

        # Add user message to conversation.
        self.conversation.add_user(
            user_text
        )

        # ===================================================================
        # 1. PYTHON TRACEBACK / EXECUTION DEBUGGER
        # ===================================================================

        traceback_answer = diagnose_traceback(
            user_text
        )

        if traceback_answer:
            traceback_answer = (
                self._format_answer(
                    traceback_answer
                )
            )

            self.last_source = "debugger"

            self._usage_for_direct_answer(
                traceback_answer
            )

            self.conversation.add_assistant(
                traceback_answer
            )

            return traceback_answer

        # ===================================================================
        # 2. STRONG MATCH FROM SFT / TRAINING Q&A
        # ===================================================================

        trained_answer = self._retrieve_answer(
            user_text
        )

        if trained_answer:
            trained_answer = (
                self._format_answer(
                    trained_answer
                )
            )

            self.last_source = "trained_qa"

            self._usage_for_direct_answer(
                trained_answer
            )

            self.conversation.add_assistant(
                trained_answer
            )

            return trained_answer

        # ===================================================================
        # 3. LOCAL CORPUS RETRIEVAL
        # ===================================================================

        references = self.retriever.retrieve(
            user_text,
            limit=2,
        )

        local_relevance = (
            self._evaluate_local_relevance(
                user_text,
                references,
            )
        )

        # Only inject local references when enough
        # of the question is covered.
        useful_references = (
            references
            if local_relevance >= 0.45
            else []
        )

        # ===================================================================
        # 4. LIVE WEB RETRIEVAL
        # ===================================================================

        needs_web = (
            self._requires_external_context(
                user_text
            )
            or not useful_references
        )

        if (
            enable_web_search
            and needs_web
        ):
            web_answer, urls = (
                self._web_answer(
                    user_text
                )
            )

            if web_answer:
                self.last_source = "web"

                self.last_sources = urls

                self._usage_for_direct_answer(
                    web_answer
                )

                self.conversation.add_assistant(
                    web_answer
                )

                return web_answer

        # ===================================================================
        # 5. LOCAL 27M MODEL + OPTIONAL LOCAL RAG
        # ===================================================================

        if useful_references:
            self.last_source = "local_rag"

            self.last_sources = [
                name
                for name, _ in useful_references
            ]

        # -------------------------------------------------------------------
        # IMPORTANT:
        #
        # max_new_tokens received from the API/UI enters this function.
        #
        # _fit_prompt_to_context() calculates the actual allowed number
        # after accounting for the 256-token model context window.
        # -------------------------------------------------------------------

        (
            prompt,
            prompt_tokens,
            allowed_new_tokens,
        ) = self._fit_prompt_to_context(
            max_new_tokens=max_new_tokens,
            references=useful_references,
        )

        try:
            raw = self.generator.generate(
                prompt,
                max_new_tokens=allowed_new_tokens,
                temperature=0.7,
                top_k=30,
                top_p=0.90,
                repetition_penalty=1.05,
                do_sample=False,
                return_full_text=False,
            )

        except (
            OSError,
            RuntimeError,
            ValueError,
        ):
            # Remove the latest user message if generation fails.
            if self.conversation.memory.messages:
                self.conversation.memory.messages.pop()

            raise

        response = self.clean_response(
            raw
        )

        completion_tokens = (
            self._count_tokens(
                response
            )
        )

        self._set_usage(
            prompt_tokens,
            completion_tokens,
        )

        # ===================================================================
        # 6. RELIABILITY FALLBACK
        # ===================================================================

        if self._looks_unreliable(
            response,
            user_text,
        ):

            # ---------------------------------------------------------------
            # LOCAL EVIDENCE FALLBACK
            # ---------------------------------------------------------------

            if useful_references:

                evidence = (
                    self.retriever.best_sentences(
                        user_text,
                        limit=2,
                    )
                )

                if evidence:
                    response = (
                        self._format_answer(
                            " ".join(evidence)
                        )
                    )

                    self.last_source = (
                        "local_evidence"
                    )

                    self.last_sources = [
                        name
                        for name, _
                        in useful_references
                    ]

                else:
                    response = UNKNOWN_ANSWER

                    self.last_source = (
                        "model_fallback"
                    )

            # ---------------------------------------------------------------
            # WEB FALLBACK
            # ---------------------------------------------------------------

            elif enable_web_search:

                web_answer, urls = (
                    self._web_answer(
                        user_text
                    )
                )

                if web_answer:
                    response = web_answer

                    self.last_source = "web"

                    self.last_sources = urls

                else:
                    response = UNKNOWN_ANSWER

                    self.last_source = (
                        "model_fallback"
                    )

            # ---------------------------------------------------------------
            # FINAL UNKNOWN ANSWER
            # ---------------------------------------------------------------

            else:
                response = UNKNOWN_ANSWER

                self.last_source = (
                    "model_fallback"
                )

            self._set_usage(
                prompt_tokens,
                self._count_tokens(
                    response
                ),
            )

        # Save final assistant response.
        self.conversation.add_assistant(
            response
        )

        return response

    # -----------------------------------------------------------------------
    # TERMINAL CHAT
    # -----------------------------------------------------------------------

    def start(self) -> None:
        print("\n" + "=" * 70)
        print("GENIEE CHAT")
        print("=" * 70)

        print(
            "Commands: exit | quit | clear"
        )

        print(
            "Model: Geniee 27M + Local RAG + Web Retrieval"
        )

        while True:

            try:
                user_input = input(
                    "\nYou: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in {
                "exit",
                "quit",
            }:
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                self.conversation.clear()
                print(
                    "Conversation cleared."
                )
                continue

            try:
                response = self.respond(
                    user_input,
                    enable_web_search=True,
                )

                print(
                    f"\nGeniee: {response}"
                )

                print(
                    f"Source: {self.last_source}"
                )

                if self.last_sources:
                    print("Sources:")

                    for source in self.last_sources:
                        print(
                            f"  - {source}"
                        )

                print(
                    f"Usage: {self.last_usage}"
                )

            except Exception as exc:
                print(
                    f"\nGeniee error: {exc}"
                )


# ---------------------------------------------------------------------------
# CHATBOT LOADER
# ---------------------------------------------------------------------------

def load_chatbot() -> GenieeChat:
    """
    Load the current Geniee SFT checkpoint.
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found: "
            f"{CHECKPOINT_PATH}\n"
            "Run training/train_pretrain.py first, "
            "then training/train_sft.py."
        )

    return GenieeChat(
        generator=load_geniee(
            CHECKPOINT_PATH
        ),
        web_retriever=WebRetriever(),
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    chatbot = load_chatbot()
    chatbot.start()


if __name__ == "__main__":
    main()