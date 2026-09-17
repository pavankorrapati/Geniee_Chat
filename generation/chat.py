# from pathlib import Path
# import sys
# import json
# import re

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from chat.conversation import GenieeConversation
# from chat.prompt import DEFAULT_SYSTEM_PROMPT
# from generation.generate import load_geniee
# from python_debugger import diagnose_traceback
# from rag.retriever import LocalRetriever, load_corpus_documents
# from rag.web_retriever import WebRetriever

# CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
# INSTRUCTION_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
# CORPUS_DIR = PROJECT_ROOT / "corpus"
# WEB_CORPUS_DIR = PROJECT_ROOT / "data" / "web" / "cleaned"
# UNKNOWN_ANSWER = (
#     "I do not have enough trained information to answer that accurately yet. "
#     "Please add a trusted example or document for this topic."
# )


# def _question_terms(text):
#     stop_words = {
#         "a", "an", "and", "are", "can", "do", "does", "how", "in",
#         "is", "it", "of", "on", "or", "the", "to", "what", "when",
#         "why", "with", "describe", "explain", "tell", "give",
#     }
#     normalized = _normalize_question(text)
#     return {
#         word
#         for word in re.findall(r"[a-z0-9]+", normalized.lower())
#         if word not in stop_words
#     }


# def _normalize_question(text):
#     normalized = text.casefold()
#     normalized = re.sub(r"\([^)]*\)", " ", normalized)
#     normalized = normalized.replace("-", " ")
#     normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
#     normalized = re.sub(r"\bsqli\b", "sql injection", normalized)
#     normalized = re.sub(r"\bxss\b", "cross site scripting", normalized)
#     exception_aliases = {
#         "why do i get a keyerror": "what is a keyerror",
#         "why do i get keyerror": "what is a keyerror",
#         "why do i get a nameerror": "what is a nameerror",
#         "why do i get a typeerror": "what is a typeerror",
#         "why do i get an indexerror": "what is an indexerror",
#         "why do i get an attributeerror": "what is an attributeerror",
#         "why do i get a valueerror": "what is a valueerror",
#     }
#     for source, target in exception_aliases.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(r"\bbruteforce\b|\bbrute force\b", "brute force", normalized)
#     normalized = normalized.replace("cross-site", "cross site")
#     normalized = normalized.replace("vulnerability scanning", "vulnerability scan")
#     normalized = normalized.replace("penetration testing", "penetration test")
#     for source, target in {
#         "zjanasena": "janasena",
#         "janasena": "jana sena",
#         "cheif": "chief",
#         "andhras": "andhra",
#         "seleniumj": "selenium",
#         "as per": "",
#         "in andhra pradesh": "",
#         "largest film budget": "largest telugu film budget",
#     }.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(
#         r"^what is brute force attack$|^what is brute force attack in security testing$",
#         "what is a brute force attack in security testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is sql injection and how do you test for it$",
#         "what is sql injection testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is cross site scripting cross site scripting and what are its main types$",
#         "what are the main types of cross site scripting",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^how do you approach testing api security$",
#         "how do you approach testing api security",
#         normalized,
#     )
#     for exception_name in (
#         "modulenotfounderror", "importerror", "syntaxerror", "indentationerror",
#         "taberror", "nameerror", "typeerror", "valueerror", "indexerror",
#         "keyerror", "attributeerror", "unboundlocalerror", "zerodivisionerror",
#         "filenotfounderror", "permissionerror",
#     ):
#         if exception_name in normalized and normalized != f"what is a {exception_name}":
#             article = "an" if exception_name[0] in "aeiou" else "a"
#             normalized = f"what is {article} {exception_name}"
#             break
#     return " ".join(normalized.split())


# def _load_instruction_answers():
#     answers = []
#     for split_path in sorted(INSTRUCTION_SPLIT_DIR.glob("*.jsonl")):
#         for line in split_path.read_text(encoding="utf-8").splitlines():
#             if not line.strip():
#                 continue
#             record = json.loads(line)
#             messages = record.get("messages", [])
#             user = next((m["content"] for m in messages if m.get("role") == "user"), None)
#             assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
#             if user and assistant:
#                 answers.append((user.strip(), assistant.strip(), _question_terms(user)))
#     return answers


# def _load_local_documents():
#     documents = load_corpus_documents(CORPUS_DIR)
#     if WEB_CORPUS_DIR.exists():
#         documents.extend(
#             (
#                 f"web/{name}",
#                 text,
#             )
#             for name, text in load_corpus_documents(WEB_CORPUS_DIR)
#         )
#     return documents


# class GenieeChat:
#     """Interactive Geniee chat using the same format used during SFT."""

#     def __init__(
#         self,
#         generator,
#         system_prompt=DEFAULT_SYSTEM_PROMPT,
#         knowledge_base=None,
#         retriever=None,
#         web_retriever=None,
#     ):
#         self.generator = generator
#         self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)
#         self.knowledge_base = knowledge_base if knowledge_base is not None else _load_instruction_answers()
#         self.retriever = retriever if retriever is not None else LocalRetriever(
#             _load_local_documents()
#         )
#         self.web_retriever = web_retriever if web_retriever is not None else WebRetriever()
#         self.last_source = "local"
#         self.last_sources: list[str] = []

#     def _retrieve_answer(self, user_text):
#         query = _normalize_question(user_text.strip())
#         query_terms = _question_terms(query)
#         if not query_terms:
#             return None

#         exact_match = next(
#             (answer for question, answer, _ in self.knowledge_base if _normalize_question(question) == query),
#             None,
#         )
#         if exact_match:
#             return exact_match

#         best_answer = None
#         best_score = 0.0
#         for _, answer, question_terms in self.knowledge_base:
#             if not question_terms:
#                 continue
#             overlap = len(query_terms & question_terms)
#             score = overlap / len(query_terms | question_terms)
#             query_coverage = overlap / len(query_terms)
#             if query_coverage >= 0.8 and score > best_score:
#                 best_score = score
#                 best_answer = answer

#         return best_answer if best_score >= 0.65 else None

#     def _fit_prompt_to_context(self, max_new_tokens):
#         """Drop oldest turns until prompt + generation fit the context window."""
#         tokenizer = self.generator.tokenizer
#         max_context = self.generator.model.config.max_seq_len
#         while True:
#             prompt = self.conversation.prompt()
#             token_count = len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
#             if token_count + max_new_tokens <= max_context:
#                 return prompt
#             messages = self.conversation.memory.messages
#             if len(messages) <= 2:
#                 return prompt
#             # Remove the oldest complete user/assistant pair.
#             del messages[:2]

#     @staticmethod
#     def clean_response(text):
#         if not text:
#             return "I am sorry, I could not generate a response."
#         for marker in ("<|user|>", "<|system|>", "<|assistant|>"):
#             if marker in text:
#                 text = text.split(marker, 1)[0]
#         text = text.replace("<|endoftext|>", "").replace("⁇", "")
#         return text.strip() or "I am sorry, I could not generate a response."

#     @staticmethod
#     def _looks_unreliable(text, question=""):
#         words = text.split()
#         if len(words) < 5 or len(set(words)) < max(3, len(words) // 3):
#             return True
#         question_terms = _question_terms(question)
#         response_terms = _question_terms(text)
#         return bool(question_terms) and not question_terms.intersection(response_terms)

#     def respond(self, user_text, max_new_tokens=80):
#         user_text = user_text.strip()
#         if not user_text:
#             return ""

#         if max_new_tokens <= 0:
#             raise ValueError("max_new_tokens must be greater than 0")

#         self.conversation.add_user(user_text)
#         self.last_source = "local"
#         self.last_sources = []

#         traceback_answer = diagnose_traceback(user_text)
#         if traceback_answer:
#             self.conversation.add_assistant(traceback_answer)
#             return traceback_answer

#         retrieved = self._retrieve_answer(user_text)
#         if retrieved:
#             self.conversation.add_assistant(retrieved)
#             return retrieved

#         if not self.retriever.retrieve(user_text):
#             try:
#                 web_evidence = self.web_retriever.retrieve(user_text)
#             except (OSError, ValueError, KeyError, TypeError):
#                 web_evidence = []
#             if web_evidence:
#                 response = " ".join(sentence for sentence, _ in web_evidence)
#                 self.last_source = "web"
#                 self.last_sources = [url for _, url in web_evidence]
#                 self.conversation.add_assistant(response)
#                 return response

#         prompt = self._fit_prompt_to_context(max_new_tokens)
#         references = self.retriever.retrieve(user_text)
#         if references:
#             reference_text = "\n\nRelevant local reference:\n" + "\n\n".join(
#                 f"[{name}] {text[:1200]}" for name, text in references
#             )
#             prompt_head, assistant_marker = prompt.rsplit("<|assistant|>\n", 1)
#             prompt = prompt_head + reference_text + "\n<|assistant|>\n" + assistant_marker

#         try:
#             raw = self.generator.generate(
#                 prompt,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_k=30,
#                 top_p=0.90,
#                 repetition_penalty=1.05,
#                 do_sample=False,
#                 return_full_text=False,
#             )
#         except (OSError, RuntimeError, ValueError):
#             self.conversation.memory.messages.pop()
#             raise

#         response = self.clean_response(raw)
#         if references and self._looks_unreliable(response, user_text):
#             evidence = self.retriever.best_sentences(user_text)
#             if evidence:
#                 response = " ".join(evidence)
#         elif not references and self._looks_unreliable(response, user_text):
#             try:
#                 web_evidence = self.web_retriever.retrieve(user_text)
#             except Exception:
#                 web_evidence = []
#             if web_evidence:
#                 response = " ".join(sentence for sentence, _ in web_evidence)
#                 self.last_source = "web"
#                 self.last_sources = [url for _, url in web_evidence]
#             else:
#                 response = UNKNOWN_ANSWER
#         self.conversation.add_assistant(response)
#         return response

#     def start(self):
#         print("\n" + "=" * 70)
#         print("GENIEE CHAT")
#         print("=" * 70)
#         print("Commands: exit | quit | clear")
#         print("Model: SFT instruction-tuned Geniee")

#         while True:
#             try:
#                 user_input = input("\nYou: ").strip()
#             except (KeyboardInterrupt, EOFError):
#                 print("\nGoodbye!")
#                 break

#             if not user_input:
#                 continue
#             if user_input.lower() in {"exit", "quit"}:
#                 print("Goodbye!")
#                 break
#             if user_input.lower() == "clear":
#                 self.conversation.clear()
#                 print("Conversation cleared.")
#                 continue

#             try:
#                 response = self.respond(user_input)
#                 label = "Geniee (web-grounded)" if self.last_source == "web" else "Geniee"
#                 print(f"\n{label}: {response}")
#             except Exception as exc:
#                 print(f"\nGeneration error: {exc}")


# def load_chatbot():
#     if not CHECKPOINT_PATH.exists():
#         raise FileNotFoundError(
#             f"SFT checkpoint not found: {CHECKPOINT_PATH}\n"
#             "Run training/train_pretrain.py first, then training/train_sft.py."
#         )
#     return GenieeChat(load_geniee(CHECKPOINT_PATH))


# def main():
#     load_chatbot().start()


# if __name__ == "__main__":
#     main()
# from pathlib import Path
# import sys
# import json
# import re

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from chat.conversation import GenieeConversation
# from chat.prompt import DEFAULT_SYSTEM_PROMPT
# from generation.generate import load_geniee
# from python_debugger import diagnose_traceback
# from rag.retriever import LocalRetriever, load_corpus_documents
# from rag.web_retriever import WebRetriever

# CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
# INSTRUCTION_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
# CORPUS_DIR = PROJECT_ROOT / "corpus"
# WEB_CORPUS_DIR = PROJECT_ROOT / "data" / "web" / "cleaned"
# UNKNOWN_ANSWER = (
#     "I do not have enough trained information to answer that accurately yet. "
#     "Please add a trusted example or document for this topic."
# )


# def _question_terms(text):
#     stop_words = {
#         "a", "an", "and", "are", "can", "do", "does", "how", "in",
#         "is", "it", "of", "on", "or", "the", "to", "what", "when",
#         "why", "with", "describe", "explain", "tell", "give",
#     }
#     normalized = _normalize_question(text)
#     return {
#         word
#         for word in re.findall(r"[a-z0-9]+", normalized.lower())
#         if word not in stop_words
#     }


# def _normalize_question(text):
#     normalized = text.casefold()
#     normalized = re.sub(r"\([^)]*\)", " ", normalized)
#     normalized = normalized.replace("-", " ")
#     normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
#     normalized = re.sub(r"\bsqli\b", "sql injection", normalized)
#     normalized = re.sub(r"\bxss\b", "cross site scripting", normalized)
#     exception_aliases = {
#         "why do i get a keyerror": "what is a keyerror",
#         "why do i get keyerror": "what is a keyerror",
#         "why do i get a nameerror": "what is a nameerror",
#         "why do i get a typeerror": "what is a typeerror",
#         "why do i get an indexerror": "what is an indexerror",
#         "why do i get an attributeerror": "what is an attributeerror",
#         "why do i get a valueerror": "what is a valueerror",
#     }
#     for source, target in exception_aliases.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(r"\bbruteforce\b|\bbrute force\b", "brute force", normalized)
#     normalized = normalized.replace("cross-site", "cross site")
#     normalized = normalized.replace("vulnerability scanning", "vulnerability scan")
#     normalized = normalized.replace("penetration testing", "penetration test")
#     for source, target in {
#         "zjanasena": "janasena",
#         "janasena": "jana sena",
#         "cheif": "chief",
#         "andhras": "andhra",
#         "seleniumj": "selenium",
#         "as per": "",
#         "in andhra pradesh": "",
#         "largest film budget": "largest telugu film budget",
#     }.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(
#         r"^what is brute force attack$|^what is brute force attack in security testing$",
#         "what is a brute force attack in security testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is sql injection and how do you test for it$",
#         "what is sql injection testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is cross site scripting cross site scripting and what are its main types$",
#         "what are the main types of cross site scripting",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^how do you approach testing api security$",
#         "how do you approach testing api security",
#         normalized,
#     )
#     for exception_name in (
#         "modulenotfounderror", "importerror", "syntaxerror", "indentationerror",
#         "taberror", "nameerror", "typeerror", "valueerror", "indexerror",
#         "keyerror", "attributeerror", "unboundlocalerror", "zerodivisionerror",
#         "filenotfounderror", "permissionerror",
#     ):
#         if exception_name in normalized and normalized != f"what is a {exception_name}":
#             article = "an" if exception_name[0] in "aeiou" else "a"
#             normalized = f"what is {article} {exception_name}"
#             break
#     return " ".join(normalized.split())


# def _load_instruction_answers():
#     answers = []
#     if not INSTRUCTION_SPLIT_DIR.exists():
#         return answers
#     for split_path in sorted(INSTRUCTION_SPLIT_DIR.glob("*.jsonl")):
#         for line in split_path.read_text(encoding="utf-8").splitlines():
#             if not line.strip():
#                 continue
#             record = json.loads(line)
#             messages = record.get("messages", [])
#             user = next((m["content"] for m in messages if m.get("role") == "user"), None)
#             assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
#             if user and assistant:
#                 answers.append((user.strip(), assistant.strip(), _question_terms(user)))
#     return answers


# def _load_local_documents():
#     documents = load_corpus_documents(CORPUS_DIR) if CORPUS_DIR.exists() else []
#     if WEB_CORPUS_DIR.exists():
#         documents.extend(
#             (
#                 f"web/{name}",
#                 text,
#             )
#             for name, text in load_corpus_documents(WEB_CORPUS_DIR)
#         )
#     return documents


# class GenieeChat:
#     """Interactive Geniee chat using the same format used during SFT."""

#     def __init__(
#         self,
#         generator,
#         system_prompt=DEFAULT_SYSTEM_PROMPT,
#         knowledge_base=None,
#         retriever=None,
#         web_retriever=None,
#     ):
#         self.generator = generator
#         self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)
#         self.knowledge_base = knowledge_base if knowledge_base is not None else _load_instruction_answers()
#         self.retriever = retriever if retriever is not None else LocalRetriever(
#             _load_local_documents()
#         )
#         self.web_retriever = web_retriever if web_retriever is not None else WebRetriever()
#         self.last_source: str = "local_model"
#         self.last_sources: list[str] = []

#     def _retrieve_answer(self, user_text):
#         query = _normalize_question(user_text.strip())
#         query_terms = _question_terms(query)
#         if not query_terms:
#             return None

#         exact_match = next(
#             (answer for question, answer, _ in self.knowledge_base if _normalize_question(question) == query),
#             None,
#         )
#         if exact_match:
#             return exact_match

#         best_answer = None
#         best_score = 0.0
#         for _, answer, question_terms in self.knowledge_base:
#             if not question_terms:
#                 continue
#             overlap = len(query_terms & question_terms)
#             score = overlap / len(query_terms | question_terms)
#             query_coverage = overlap / len(query_terms)
#             if query_coverage >= 0.8 and score > best_score:
#                 best_score = score
#                 best_answer = answer

#         return best_answer if best_score >= 0.65 else None

#     def _fit_prompt_to_context(self, max_new_tokens):
#         """Drop oldest turns until prompt + generation fit the context window."""
#         tokenizer = self.generator.tokenizer
#         max_context = getattr(self.generator.model.config, "max_seq_len", 2048)
#         while True:
#             prompt = self.conversation.prompt()
#             token_count = len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
#             if token_count + max_new_tokens <= max_context:
#                 return prompt
#             messages = self.conversation.memory.messages
#             if len(messages) <= 2:
#                 return prompt
#             # Remove the oldest complete user/assistant pair.
#             del messages[:2]

#     @staticmethod
#     def clean_response(text):
#         if not text:
#             return "I am sorry, I could not generate a response."
#         for marker in ("<|user|>", "<|system|>", "<|assistant|>"):
#             if marker in text:
#                 text = text.split(marker, 1)[0]
#         text = text.replace("<|endoftext|>", "").replace("⁇", "")
#         return text.strip() or "I am sorry, I could not generate a response."

#     @staticmethod
#     def _looks_unreliable(text, question=""):
#         words = text.split()
#         if len(words) < 5 or len(set(words)) < max(3, len(words) // 3):
#             return True
#         question_terms = _question_terms(question)
#         response_terms = _question_terms(text)
#         return bool(question_terms) and not question_terms.intersection(response_terms)

#     def respond(self, user_text: str, max_new_tokens: int = 80, enable_web_search: bool = True) -> str:
#         user_text = user_text.strip()
#         if not user_text:
#             return ""

#         if max_new_tokens <= 0:
#             raise ValueError("max_new_tokens must be greater than 0")

#         self.conversation.add_user(user_text)
#         self.last_source = "local_model"
#         self.last_sources = []

#         # 1. Python Traceback Debugger
#         traceback_answer = diagnose_traceback(user_text)
#         if traceback_answer:
#             self.last_source = "debugger"
#             self.last_sources = []
#             self.conversation.add_assistant(traceback_answer)
#             return traceback_answer

#         # 2. Instruction/Knowledge Base Direct Match
#         retrieved = self._retrieve_answer(user_text)
#         if retrieved:
#             self.last_source = "knowledge_base"
#             self.last_sources = []
#             self.conversation.add_assistant(retrieved)
#             return retrieved

#         # 3. Retrieve local corpus references
#         references = self.retriever.retrieve(user_text)

#         # 4. Web search fallback if no local references found
#         if not references and enable_web_search:
#             try:
#                 web_evidence = self.web_retriever.retrieve(user_text)
#             except Exception:
#                 web_evidence = []
#             if web_evidence:
#                 response = " ".join(sentence for sentence, _ in web_evidence)
#                 self.last_source = "web"
#                 self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
#                 self.conversation.add_assistant(response)
#                 return response

#         # 5. Model Generation with RAG Injection
#         prompt = self._fit_prompt_to_context(max_new_tokens)
#         if references:
#             self.last_source = "local_rag"
#             self.last_sources = [name for name, _ in references]
#             reference_text = "\n\nRelevant local reference:\n" + "\n\n".join(
#                 f"[{name}] {text[:1200]}" for name, text in references
#             )
#             if "<|assistant|>\n" in prompt:
#                 prompt_head, assistant_marker = prompt.rsplit("<|assistant|>\n", 1)
#                 prompt = prompt_head + reference_text + "\n<|assistant|>\n" + assistant_marker
#             else:
#                 prompt = prompt + "\n" + reference_text

#         try:
#             raw = self.generator.generate(
#                 prompt,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_k=30,
#                 top_p=0.90,
#                 repetition_penalty=1.05,
#                 do_sample=False,
#                 return_full_text=False,
#             )
#         except (OSError, RuntimeError, ValueError):
#             if self.conversation.memory.messages:
#                 self.conversation.memory.messages.pop()
#             raise

#         response = self.clean_response(raw)

#         # 6. Unreliable response verification and fallback
#         if references and self._looks_unreliable(response, user_text):
#             evidence = self.retriever.best_sentences(user_text)
#             if evidence:
#                 response = " ".join(evidence)
#         elif not references and self._looks_unreliable(response, user_text):
#             if enable_web_search:
#                 try:
#                     web_evidence = self.web_retriever.retrieve(user_text)
#                 except Exception:
#                     web_evidence = []
#                 if web_evidence:
#                     response = " ".join(sentence for sentence, _ in web_evidence)
#                     self.last_source = "web"
#                     self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
#                 else:
#                     response = UNKNOWN_ANSWER
#             else:
#                 response = UNKNOWN_ANSWER

#         self.conversation.add_assistant(response)
#         return response

#     def start(self):
#         print("\n" + "=" * 70)
#         print("GENIEE CHAT")
#         print("=" * 70)
#         print("Commands: exit | quit | clear")
#         print("Model: SFT instruction-tuned Geniee")

#         while True:
#             try:
#                 user_input = input("\nYou: ").strip()
#             except (KeyboardInterrupt, EOFError):
#                 print("\nGoodbye!")
#                 break

#             if not user_input:
#                 continue
#             if user_input.lower() in {"exit", "quit"}:
#                 print("Goodbye!")
#                 break
#             if user_input.lower() == "clear":
#                 self.conversation.clear()
#                 print("Conversation cleared.")
#                 continue

#             try:
#                 response = self.respond(user_input)
#                 label = "Geniee (web-grounded)" if self.last_source == "web" else "Geniee"
#                 print(f"\n{label}: {response}")
#             except Exception as exc:
#                 print(f"\nGeneration error: {exc}")


# def load_chatbot():
#     if not CHECKPOINT_PATH.exists():
#         raise FileNotFoundError(
#             f"SFT checkpoint not found: {CHECKPOINT_PATH}\n"
#             "Run training/train_pretrain.py first, then training/train_sft.py."
#         )
#     return GenieeChat(load_geniee(CHECKPOINT_PATH))


# def main():
#     load_chatbot().start()


# if __name__ == "__main__":
#     main()

# from pathlib import Path
# import sys
# import json
# import re

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from chat.conversation import GenieeConversation
# from chat.prompt import DEFAULT_SYSTEM_PROMPT
# from generation.generate import load_geniee
# from python_debugger import diagnose_traceback
# from rag.retriever import LocalRetriever, load_corpus_documents
# from rag.web_retriever import WebRetriever

# CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
# INSTRUCTION_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
# CORPUS_DIR = PROJECT_ROOT / "corpus"
# WEB_CORPUS_DIR = PROJECT_ROOT / "data" / "web" / "cleaned"
# UNKNOWN_ANSWER = (
#     "I do not have enough trained information to answer that accurately yet. "
#     "Please add a trusted example or document for this topic."
# )


# def _question_terms(text):
#     stop_words = {
#         "a", "an", "and", "are", "can", "do", "does", "how", "in",
#         "is", "it", "of", "on", "or", "the", "to", "what", "when",
#         "why", "with", "describe", "explain", "tell", "give",
#     }
#     normalized = _normalize_question(text)
#     return {
#         word
#         for word in re.findall(r"[a-z0-9]+", normalized.lower())
#         if word not in stop_words
#     }


# def _normalize_question(text):
#     normalized = text.casefold()
#     normalized = re.sub(r"\([^)]*\)", " ", normalized)
#     normalized = normalized.replace("-", " ")
#     normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
#     normalized = re.sub(r"\bsqli\b", "sql injection", normalized)
#     normalized = re.sub(r"\bxss\b", "cross site scripting", normalized)
#     exception_aliases = {
#         "why do i get a keyerror": "what is a keyerror",
#         "why do i get keyerror": "what is a keyerror",
#         "why do i get a nameerror": "what is a nameerror",
#         "why do i get a typeerror": "what is a typeerror",
#         "why do i get an indexerror": "what is an indexerror",
#         "why do i get an attributeerror": "what is an attributeerror",
#         "why do i get a valueerror": "what is a valueerror",
#     }
#     for source, target in exception_aliases.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(r"\bbruteforce\b|\bbrute force\b", "brute force", normalized)
#     normalized = normalized.replace("cross-site", "cross site")
#     normalized = normalized.replace("vulnerability scanning", "vulnerability scan")
#     normalized = normalized.replace("penetration testing", "penetration test")
#     for source, target in {
#         "zjanasena": "janasena",
#         "janasena": "jana sena",
#         "cheif": "chief",
#         "andhras": "andhra",
#         "seleniumj": "selenium",
#         "as per": "",
#         "in andhra pradesh": "",
#         "largest film budget": "largest telugu film budget",
#     }.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(
#         r"^what is brute force attack$|^what is brute force attack in security testing$",
#         "what is a brute force attack in security testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is sql injection and how do you test for it$",
#         "what is sql injection testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is cross site scripting cross site scripting and what are its main types$",
#         "what are the main types of cross site scripting",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^how do you approach testing api security$",
#         "how do you approach testing api security",
#         normalized,
#     )
#     for exception_name in (
#         "modulenotfounderror", "importerror", "syntaxerror", "indentationerror",
#         "taberror", "nameerror", "typeerror", "valueerror", "indexerror",
#         "keyerror", "attributeerror", "unboundlocalerror", "zerodivisionerror",
#         "filenotfounderror", "permissionerror",
#     ):
#         if exception_name in normalized and normalized != f"what is a {exception_name}":
#             article = "an" if exception_name[0] in "aeiou" else "a"
#             normalized = f"what is {article} {exception_name}"
#             break
#     return " ".join(normalized.split())


# def _load_instruction_answers():
#     answers = []
#     if not INSTRUCTION_SPLIT_DIR.exists():
#         return answers
#     for split_path in sorted(INSTRUCTION_SPLIT_DIR.glob("*.jsonl")):
#         for line in split_path.read_text(encoding="utf-8").splitlines():
#             if not line.strip():
#                 continue
#             record = json.loads(line)
#             messages = record.get("messages", [])
#             user = next((m["content"] for m in messages if m.get("role") == "user"), None)
#             assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
#             if user and assistant:
#                 answers.append((user.strip(), assistant.strip(), _question_terms(user)))
#     return answers


# def _load_local_documents():
#     documents = load_corpus_documents(CORPUS_DIR) if CORPUS_DIR.exists() else []
#     if WEB_CORPUS_DIR.exists():
#         documents.extend(
#             (
#                 f"web/{name}",
#                 text,
#             )
#             for name, text in load_corpus_documents(WEB_CORPUS_DIR)
#         )
#     return documents


# class GenieeChat:
#     """Interactive Geniee chat using the same format used during SFT."""

#     def __init__(
#         self,
#         generator,
#         system_prompt=DEFAULT_SYSTEM_PROMPT,
#         knowledge_base=None,
#         retriever=None,
#         web_retriever=None,
#     ):
#         self.generator = generator
#         self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)
#         self.knowledge_base = knowledge_base if knowledge_base is not None else _load_instruction_answers()
#         self.retriever = retriever if retriever is not None else LocalRetriever(
#             _load_local_documents()
#         )
#         self.web_retriever = web_retriever if web_retriever is not None else WebRetriever()
#         self.last_source: str = "local_model"
#         self.last_sources: list[str] = []

#     def _retrieve_answer(self, user_text):
#         query = _normalize_question(user_text.strip())
#         query_terms = _question_terms(query)
#         if not query_terms:
#             return None

#         exact_match = next(
#             (answer for question, answer, _ in self.knowledge_base if _normalize_question(question) == query),
#             None,
#         )
#         if exact_match:
#             return exact_match

#         best_answer = None
#         best_score = 0.0
#         for _, answer, question_terms in self.knowledge_base:
#             if not question_terms:
#                 continue
#             overlap = len(query_terms & question_terms)
#             score = overlap / len(query_terms | question_terms)
#             query_coverage = overlap / len(query_terms)
#             if query_coverage >= 0.8 and score > best_score:
#                 best_score = score
#                 best_answer = answer

#         return best_answer if best_score >= 0.65 else None

#     def _evaluate_local_relevance(self, user_text: str, references: list) -> float:
#         """Content-based evaluation: computes key query term coverage in retrieved local text."""
#         if not references:
#             return 0.0
#         query_terms = _question_terms(user_text)
#         if not query_terms:
#             return 0.0

#         corpus_text = " ".join(text.lower() for _, text in references)
#         matched_terms = {term for term in query_terms if term in corpus_text}
#         coverage = len(matched_terms) / len(query_terms)
#         return coverage

#     def _requires_external_context(self, user_text: str) -> bool:
#         """Detects whether query intent fundamentally requires dynamic or real-time external data."""
#         dynamic_concepts = {
#             "today", "todays", "today's", "latest", "current", "now", "price", "rate", 
#             "cost", "gold", "silver", "stock", "weather", "news", "score", "market"
#         }
#         tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
#         return bool(tokens.intersection(dynamic_concepts))

#     def _fit_prompt_to_context(self, max_new_tokens):
#         """Drop oldest turns until prompt + generation fit the context window."""
#         tokenizer = self.generator.tokenizer
#         max_context = getattr(self.generator.model.config, "max_seq_len", 2048)
#         while True:
#             prompt = self.conversation.prompt()
#             token_count = len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
#             if token_count + max_new_tokens <= max_context:
#                 return prompt
#             messages = self.conversation.memory.messages
#             if len(messages) <= 2:
#                 return prompt
#             del messages[:2]

#     @staticmethod
#     def clean_response(text):
#         if not text:
#             return "I am sorry, I could not generate a response."
#         for marker in ("<|user|>", "<|system|>", "<|assistant|>"):
#             if marker in text:
#                 text = text.split(marker, 1)[0]
#         text = text.replace("<|endoftext|>", "").replace("⁇", "")
#         return text.strip() or "I am sorry, I could not generate a response."

#     @staticmethod
#     def _looks_unreliable(text, question=""):
#         words = text.split()
#         if len(words) < 5 or len(set(words)) < max(3, len(words) // 3):
#             return True
#         question_terms = _question_terms(question)
#         response_terms = _question_terms(text)
#         return bool(question_terms) and not question_terms.intersection(response_terms)

#     def respond(self, user_text: str, max_new_tokens: int = 80, enable_web_search: bool = True) -> str:
#         user_text = user_text.strip()
#         if not user_text:
#             return ""

#         if max_new_tokens <= 0:
#             raise ValueError("max_new_tokens must be greater than 0")

#         self.conversation.add_user(user_text)
#         self.last_source = "local_model"
#         self.last_sources = []

#         # 1. Python Traceback Debugger
#         traceback_answer = diagnose_traceback(user_text)
#         if traceback_answer:
#             self.last_source = "debugger"
#             self.last_sources = []
#             self.conversation.add_assistant(traceback_answer)
#             return traceback_answer

#         # 2. Instruction/Knowledge Base Direct Match
#         retrieved = self._retrieve_answer(user_text)
#         if retrieved:
#             self.last_source = "knowledge_base"
#             self.last_sources = []
#             self.conversation.add_assistant(retrieved)
#             return retrieved

#         # 3. Retrieve local corpus references and evaluate content relevance
#         raw_references = self.retriever.retrieve(user_text)
#         relevance_score = self._evaluate_local_relevance(user_text, raw_references)
        
#         # Consider references valid ONLY if content coverage is at least 50%
#         has_sufficient_local_data = relevance_score >= 0.50
#         references = raw_references if has_sufficient_local_data else []
#         needs_web_data = self._requires_external_context(user_text) or not has_sufficient_local_data

#         # 4. Web Search Route (Triggered if local data is missing/irrelevant OR external context is needed)
#         if needs_web_data and enable_web_search:
#             print(f"--> [WEB RETRIEVAL TRIGGERED] Query: '{user_text}' | Local Coverage: {relevance_score:.2f}")
#             try:
#                 web_evidence = self.web_retriever.retrieve(user_text)
#             except Exception as e:
#                 print(f"--> [WEB RETRIEVAL ERROR] {e}")
#                 web_evidence = []
                
#             if web_evidence:
#                 response = " ".join(sentence for sentence, _ in web_evidence)
#                 self.last_source = "web"
#                 self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
#                 self.conversation.add_assistant(response)
#                 return response

#         # 5. Model Generation with RAG Injection (used when relevant local context exists)
#         prompt = self._fit_prompt_to_context(max_new_tokens)
#         if references:
#             self.last_source = "local_rag"
#             self.last_sources = [name for name, _ in references]
#             reference_text = "\n\nRelevant local reference:\n" + "\n\n".join(
#                 f"[{name}] {text[:1200]}" for name, text in references
#             )
#             if "<|assistant|>\n" in prompt:
#                 prompt_head, assistant_marker = prompt.rsplit("<|assistant|>\n", 1)
#                 prompt = prompt_head + reference_text + "\n<|assistant|>\n" + assistant_marker
#             else:
#                 prompt = prompt + "\n" + reference_text

#         try:
#             raw = self.generator.generate(
#                 prompt,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_k=30,
#                 top_p=0.90,
#                 repetition_penalty=1.05,
#                 do_sample=False,
#                 return_full_text=False,
#             )
#         except (OSError, RuntimeError, ValueError):
#             if self.conversation.memory.messages:
#                 self.conversation.memory.messages.pop()
#             raise

#         response = self.clean_response(raw)

#         # 6. Unreliable response verification and fallback
#         if references and self._looks_unreliable(response, user_text):
#             evidence = self.retriever.best_sentences(user_text)
#             if evidence:
#                 response = " ".join(evidence)
#         elif not references and self._looks_unreliable(response, user_text):
#             if enable_web_search:
#                 print(f"--> [WEB RETRIEVAL FALLBACK] Unreliable local output for: '{user_text}'")
#                 try:
#                     web_evidence = self.web_retriever.retrieve(user_text)
#                 except Exception:
#                     web_evidence = []
#                 if web_evidence:
#                     response = " ".join(sentence for sentence, _ in web_evidence)
#                     self.last_source = "web"
#                     self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
#                 else:
#                     response = UNKNOWN_ANSWER
#             else:
#                 response = UNKNOWN_ANSWER

#         self.conversation.add_assistant(response)
#         return response

#     def start(self):
#         print("\n" + "=" * 70)
#         print("GENIEE CHAT")
#         print("=" * 70)
#         print("Commands: exit | quit | clear")
#         print("Model: SFT instruction-tuned Geniee")

#         while True:
#             try:
#                 user_input = input("\nYou: ").strip()
#             except (KeyboardInterrupt, EOFError):
#                 print("\nGoodbye!")
#                 break

#             if not user_input:
#                 continue
#             if user_input.lower() in {"exit", "quit"}:
#                 print("Goodbye!")
#                 break
#             if user_input.lower() == "clear":
#                 self.conversation.clear()
#                 print("Conversation cleared.")
#                 continue

#             try:
#                 response = self.respond(user_input)
#                 label = "Geniee (web-grounded)" if self.last_source == "web" else "Geniee"
#                 print(f"\n{label}: {response}")
#             except Exception as exc:
#                 print(f"\nGeneration error: {exc}")


# def load_chatbot():
#     if not CHECKPOINT_PATH.exists():
#         raise FileNotFoundError(
#             f"SFT checkpoint not found: {CHECKPOINT_PATH}\n"
#             "Run training/train_pretrain.py first, then training/train_sft.py."
#         )
#     return GenieeChat(load_geniee(CHECKPOINT_PATH))


# def main():
#     load_chatbot().start()


# if __name__ == "__main__":
#     main()

# from pathlib import Path
# import sys
# import json
# import re

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from chat.conversation import GenieeConversation
# from chat.prompt import DEFAULT_SYSTEM_PROMPT
# from generation.generate import load_geniee
# from python_debugger import diagnose_traceback
# from rag.retriever import LocalRetriever, load_corpus_documents
# from rag.web_retriever import WebRetriever

# CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
# INSTRUCTION_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
# CORPUS_DIR = PROJECT_ROOT / "corpus"
# WEB_CORPUS_DIR = PROJECT_ROOT / "data" / "web" / "cleaned"
# UNKNOWN_ANSWER = (
#     "I do not have enough trained information to answer that accurately yet. "
#     "Please add a trusted example or document for this topic."
# )


# def _question_terms(text):
#     stop_words = {
#         "a", "an", "and", "are", "can", "do", "does", "how", "in",
#         "is", "it", "of", "on", "or", "the", "to", "what", "when",
#         "why", "with", "describe", "explain", "tell", "give",
#     }
#     normalized = _normalize_question(text)
#     return {
#         word
#         for word in re.findall(r"[a-z0-9]+", normalized.lower())
#         if word not in stop_words
#     }


# def _normalize_question(text):
#     normalized = text.casefold()
#     normalized = re.sub(r"\([^)]*\)", " ", normalized)
#     normalized = normalized.replace("-", " ")
#     normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
#     normalized = re.sub(r"\bsqli\b", "sql injection", normalized)
#     normalized = re.sub(r"\bxss\b", "cross site scripting", normalized)
#     exception_aliases = {
#         "why do i get a keyerror": "what is a keyerror",
#         "why do i get keyerror": "what is a keyerror",
#         "why do i get a nameerror": "what is a nameerror",
#         "why do i get a typeerror": "what is a typeerror",
#         "why do i get an indexerror": "what is an indexerror",
#         "why do i get an attributeerror": "what is an attributeerror",
#         "why do i get a valueerror": "what is a valueerror",
#     }
#     for source, target in exception_aliases.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(r"\bbruteforce\b|\bbrute force\b", "brute force", normalized)
#     normalized = normalized.replace("cross-site", "cross site")
#     normalized = normalized.replace("vulnerability scanning", "vulnerability scan")
#     normalized = normalized.replace("penetration testing", "penetration test")
#     for source, target in {
#         "zjanasena": "janasena",
#         "janasena": "jana sena",
#         "cheif": "chief",
#         "andhras": "andhra",
#         "seleniumj": "selenium",
#         "as per": "",
#         "in andhra pradesh": "",
#         "largest film budget": "largest telugu film budget",
#     }.items():
#         normalized = normalized.replace(source, target)
#     normalized = re.sub(
#         r"^what is brute force attack$|^what is brute force attack in security testing$",
#         "what is a brute force attack in security testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is sql injection and how do you test for it$",
#         "what is sql injection testing",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^what is cross site scripting cross site scripting and what are its main types$",
#         "what are the main types of cross site scripting",
#         normalized,
#     )
#     normalized = re.sub(
#         r"^how do you approach testing api security$",
#         "how do you approach testing api security",
#         normalized,
#     )
#     for exception_name in (
#         "modulenotfounderror", "importerror", "syntaxerror", "indentationerror",
#         "taberror", "nameerror", "typeerror", "valueerror", "indexerror",
#         "keyerror", "attributeerror", "unboundlocalerror", "zerodivisionerror",
#         "filenotfounderror", "permissionerror",
#     ):
#         if exception_name in normalized and normalized != f"what is a {exception_name}":
#             article = "an" if exception_name[0] in "aeiou" else "a"
#             normalized = f"what is {article} {exception_name}"
#             break
#     return " ".join(normalized.split())


# def _load_instruction_answers():
#     answers = []
#     if not INSTRUCTION_SPLIT_DIR.exists():
#         return answers
#     for split_path in sorted(INSTRUCTION_SPLIT_DIR.glob("*.jsonl")):
#         for line in split_path.read_text(encoding="utf-8").splitlines():
#             if not line.strip():
#                 continue
#             record = json.loads(line)
#             messages = record.get("messages", [])
#             user = next((m["content"] for m in messages if m.get("role") == "user"), None)
#             assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
#             if user and assistant:
#                 answers.append((user.strip(), assistant.strip(), _question_terms(user)))
#     return answers


# def _load_local_documents():
#     documents = load_corpus_documents(CORPUS_DIR) if CORPUS_DIR.exists() else []
#     if WEB_CORPUS_DIR.exists():
#         documents.extend(
#             (
#                 f"web/{name}",
#                 text,
#             )
#             for name, text in load_corpus_documents(WEB_CORPUS_DIR)
#         )
#     return documents


# class GenieeChat:
#     """Interactive Geniee chat using the same format used during SFT."""

#     def __init__(
#         self,
#         generator,
#         system_prompt=DEFAULT_SYSTEM_PROMPT,
#         knowledge_base=None,
#         retriever=None,
#         web_retriever=None,
#     ):
#         self.generator = generator
#         self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)
#         self.knowledge_base = knowledge_base if knowledge_base is not None else _load_instruction_answers()
#         self.retriever = retriever if retriever is not None else LocalRetriever(
#             _load_local_documents()
#         )
#         self.web_retriever = web_retriever if web_retriever is not None else WebRetriever()
#         self.last_source: str = "local_model"
#         self.last_sources: list[str] = []

#     def _retrieve_answer(self, user_text):
#         query = _normalize_question(user_text.strip())
#         query_terms = _question_terms(query)
#         if not query_terms:
#             return None

#         exact_match = next(
#             (answer for question, answer, _ in self.knowledge_base if _normalize_question(question) == query),
#             None,
#         )
#         if exact_match:
#             return exact_match

#         best_answer = None
#         best_score = 0.0
#         for _, answer, question_terms in self.knowledge_base:
#             if not question_terms:
#                 continue
#             overlap = len(query_terms & question_terms)
#             score = overlap / len(query_terms | question_terms)
#             query_coverage = overlap / len(query_terms)
#             if query_coverage >= 0.8 and score > best_score:
#                 best_score = score
#                 best_answer = answer

#         return best_answer if best_score >= 0.65 else None

#     def _evaluate_local_relevance(self, user_text: str, references: list) -> float:
#         """Content-based evaluation: computes key query term coverage in retrieved local text."""
#         if not references:
#             return 0.0
#         query_terms = _question_terms(user_text)
#         if not query_terms:
#             return 0.0

#         corpus_text = " ".join(text.lower() for _, text in references)
#         matched_terms = {term for term in query_terms if term in corpus_text}
#         coverage = len(matched_terms) / len(query_terms)
#         return coverage

#     def _requires_external_context(self, user_text: str) -> bool:
#         """Detects whether query intent fundamentally requires dynamic or real-time external data."""
#         dynamic_concepts = {
#             "today", "todays", "today's", "latest", "current", "now", "price", "rate", 
#             "cost", "gold", "silver", "stock", "weather", "news", "score", "market"
#         }
#         tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
#         return bool(tokens.intersection(dynamic_concepts))

#     def _fit_prompt_to_context(self, max_new_tokens):
#         """Drop oldest turns until prompt + generation fit the context window."""
#         tokenizer = self.generator.tokenizer
#         max_context = getattr(self.generator.model.config, "max_seq_len", 2048)
#         while True:
#             prompt = self.conversation.prompt()
#             token_count = len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
#             if token_count + max_new_tokens <= max_context:
#                 return prompt
#             messages = self.conversation.memory.messages
#             if len(messages) <= 2:
#                 return prompt
#             del messages[:2]

#     @staticmethod
#     def clean_response(text):
#         if not text:
#             return "I am sorry, I could not generate a response."
#         for marker in ("<|user|>", "<|system|>", "<|assistant|>"):
#             if marker in text:
#                 text = text.split(marker, 1)[0]
#         text = text.replace("<|endoftext|>", "").replace("⁇", "")
#         return text.strip() or "I am sorry, I could not generate a response."

#     @staticmethod
#     def _looks_unreliable(text, question=""):
#         words = text.split()
#         if len(words) < 5 or len(set(words)) < max(3, len(words) // 3):
#             return True
#         question_terms = _question_terms(question)
#         response_terms = _question_terms(text)
#         return bool(question_terms) and not question_terms.intersection(response_terms)

#     def respond(self, user_text: str, max_new_tokens: int = 80, enable_web_search: bool = True) -> str:
#         user_text = user_text.strip()
#         if not user_text:
#             return ""

#         if max_new_tokens <= 0:
#             raise ValueError("max_new_tokens must be greater than 0")

#         self.conversation.add_user(user_text)
#         self.last_source = "local_model"
#         self.last_sources = []

#         # 1. Python Traceback Debugger
#         traceback_answer = diagnose_traceback(user_text)
#         if traceback_answer:
#             self.last_source = "debugger"
#             self.last_sources = []
#             self.conversation.add_assistant(traceback_answer)
#             return traceback_answer

#         # 2. Instruction/Knowledge Base Direct Match
#         retrieved = self._retrieve_answer(user_text)
#         if retrieved:
#             self.last_source = "knowledge_base"
#             self.last_sources = []
#             self.conversation.add_assistant(retrieved)
#             return retrieved

#         # 3. Retrieve local corpus references and evaluate content relevance
#         raw_references = self.retriever.retrieve(user_text)
#         relevance_score = self._evaluate_local_relevance(user_text, raw_references)
        
#         # Consider references valid ONLY if content coverage is at least 50%
#         has_sufficient_local_data = relevance_score >= 0.50
#         references = raw_references if has_sufficient_local_data else []
#         needs_web_data = self._requires_external_context(user_text) or not has_sufficient_local_data

#         # 4. Web Search Route (Triggered if local data is missing/irrelevant OR external context is needed)
#         if needs_web_data and enable_web_search:
#             print(f"--> [WEB RETRIEVAL TRIGGERED] Query: '{user_text}' | Local Coverage: {relevance_score:.2f}")
#             try:
#                 web_evidence = self.web_retriever.retrieve(user_text)
#             except Exception as e:
#                 print(f"--> [WEB RETRIEVAL ERROR] {e}")
#                 web_evidence = []
                
#             if web_evidence:
#                 response = " ".join(sentence for sentence, _ in web_evidence)
#                 self.last_source = "web"
#                 self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
#                 self.conversation.add_assistant(response)
#                 return response

#         # 5. Model Generation with RAG Injection (used when relevant local context exists)
#         prompt = self._fit_prompt_to_context(max_new_tokens)
#         if references:
#             self.last_source = "local_rag"
#             self.last_sources = [name for name, _ in references]
#             reference_text = "\n\nRelevant local reference:\n" + "\n\n".join(
#                 f"[{name}] {text[:1200]}" for name, text in references
#             )
#             if "<|assistant|>\n" in prompt:
#                 prompt_head, assistant_marker = prompt.rsplit("<|assistant|>\n", 1)
#                 prompt = prompt_head + reference_text + "\n<|assistant|>\n" + assistant_marker
#             else:
#                 prompt = prompt + "\n" + reference_text

#         try:
#             raw = self.generator.generate(
#                 prompt,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_k=30,
#                 top_p=0.90,
#                 repetition_penalty=1.05,
#                 do_sample=False,
#                 return_full_text=False,
#             )
#         except (OSError, RuntimeError, ValueError):
#             if self.conversation.memory.messages:
#                 self.conversation.memory.messages.pop()
#             raise

#         response = self.clean_response(raw)

#         # 6. Unreliable response verification and fallback
#         if references and self._looks_unreliable(response, user_text):
#             evidence = self.retriever.best_sentences(user_text)
#             if evidence:
#                 response = " ".join(evidence)
#         elif not references and self._looks_unreliable(response, user_text):
#             if enable_web_search:
#                 print(f"--> [WEB RETRIEVAL FALLBACK] Unreliable local output for: '{user_text}'")
#                 try:
#                     web_evidence = self.web_retriever.retrieve(user_text)
#                 except Exception:
#                     web_evidence = []
#                 if web_evidence:
#                     response = " ".join(sentence for sentence, _ in web_evidence)
#                     self.last_source = "web"
#                     self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
#                 else:
#                     response = UNKNOWN_ANSWER
#             else:
#                 response = UNKNOWN_ANSWER

#         self.conversation.add_assistant(response)
#         return response

#     def start(self):
#         print("\n" + "=" * 70)
#         print("GENIEE CHAT")
#         print("=" * 70)
#         print("Commands: exit | quit | clear")
#         print("Model: SFT instruction-tuned Geniee")

#         while True:
#             try:
#                 user_input = input("\nYou: ").strip()
#             except (KeyboardInterrupt, EOFError):
#                 print("\nGoodbye!")
#                 break

#             if not user_input:
#                 continue
#             if user_input.lower() in {"exit", "quit"}:
#                 print("Goodbye!")
#                 break
#             if user_input.lower() == "clear":
#                 self.conversation.clear()
#                 print("Conversation cleared.")
#                 continue

#             try:
#                 response = self.respond(user_input)
#                 label = "Geniee (web-grounded)" if self.last_source == "web" else "Geniee"
#                 print(f"\n{label}: {response}")
#             except Exception as exc:
#                 print(f"\nGeneration error: {exc}")


# def load_chatbot():
#     if not CHECKPOINT_PATH.exists():
#         raise FileNotFoundError(
#             f"SFT checkpoint not found: {CHECKPOINT_PATH}\n"
#             "Run training/train_pretrain.py first, then training/train_sft.py."
#         )
#     return GenieeChat(load_geniee(CHECKPOINT_PATH))


# def main():
#     load_chatbot().start()


# if __name__ == "__main__":
#     main()

from __future__ import annotations

from pathlib import Path
import sys
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat.conversation import GenieeConversation
from chat.prompt import DEFAULT_SYSTEM_PROMPT
from generation.generate import load_geniee
from python_debugger import diagnose_traceback
from rag.retriever import LocalRetriever, load_corpus_documents
from rag.web_retriever import WebRetriever

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "geniee_sft_best.pt"
INSTRUCTION_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
CORPUS_DIR = PROJECT_ROOT / "corpus"
WEB_CORPUS_DIR = PROJECT_ROOT / "data" / "web" / "cleaned"
UNKNOWN_ANSWER = (
    "I do not have enough trained information to answer that accurately yet. "
    "Please add a trusted example or document for this topic."
)


def _question_terms(text):
    stop_words = {
        "a", "an", "and", "are", "can", "do", "does", "how", "in",
        "is", "it", "of", "on", "or", "the", "to", "what", "when",
        "why", "with", "describe", "explain", "tell", "give",
    }
    normalized = _normalize_question(text)
    return {
        word
        for word in re.findall(r"[a-z0-9]+", normalized.lower())
        if word not in stop_words
    }


def _normalize_question(text):
    normalized = text.casefold()
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\bsqli\b", "sql injection", normalized)
    normalized = re.sub(r"\bxss\b", "cross site scripting", normalized)
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
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\bbruteforce\b|\bbrute force\b", "brute force", normalized)
    normalized = normalized.replace("cross-site", "cross site")
    normalized = normalized.replace("vulnerability scanning", "vulnerability scan")
    normalized = normalized.replace("penetration testing", "penetration test")
    for source, target in {
        "zjanasena": "janasena",
        "janasena": "jana sena",
        "cheif": "chief",
        "andhras": "andhra",
        "seleniumj": "selenium",
        "as per": "",
        "in andhra pradesh": "",
        "largest film budget": "largest telugu film budget",
    }.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(
        r"^what is brute force attack$|^what is brute force attack in security testing$",
        "what is a brute force attack in security testing",
        normalized,
    )
    normalized = re.sub(
        r"^what is sql injection and how do you test for it$",
        "what is sql injection testing",
        normalized,
    )
    normalized = re.sub(
        r"^what is cross site scripting cross site scripting and what are its main types$",
        "what are the main types of cross site scripting",
        normalized,
    )
    normalized = re.sub(
        r"^how do you approach testing api security$",
        "how do you approach testing api security",
        normalized,
    )
    for exception_name in (
        "modulenotfounderror", "importerror", "syntaxerror", "indentationerror",
        "taberror", "nameerror", "typeerror", "valueerror", "indexerror",
        "keyerror", "attributeerror", "unboundlocalerror", "zerodivisionerror",
        "filenotfounderror", "permissionerror",
    ):
        if exception_name in normalized and normalized != f"what is a {exception_name}":
            article = "an" if exception_name[0] in "aeiou" else "a"
            normalized = f"what is {article} {exception_name}"
            break
    return " ".join(normalized.split())


def _load_instruction_answers():
    answers = []
    if not INSTRUCTION_SPLIT_DIR.exists():
        return answers
    for split_path in sorted(INSTRUCTION_SPLIT_DIR.glob("*.jsonl")):
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            user = next((m["content"] for m in messages if m.get("role") == "user"), None)
            assistant = next((m["content"] for m in messages if m.get("role") == "assistant"), None)
            if user and assistant:
                answers.append((user.strip(), assistant.strip(), _question_terms(user)))
    return answers


def _load_local_documents():
    documents = load_corpus_documents(CORPUS_DIR) if CORPUS_DIR.exists() else []
    if WEB_CORPUS_DIR.exists():
        documents.extend(
            (
                f"web/{name}",
                text,
            )
            for name, text in load_corpus_documents(WEB_CORPUS_DIR)
        )
    return documents


class GenieeChat:
    """Interactive Geniee chat with flashman / Geniee branding and enhanced response styling."""

    def __init__(
        self,
        generator,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        knowledge_base=None,
        retriever=None,
        web_retriever=None,
    ):
        self.generator = generator
        self.conversation = GenieeConversation(system_prompt=system_prompt, max_turns=6)
        self.knowledge_base = knowledge_base if knowledge_base is not None else _load_instruction_answers()
        self.retriever = retriever if retriever is not None else LocalRetriever(
            _load_local_documents()
        )
        self.web_retriever = web_retriever if web_retriever is not None else WebRetriever()
        self.last_source: str = "local_model"
        self.last_sources: list[str] = []

    def _retrieve_answer(self, user_text):
        query = _normalize_question(user_text.strip())
        query_terms = _question_terms(query)
        if not query_terms:
            return None

        exact_match = next(
            (answer for question, answer, _ in self.knowledge_base if _normalize_question(question) == query),
            None,
        )
        if exact_match:
            return exact_match

        best_answer = None
        best_score = 0.0
        for _, answer, question_terms in self.knowledge_base:
            if not question_terms:
                continue
            overlap = len(query_terms & question_terms)
            score = overlap / len(query_terms | question_terms)
            query_coverage = overlap / len(query_terms)
            if query_coverage >= 0.8 and score > best_score:
                best_score = score
                best_answer = answer

        return best_answer if best_score >= 0.65 else None

    def _evaluate_local_relevance(self, user_text: str, references: list) -> float:
        if not references:
            return 0.0
        query_terms = _question_terms(user_text)
        if not query_terms:
            return 0.0

        corpus_text = " ".join(text.lower() for _, text in references)
        matched_terms = {term for term in query_terms if term in corpus_text}
        coverage = len(matched_terms) / len(query_terms)
        return coverage

    def _requires_external_context(self, user_text: str) -> bool:
        dynamic_concepts = {
            "today", "todays", "today's", "latest", "current", "now", "price", "rate", 
            "cost", "gold", "silver", "stock", "weather", "news", "score", "market",
            "vs", "versus", "match", "man of the match", "player of the match"
        }
        tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
        return bool(tokens.intersection(dynamic_concepts))

    def _fit_prompt_to_context(self, max_new_tokens):
        tokenizer = self.generator.tokenizer
        max_context = getattr(self.generator.model.config, "max_seq_len", 2048)
        while True:
            prompt = self.conversation.prompt()
            token_count = len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
            if token_count + max_new_tokens <= max_context:
                return prompt
            messages = self.conversation.memory.messages
            if len(messages) <= 2:
                return prompt
            del messages[:2]

    @staticmethod
    def clean_response(text):
        if not text:
            return "I am sorry, I could not generate a response."
        for marker in ("<|user|>", "<|system|>", "<|assistant|>", "flashman:", "Geniee:"):
            if marker in text:
                text = text.split(marker, 1)[0]
        text = text.replace("<|endoftext|>", "").replace("⁇", "")
        return text.strip() or "I am sorry, I could not generate a response."

    @staticmethod
    def _format_attractive_output(text: str) -> str:
        """Applies clean markdown highlights to key statistics, scores, and currency values."""
        text = re.sub(r"\s+", " ", text).strip()
        
        # Highlight scores, run totals, wickets
        text = re.sub(r"(\b\d+/\d+\b|\b\d+\s*wickets?\b|\b\d+\s*runs?\b)", r"**\1**", text, flags=re.IGNORECASE)
        # Highlight monetary/currency figures
        text = re.sub(r"(₹\s*[\d,]+|Rs\.?\s*[\d,]+|\$\s*[\d,]+|\b\d+\s*carat\b)", r"**\1**", text, flags=re.IGNORECASE)
        
        if text and not text.endswith((".", "!", "?")):
            text += "."
        return text

    @staticmethod
    def _looks_unreliable(text, question=""):
        words = text.split()
        if len(words) < 5 or len(set(words)) < max(3, len(words) // 3):
            return True
        question_terms = _question_terms(question)
        response_terms = _question_terms(text)
        return bool(question_terms) and not question_terms.intersection(response_terms)

    def respond(
        self,
        user_text: str,
        history: list[dict[str, str]] | None = None,
        max_new_tokens: int = 80,
        enable_web_search: bool = True,
    ) -> str:
        user_text = user_text.strip()
        if not user_text:
            return ""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")

        # ------------------------------------------------------------
        # SYNC CONVERSATION HISTORY (FOR API & MULTI-TURN CONTEXT)
        # ------------------------------------------------------------
        if history is not None:
            self.conversation.clear()
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "").strip()
                if not content:
                    continue
                if role == "user":
                    self.conversation.add_user(content)
                elif role in ("assistant", "bot", "geniee"):
                    self.conversation.add_assistant(content)

        # Ensure current user turn is added if not already trailing history
        memory_messages = getattr(self.conversation.memory, "messages", [])
        if not memory_messages or memory_messages[-1].get("content") != user_text:
            self.conversation.add_user(user_text)

        self.last_source = "local_model"
        self.last_sources = []

        # 1. Python Traceback Debugger
        traceback_answer = diagnose_traceback(user_text)
        if traceback_answer:
            self.last_source = "debugger"
            self.last_sources = []
            self.conversation.add_assistant(traceback_answer)
            return traceback_answer

        # 2. Instruction/Knowledge Base Direct Match
        retrieved = self._retrieve_answer(user_text)
        if retrieved:
            self.last_source = "knowledge_base"
            self.last_sources = []
            formatted_retrieved = self._format_attractive_output(retrieved)
            self.conversation.add_assistant(formatted_retrieved)
            return formatted_retrieved

        # 3. Retrieve local corpus references and evaluate content relevance
        raw_references = self.retriever.retrieve(user_text)
        relevance_score = self._evaluate_local_relevance(user_text, raw_references)
        
        has_sufficient_local_data = relevance_score >= 0.50
        references = raw_references if has_sufficient_local_data else []
        needs_web_data = self._requires_external_context(user_text) or not has_sufficient_local_data

        # 4. Web Search Route (Generates grounded, concise answers)
        if needs_web_data and enable_web_search:
            try:
                web_evidence = self.web_retriever.retrieve(user_text)
            except Exception:
                web_evidence = []
                
            if web_evidence:
                self.last_source = "web"
                self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
                
                snippets = [sentence for sentence, _ in web_evidence[:2]]
                web_context = " ".join(snippets)
                
                prompt = (
                    f"Context: {web_context}\n\n"
                    f"Question: {user_text}\n"
                    f"Instruction: Answer directly in 1 precise sentence using only the context.\n"
                    f"Answer:"
                )
                
                try:
                    raw = self.generator.generate(
                        prompt,
                        max_new_tokens=40,
                        temperature=0.1,
                        do_sample=False,
                        return_full_text=False,
                    )
                    response = self.clean_response(raw)
                    if len(response.split()) < 3 or self._looks_unreliable(response, user_text):
                        response = snippets[0]
                except Exception:
                    response = snippets[0]

                formatted_response = self._format_attractive_output(response)
                self.conversation.add_assistant(formatted_response)
                return formatted_response

        # 5. Model Generation with RAG Injection
        prompt = self._fit_prompt_to_context(max_new_tokens)
        if references:
            self.last_source = "local_rag"
            self.last_sources = [name for name, _ in references]
            reference_text = "\n\nRelevant local reference:\n" + "\n\n".join(
                f"[{name}] {text[:1200]}" for name, text in references
            )
            if "<|assistant|>\n" in prompt:
                prompt_head, assistant_marker = prompt.rsplit("<|assistant|>\n", 1)
                prompt = prompt_head + reference_text + "\n<|assistant|>\n" + assistant_marker
            else:
                prompt = prompt + "\n" + reference_text

        try:
            raw = self.generator.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=30,
                top_p=0.90,
                repetition_penalty=1.05,
                do_sample=False,
                return_full_text=False,
            )
        except (OSError, RuntimeError, ValueError):
            if self.conversation.memory.messages:
                self.conversation.memory.messages.pop()
            raise

        response = self.clean_response(raw)

        # 6. Unreliable response verification and fallback
        if references and self._looks_unreliable(response, user_text):
            evidence = self.retriever.best_sentences(user_text)
            if evidence:
                response = " ".join(evidence)
        elif not references and self._looks_unreliable(response, user_text):
            if enable_web_search:
                try:
                    web_evidence = self.web_retriever.retrieve(user_text)
                except Exception:
                    web_evidence = []
                if web_evidence:
                    snippets = [sentence for sentence, _ in web_evidence[:2]]
                    web_context = " ".join(snippets)
                    prompt = (
                        f"Context: {web_context}\n\n"
                        f"Question: {user_text}\n"
                        f"Instruction: Answer directly in 1 precise sentence using only the context.\n"
                        f"Answer:"
                    )
                    try:
                        raw = self.generator.generate(
                            prompt,
                            max_new_tokens=40,
                            temperature=0.1,
                            do_sample=False,
                            return_full_text=False,
                        )
                        response = self.clean_response(raw)
                        if len(response.split()) < 3 or self._looks_unreliable(response, user_text):
                            response = snippets[0]
                    except Exception:
                        response = snippets[0]

                    self.last_source = "web"
                    self.last_sources = list(dict.fromkeys(url for _, url in web_evidence if url))
                else:
                    response = UNKNOWN_ANSWER
            else:
                response = UNKNOWN_ANSWER

        formatted_response = self._format_attractive_output(response)
        self.conversation.add_assistant(formatted_response)
        return formatted_response

    def start(self):
        print("\n" + "═" * 60)
        print("                ✨ GENIEE INTERACTIVE CHAT ✨")
        print("═" * 60)
        print("Commands: exit | quit | clear\n")

        while True:
            try:
                user_input = input("\nflashman: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break
            if user_input.lower() == "clear":
                self.conversation.clear()
                print("Conversation cleared.")
                continue

            try:
                response = self.respond(user_input)
                print(f"\nGeniee: {response}")
            except Exception as exc:
                print(f"\nGeniee error: {exc}")


def load_chatbot():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run training/train_pretrain.py first, then training/train_sft.py."
        )
    return GenieeChat(load_geniee(CHECKPOINT_PATH))


def main():
    load_chatbot().start()


if __name__ == "__main__":
    main()