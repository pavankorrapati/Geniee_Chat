from __future__ import annotations

from pathlib import Path
import re


_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "in",
    "is", "it", "of", "on", "or", "the", "to", "what", "when", "why",
    "with", "describe", "explain", "tell", "give",
}


def _terms(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _STOP_WORDS
    }


def load_corpus_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    documents = []
    for path in sorted(corpus_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text and not ("User:" in text and "Assistant:" in text):
            documents.append((str(path.relative_to(corpus_dir)), text))
    return documents


class LocalRetriever:
    """Small dependency-free lexical retriever for the local corpus."""

    def __init__(self, documents: list[tuple[str, str]]):
        self.documents = []
        for name, text in documents:
            chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk:
                    self.documents.append((name, chunk, _terms(chunk)))

    def retrieve(self, query: str, limit: int = 2) -> list[tuple[str, str]]:
        query_terms = _terms(query)
        if not query_terms:
            return []

        ranked = []
        for name, text, document_terms in self.documents:
            if len(text) < 80:
                continue
            overlap = len(query_terms & document_terms)
            if overlap:
                coverage = overlap / len(query_terms)
                score = coverage + overlap / len(query_terms | document_terms)
                ranked.append((score, name, text))
        ranked.sort(reverse=True)
        return [(name, text) for score, name, text in ranked[:limit] if score >= 0.03]

    def best_sentences(self, query: str, limit: int = 3) -> list[str]:
        query_terms = _terms(query)
        ranked = []
        for _, text, _ in self.documents:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if len(sentence.strip()) < 80:
                    continue
                sentence_terms = _terms(sentence)
                overlap = len(query_terms & sentence_terms)
                if overlap:
                    coverage = overlap / len(query_terms)
                    score = coverage + overlap / len(query_terms | sentence_terms)
                    ranked.append((score, sentence.strip()))
        ranked.sort(reverse=True)
        return [sentence for _, sentence in ranked[:limit]]
