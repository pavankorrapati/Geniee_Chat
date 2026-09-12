from __future__ import annotations

from html.parser import HTMLParser
import json
import re
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            value = " ".join(data.split())
            if value:
                self.parts.append(value)


class WebRetriever:
    """Dependency-free web search and extractive answer retrieval."""

    def __init__(self, timeout=8, max_pages=3):
        self.timeout = timeout
        self.max_pages = max_pages
        self._cache = {}

    def _get(self, url):
        request = Request(url, headers={"User-Agent": "Geniee/1.0 research client"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="ignore")

    def search(self, query):
        results = []
        search_urls = [
            "https://html.duckduckgo.com/html/?q=" + quote(query),
            "https://www.bing.com/search?q=" + quote(query),
        ]
        for search_url in search_urls:
            try:
                html = self._get(search_url)
            except Exception:
                continue
            for match in re.finditer(r"href=\"(https?://[^\"]+)\"", html):
                url = match.group(1)
                if "bing.com" not in url and "duckduckgo.com" not in url and url not in results:
                    results.append(url)
                if len(results) >= self.max_pages:
                    return results
        return results

    def _wikipedia(self, query):
        search_query = re.sub(
            r"\b(what|is|are|who|explain|describe|tell|me|about)\b",
            " ",
            query.lower(),
        )
        search_query = " ".join(search_query.split()).strip(" ?.!\\t") or query
        replacements = {
            "zjanasena": "Janasena Party",
            "janasena party cheif in andhras pradesh": "Janasena Party Andhra Pradesh chief",
            "megastar telugu film industry": "Chiranjeevi Mega Star Telugu actor",
            "tdp": "Telugu Desam Party",
        }
        search_query = replacements.get(search_query, search_query)
        direct_titles = {
            "telugu desam party": "Telugu Desam Party",
            "janasena party andhra pradesh chief": "Janasena Party",
            "chiranjeevi mega star telugu actor": "Chiranjeevi",
            "orbital mechanics": "Orbital mechanics",
        }
        direct_title = direct_titles.get(search_query)
        if direct_title:
            try:
                summary_url = (
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    + quote(direct_title.replace(" ", "_"))
                )
                summary = json.loads(self._get(summary_url))
                extract = " ".join(summary.get("extract", "").split())
                if extract:
                    return [(
                        extract,
                        f"https://en.wikipedia.org/wiki/{quote(direct_title.replace(' ', '_'))}",
                    )]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        cache_key = search_query.casefold()
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 300:
            return cached[1]
        search_url = "https://en.wikipedia.org/w/api.php?" + urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": self.max_pages,
                "format": "json",
                "origin": "*",
            }
        )
        data = json.loads(self._get(search_url))
        results = []
        titles = []
        query_terms = set(re.findall(r"[a-z0-9]+", search_query))
        for item in data.get("query", {}).get("search", []):
            title = item.get("title")
            if not title:
                continue
            titles.append(title)
            snippet = " ".join(re.sub(r"<[^>]+>", "", item.get("snippet", "")).split())
            title_terms = set(re.findall(r"[a-z0-9]+", title.lower()))
            score = len(query_terms & title_terms) / max(1, len(query_terms))
            if snippet:
                results.append((score, snippet, f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"))
        if titles:
            try:
                extract_url = "https://en.wikipedia.org/w/api.php?" + urlencode(
                    {
                        "action": "query", "prop": "extracts", "exintro": "1",
                        "explaintext": "1", "titles": "|".join(titles),
                        "format": "json", "origin": "*",
                    }
                )
                page_data = json.loads(self._get(extract_url))
                for page in page_data.get("query", {}).get("pages", {}).values():
                    title = page.get("title", "")
                    extract = " ".join(page.get("extract", "").split())
                    if extract:
                        title_terms = set(re.findall(r"[a-z0-9]+", title.lower()))
                        score = len(query_terms & title_terms) / max(1, len(query_terms))
                        results.append((score + 0.1, extract, f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        results.sort(key=lambda item: item[0], reverse=True)
        value = [(extract, url) for _, extract, url in results]
        self._cache[cache_key] = (time.monotonic(), value)
        return value

    def retrieve(self, query):
        sentences = []
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        try:
            wikipedia_results = self._wikipedia(query)
        except (OSError, ValueError, KeyError, TypeError):
            wikipedia_results = []
        for text, url in wikipedia_results[:1]:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = sentence.strip()
                if 40 <= len(sentence) <= 500:
                    sentences.append((1.0, sentence, url))
                    if len(sentences) >= 3:
                        break
        if sentences:
            sentences.sort(key=lambda item: item[0], reverse=True)
            return [(sentence, url) for _, sentence, url in sentences[:3]]

        for url in self.search(query):
            try:
                page = self._get(url)
            except Exception:
                continue
            parser = _TextParser()
            parser.feed(page)
            text = " ".join(parser.parts)
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = sentence.strip()
                if 40 <= len(sentence) <= 500:
                    terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                    overlap = len(query_terms & terms)
                    if overlap:
                        sentences.append((overlap / max(1, len(terms)), sentence, url))
        sentences.sort(key=lambda item: item[0], reverse=True)
        unique = []
        seen = set()
        for _, sentence, url in sentences:
            key = sentence.lower()
            if key not in seen:
                seen.add(key)
                unique.append((sentence, url))
            if len(unique) >= 3:
                break
        return unique
