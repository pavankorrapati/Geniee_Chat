# from __future__ import annotations

# from html.parser import HTMLParser
# import json
# import re
# import time
# from urllib.parse import quote, urlencode
# from urllib.request import Request, urlopen


# class _TextParser(HTMLParser):
#     def __init__(self):
#         super().__init__()
#         self.parts = []
#         self.skip_depth = 0

#     def handle_starttag(self, tag, attrs):
#         if tag in {"script", "style", "noscript", "svg"}:
#             self.skip_depth += 1

#     def handle_endtag(self, tag):
#         if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
#             self.skip_depth -= 1

#     def handle_data(self, data):
#         if not self.skip_depth:
#             value = " ".join(data.split())
#             if value:
#                 self.parts.append(value)


# class WebRetriever:
#     """Dependency-free web search and extractive answer retrieval."""

#     def __init__(self, timeout=8, max_pages=3):
#         self.timeout = timeout
#         self.max_pages = max_pages
#         self._cache = {}

#     def _get(self, url):
#         request = Request(url, headers={"User-Agent": "Geniee/1.0 research client"})
#         with urlopen(request, timeout=self.timeout) as response:
#             return response.read().decode("utf-8", errors="ignore")

#     def search(self, query):
#         results = []
#         search_urls = [
#             "https://html.duckduckgo.com/html/?q=" + quote(query),
#             "https://www.bing.com/search?q=" + quote(query),
#         ]
#         for search_url in search_urls:
#             try:
#                 html = self._get(search_url)
#             except Exception:
#                 continue
#             for match in re.finditer(r"href=\"(https?://[^\"]+)\"", html):
#                 url = match.group(1)
#                 if "bing.com" not in url and "duckduckgo.com" not in url and url not in results:
#                     results.append(url)
#                 if len(results) >= self.max_pages:
#                     return results
#         return results

#     def _wikipedia(self, query):
#         search_query = re.sub(
#             r"\b(what|is|are|who|explain|describe|tell|me|about)\b",
#             " ",
#             query.lower(),
#         )
#         search_query = " ".join(search_query.split()).strip(" ?.!\\t") or query
#         replacements = {
#             "zjanasena": "Janasena Party",
#             "janasena party cheif in andhras pradesh": "Janasena Party Andhra Pradesh chief",
#             "megastar telugu film industry": "Chiranjeevi Mega Star Telugu actor",
#             "tdp": "Telugu Desam Party",
#         }
#         search_query = replacements.get(search_query, search_query)
#         direct_titles = {
#             "telugu desam party": "Telugu Desam Party",
#             "janasena party andhra pradesh chief": "Janasena Party",
#             "chiranjeevi mega star telugu actor": "Chiranjeevi",
#             "orbital mechanics": "Orbital mechanics",
#         }
#         direct_title = direct_titles.get(search_query)
#         if direct_title:
#             try:
#                 summary_url = (
#                     "https://en.wikipedia.org/api/rest_v1/page/summary/"
#                     + quote(direct_title.replace(" ", "_"))
#                 )
#                 summary = json.loads(self._get(summary_url))
#                 extract = " ".join(summary.get("extract", "").split())
#                 if extract:
#                     return [(
#                         extract,
#                         f"https://en.wikipedia.org/wiki/{quote(direct_title.replace(' ', '_'))}",
#                     )]
#             except (OSError, ValueError, KeyError, TypeError):
#                 pass
#         cache_key = search_query.casefold()
#         cached = self._cache.get(cache_key)
#         if cached and time.monotonic() - cached[0] < 300:
#             return cached[1]
#         search_url = "https://en.wikipedia.org/w/api.php?" + urlencode(
#             {
#                 "action": "query",
#                 "list": "search",
#                 "srsearch": search_query,
#                 "srlimit": self.max_pages,
#                 "format": "json",
#                 "origin": "*",
#             }
#         )
#         data = json.loads(self._get(search_url))
#         results = []
#         titles = []
#         query_terms = set(re.findall(r"[a-z0-9]+", search_query))
#         for item in data.get("query", {}).get("search", []):
#             title = item.get("title")
#             if not title:
#                 continue
#             titles.append(title)
#             snippet = " ".join(re.sub(r"<[^>]+>", "", item.get("snippet", "")).split())
#             title_terms = set(re.findall(r"[a-z0-9]+", title.lower()))
#             score = len(query_terms & title_terms) / max(1, len(query_terms))
#             if snippet:
#                 results.append((score, snippet, f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"))
#         if titles:
#             try:
#                 extract_url = "https://en.wikipedia.org/w/api.php?" + urlencode(
#                     {
#                         "action": "query", "prop": "extracts", "exintro": "1",
#                         "explaintext": "1", "titles": "|".join(titles),
#                         "format": "json", "origin": "*",
#                     }
#                 )
#                 page_data = json.loads(self._get(extract_url))
#                 for page in page_data.get("query", {}).get("pages", {}).values():
#                     title = page.get("title", "")
#                     extract = " ".join(page.get("extract", "").split())
#                     if extract:
#                         title_terms = set(re.findall(r"[a-z0-9]+", title.lower()))
#                         score = len(query_terms & title_terms) / max(1, len(query_terms))
#                         results.append((score + 0.1, extract, f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"))
#             except (OSError, ValueError, KeyError, TypeError):
#                 pass
#         results.sort(key=lambda item: item[0], reverse=True)
#         value = [(extract, url) for _, extract, url in results]
#         self._cache[cache_key] = (time.monotonic(), value)
#         return value

#     def retrieve(self, query):
#         sentences = []
#         query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
#         try:
#             wikipedia_results = self._wikipedia(query)
#         except (OSError, ValueError, KeyError, TypeError):
#             wikipedia_results = []
#         for text, url in wikipedia_results[:1]:
#             for sentence in re.split(r"(?<=[.!?])\s+", text):
#                 sentence = sentence.strip()
#                 if 40 <= len(sentence) <= 500:
#                     sentences.append((1.0, sentence, url))
#                     if len(sentences) >= 3:
#                         break
#         if sentences:
#             sentences.sort(key=lambda item: item[0], reverse=True)
#             return [(sentence, url) for _, sentence, url in sentences[:3]]

#         for url in self.search(query):
#             try:
#                 page = self._get(url)
#             except Exception:
#                 continue
#             parser = _TextParser()
#             parser.feed(page)
#             text = " ".join(parser.parts)
#             for sentence in re.split(r"(?<=[.!?])\s+", text):
#                 sentence = sentence.strip()
#                 if 40 <= len(sentence) <= 500:
#                     terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
#                     overlap = len(query_terms & terms)
#                     if overlap:
#                         sentences.append((overlap / max(1, len(terms)), sentence, url))
#         sentences.sort(key=lambda item: item[0], reverse=True)
#         unique = []
#         seen = set()
#         for _, sentence, url in sentences:
#             key = sentence.lower()
#             if key not in seen:
#                 seen.add(key)
#                 unique.append((sentence, url))
#             if len(unique) >= 3:
#                 break
#         return unique

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
    """Dependency-free web search and extractive answer retrieval with strict query relevance filtering."""

    def __init__(self, timeout=8, max_pages=3):
        self.timeout = timeout
        self.max_pages = max_pages
        self._cache = {}

    def _get(self, url):
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _clean_query(self, query: str) -> str:
        """Strips conversational fluff and stop phrases to maximize search engine accuracy."""
        stop_phrases = [
            "what is", "what was", "what are", "tell me about", "can you tell me",
            "yesterday's", "yesterdays", "today's", "todays", "show me",
            "results for", "result of", "results of", "results", "result",
            "details of", "information on", "who is", "who was"
        ]
        cleaned = query.lower()
        for phrase in stop_phrases:
            cleaned = cleaned.replace(phrase, " ")
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = " ".join(cleaned.split()).strip()
        return cleaned if cleaned else query

    def _is_realtime_query(self, query: str) -> bool:
        """Detects if a query requires real-time/live web search rather than static encyclopedia pages."""
        realtime_keywords = {
            "today", "todays", "today's", "yesterday", "yesterdays", "yesterday's",
            "match", "score", "vs", "versus", "price", "rate", "latest", "news",
            "current", "live", "weather", "stock", "t20", "ipl", "cricket"
        }
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        return bool(tokens.intersection(realtime_keywords))

    def _get_query_terms(self, query: str) -> set[str]:
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "what", "where", "when", "vs"
        }
        words = re.findall(r"[a-z0-9]+", query.lower())
        return {w for w in words if w not in stop_words and len(w) > 1}

    def search_duckduckgo_snippets(self, query: str) -> list[tuple[str, str]]:
        """Directly extracts high-relevance search result snippets from DuckDuckGo HTML."""
        cleaned_query = self._clean_query(query)
        search_url = "https://html.duckduckgo.com/html/?q=" + quote(cleaned_query)
        results = []
        try:
            html = self._get(search_url)
            matches = re.findall(
                r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
                html,
                re.DOTALL | re.IGNORECASE,
            )
            urls = re.findall(
                r'<a[^>]*class="[^"]*result__url[^"]*"[^>]*href="([^"]+)"',
                html,
                re.IGNORECASE,
            )
            for i, match in enumerate(matches):
                snippet = " ".join(re.sub(r"<[^>]+>", "", match).split())
                url = urls[i].strip() if i < len(urls) else "https://duckduckgo.com"
                if not url.startswith("http"):
                    url = "https://" + url.lstrip("/")
                if snippet:
                    results.append((snippet, url))
        except Exception:
            pass
        return results

    def search_urls(self, query: str) -> list[str]:
        results = []
        cleaned_query = self._clean_query(query)
        search_urls = [
            "https://html.duckduckgo.com/html/?q=" + quote(cleaned_query),
            "https://www.bing.com/search?q=" + quote(cleaned_query),
        ]
        for search_url in search_urls:
            try:
                html = self._get(search_url)
            except Exception:
                continue
            for match in re.finditer(r"href=\"(https?://[^\"]+)\"", html):
                url = match.group(1)
                if (
                    "bing.com" not in url
                    and "duckduckgo.com" not in url
                    and "microsoft.com" not in url
                    and url not in results
                ):
                    results.append(url)
                if len(results) >= self.max_pages:
                    return results
        return results

    def _wikipedia(self, query: str):
        search_query = self._clean_query(query)
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
        query_terms = self._get_query_terms(search_query)

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

    def retrieve(self, query: str) -> list[tuple[str, str]]:
        query_terms = self._get_query_terms(query)
        if not query_terms:
            return []

        candidates = []
        is_realtime = self._is_realtime_query(query)

        # 1. DuckDuckGo Snippets (Preferred for Real-time, Sports Scores, and Price queries)
        ddg_snippets = self.search_duckduckgo_snippets(query)
        for snippet, url in ddg_snippets:
            snippet_terms = set(re.findall(r"[a-z0-9]+", snippet.lower()))
            overlap = len(query_terms & snippet_terms)
            coverage = overlap / len(query_terms)
            if coverage >= 0.35:
                candidates.append((coverage + (0.2 if is_realtime else 0.0), snippet, url))

        # 2. Wikipedia Search (Only if NOT real-time or if DDG returned no high-coverage candidates)
        if not is_realtime or not candidates:
            try:
                wikipedia_results = self._wikipedia(query)
            except (OSError, ValueError, KeyError, TypeError):
                wikipedia_results = []

            for text, url in wikipedia_results:
                text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
                overlap = len(query_terms & text_terms)
                coverage = overlap / len(query_terms)
                # Enforce coverage guard: discard Wikipedia pages missing core query terms
                if coverage >= 0.40:
                    for sentence in re.split(r"(?<=[.!?])\s+", text):
                        sentence = sentence.strip()
                        if 40 <= len(sentence) <= 500:
                            s_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                            s_overlap = len(query_terms & s_terms)
                            s_coverage = s_overlap / len(query_terms)
                            if s_coverage >= 0.30:
                                candidates.append((s_coverage, sentence, url))

        # 3. Full Web Page Crawler Fallback
        if not candidates:
            for url in self.search_urls(query):
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
                        coverage = overlap / len(query_terms)
                        if coverage >= 0.35:
                            candidates.append((coverage, sentence, url))

        candidates.sort(key=lambda item: item[0], reverse=True)

        unique = []
        seen = set()
        for _, sentence, url in candidates:
            key = sentence.lower()
            if key not in seen:
                seen.add(key)
                unique.append((sentence, url))
            if len(unique) >= 3:
                break

        return unique