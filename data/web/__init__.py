"""
GENIEE WEB DATA DOWNLOADER
==========================

Downloads the official Python 3.14 documentation pages that are useful
for building a Python knowledge corpus.

IMPORTANT:
- This script does NOT modify any Geniee checkpoint.
- Raw downloaded HTML is stored under data/web/downloaded/.
- Only the Python documentation is downloaded here.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import time


# ----------------------------------------------------------------------
# PROJECT PATHS
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEB_DIR = PROJECT_ROOT / "data" / "web"
DOWNLOAD_DIR = WEB_DIR / "downloaded"


# ----------------------------------------------------------------------
# SOURCE
# ----------------------------------------------------------------------

BASE_URL = "https://docs.python.org/3.14/"

# Important Python 3.14 documentation pages.
DOCUMENTATION_URLS = [
    "https://docs.python.org/3.14/whatsnew/3.14.html",
    "https://docs.python.org/3.14/tutorial/index.html",
    "https://docs.python.org/3.14/library/index.html",
    "https://docs.python.org/3.14/reference/index.html",
    "https://docs.python.org/3.14/howto/index.html",
    "https://docs.python.org/3.14/using/index.html",
    "https://docs.python.org/3.14/installing/index.html",
    "https://docs.python.org/3.14/faq/index.html",
]


# ----------------------------------------------------------------------
# HTTP SETTINGS
# ----------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36 "
    "Geniee-Web-Corpus/1.0"
)


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def safe_filename(url: str) -> str:
    """Convert a URL into a safe local filename."""

    parsed = urlparse(url)

    path = parsed.path.strip("/")

    if not path:
        path = "index"

    filename = path.replace("/", "__")

    if filename.endswith(".html"):
        filename = filename[:-5]

    return f"{filename}.html"


def download_url(url: str, output_path: Path) -> bool:
    """Download one URL."""

    print(f"\nDownloading:")
    print(f"  {url}")
    print(f"  -> {output_path}")

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            content = response.read()

        output_path.write_bytes(content)

        print(f"  OK - {len(content):,} bytes")
        return True

    except Exception as exc:
        print(f"  FAILED - {type(exc).__name__}: {exc}")
        return False


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("GENIEE PYTHON 3.14 WEB DATA DOWNLOADER")
    print("=" * 72)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Download dir : {DOWNLOAD_DIR}")
    print(f"Source       : {BASE_URL}")
    print()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for index, url in enumerate(DOCUMENTATION_URLS, start=1):

        filename = safe_filename(url)
        output_path = DOWNLOAD_DIR / filename

        print("-" * 72)
        print(f"[{index}/{len(DOCUMENTATION_URLS)}]")

        if download_url(url, output_path):
            successful += 1
        else:
            failed += 1

        # Be polite to the documentation server.
        if index < len(DOCUMENTATION_URLS):
            time.sleep(1)

    print()
    print("=" * 72)
    print("DOWNLOAD COMPLETE")
    print("=" * 72)

    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")
    print(f"Files      : {DOWNLOAD_DIR}")

    if failed:
        print()
        print("WARNING: Some pages could not be downloaded.")
        print("You can run this script again.")

    print()
    print("Checkpoint status:")
    print("  NO CHECKPOINT WAS MODIFIED.")
    print("  Existing Geniee checkpoints remain untouched.")


if __name__ == "__main__":
    main()