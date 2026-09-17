from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import time


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEB_DIR = PROJECT_ROOT / "data" / "web"
DOWNLOAD_DIR = WEB_DIR / "downloaded"


# ================================================================
# PYTHON 3.14 DOCUMENTATION
# ================================================================

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


USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36 "
    "Geniee-Web-Corpus/1.0"
)


# ================================================================
# SAFE FILE NAME
# ================================================================

def safe_filename(url: str) -> str:
    parsed = urlparse(url)

    path = parsed.path.strip("/")

    if not path:
        path = "index"

    filename = path.replace("/", "__")

    if filename.endswith(".html"):
        filename = filename[:-5]

    return filename + ".html"


# ================================================================
# DOWNLOAD ONE PAGE
# ================================================================

def download_page(url: str, output_file: Path) -> bool:

    print()
    print("Downloading:")
    print(f"  URL : {url}")
    print(f"  OUT : {output_file}")

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:

        with urlopen(request, timeout=60) as response:
            data = response.read()

        output_file.write_bytes(data)

        print(
            f"  SUCCESS - "
            f"{len(data):,} bytes"
        )

        return True

    except Exception as exc:

        print(
            f"  FAILED - "
            f"{type(exc).__name__}: {exc}"
        )

        return False


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 72)
    print("GENIEE PYTHON 3.14 WEB DATA DOWNLOADER")
    print("=" * 72)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output       : {DOWNLOAD_DIR}")
    print()

    # Create directory BEFORE downloading.
    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Documentation pages to download: "
        f"{len(DOCUMENTATION_URLS)}"
    )

    successful = 0
    failed = 0

    for index, url in enumerate(
        DOCUMENTATION_URLS,
        start=1,
    ):

        print()
        print("-" * 72)
        print(
            f"[{index}/{len(DOCUMENTATION_URLS)}]"
        )

        filename = safe_filename(url)

        output_file = DOWNLOAD_DIR / filename

        if download_page(
            url,
            output_file,
        ):
            successful += 1
        else:
            failed += 1

        # Small delay between requests.
        if index < len(DOCUMENTATION_URLS):
            time.sleep(1)

    print()
    print("=" * 72)
    print("DOWNLOAD COMPLETE")
    print("=" * 72)

    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")
    print(f"Output     : {DOWNLOAD_DIR}")

    print()

    # Verify that files really exist.
    downloaded_files = list(
        DOWNLOAD_DIR.glob("*.html")
    )

    print(
        f"HTML files actually present: "
        f"{len(downloaded_files)}"
    )

    if downloaded_files:

        print()
        print("Downloaded files:")

        for file in downloaded_files:
            print(
                f"  {file.name} "
                f"({file.stat().st_size:,} bytes)"
            )

    else:

        print()
        print(
            "ERROR: No HTML files were created."
        )

    print()
    print("CHECKPOINT SAFETY:")
    print(
        "  No Geniee checkpoint was read or modified."
    )


# ================================================================
# SCRIPT ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()