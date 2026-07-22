"""Download the MarxGraph V1 corpus from the Marxists Internet Archive.

Uses MIA's official EPUB collection instead of crawling HTML pages: far cleaner
text, and roughly 40 requests instead of thousands against a volunteer-run site.

Run on your own machine (not sandboxed):
    python src/download.py --config config/works.yaml
    python src/download.py --config config/works.yaml --skip-index   # direct links only
"""

import argparse
import hashlib
import json
import re
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup

BASE = "https://www.marxists.org"


def polite_get(session: requests.Session, url: str, delay: float) -> requests.Response:
    time.sleep(delay)
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp


def robots_allows(url: str, user_agent: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urljoin(BASE, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        return True  # robots.txt unreadable; proceed but stay polite
    return rp.can_fetch(user_agent, url)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def title_from_url(url: str) -> str:
    stem = url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    words = re.split(r"[-_]+", stem)
    return " ".join(w.capitalize() for w in words if w)


def harvest_index(session, page: dict, delay: float) -> list[dict]:
    """Collect .epub links from an MIA index page, filtered by keywords."""
    print(f"[index] {page['url']}")
    resp = polite_get(session, page["url"], delay)
    soup = BeautifulSoup(resp.text, "html.parser")
    keywords = [str(k).lower() for k in page.get("include_keywords", [])]
    seen_urls = set()
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".epub"):
            continue
        full = urljoin(page["url"], href)
        if full in seen_urls:
            continue
        fname = full.rsplit("/", 1)[-1].lower()
        if keywords and not any(k in fname for k in keywords):
            continue
        seen_urls.add(full)
        link_text = a.get_text(strip=True)
        # MIA link text is often just the format label ("epub"), not a real title
        title = link_text if link_text and link_text.lower() not in {"epub", "mobi", "pdf", "prc", "azw", "azw3"} \
            else title_from_url(full)
        found.append({
            "author": page["author"],
            "tradition": page["tradition"],
            "title": title,
            "year": None,  # fill manually in the manifest after review
            "url": full,
            "license": "verify",
            "harvested_from": page["url"],
        })
    print(f"        -> {len(found)} matching EPUBs")
    return found


def download_work(session, work: dict, out_dir: Path, delay: float) -> dict:
    # derive from the URL, not the (possibly generic/duplicated) title, so every
    # distinct work gets a distinct filename and work_id
    url_stem = slugify(work["url"].rsplit("/", 1)[-1].rsplit(".", 1)[0])[:50]
    fname = f"{work['author']}__{url_stem}.epub"
    dest = out_dir / fname
    if dest.exists():
        print(f"[skip ] {fname} (cached)")
    else:
        print(f"[fetch] {work['url']}")
        resp = polite_get(session, work["url"], delay)
        dest.write_bytes(resp.content)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {**work, "local_path": str(dest), "sha256": sha,
            "work_id": f"{work['author']}_{url_stem}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/works.yaml")
    ap.add_argument("--skip-index", action="store_true",
                    help="only download directly listed works")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    settings = cfg["settings"]
    delay = float(settings.get("delay_seconds", 5))
    out_dir = Path(settings.get("out_dir", "data/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = settings["user_agent"]

    if not robots_allows(BASE + "/ebooks/", settings["user_agent"]):
        raise SystemExit("robots.txt disallows fetching; aborting.")

    works = list(cfg.get("works", []))
    if not args.skip_index:
        for page in cfg.get("index_pages", []):
            works.extend(harvest_index(session, page, delay))

    manifest, failed = [], []
    for work in works:
        try:
            manifest.append(download_work(session, work, out_dir, delay))
        except Exception as exc:  # keep going; log failures
            print(f"[FAIL ] {work['url']}: {exc}")
            failed.append({**work, "error": str(exc)})

    Path(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if failed:
        Path(out_dir / "failed.json").write_text(json.dumps(failed, indent=2))
    print(f"\nDone. {len(manifest)} works downloaded, {len(failed)} failed.")
    print(f"Manifest: {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
