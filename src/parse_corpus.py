"""Parse downloaded EPUBs into works.parquet and passages.parquet.

    python src/parse_corpus.py --manifest data/raw/manifest.json --out data/parquet

works.parquet:    work_id, author, title, year, tradition, language, source_url, license, sha256, n_passages
passages.parquet: passage_id, work_id, chapter, seq, text, n_words
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
import ebooklib

# EPUB boilerplate chapters to drop
SKIP_TITLES = re.compile(
    r"(table of contents|contents|title page|copyright|colophon|about this book|"
    r"cover|index of names|transcriber)", re.I)

MIN_PASSAGE_WORDS = 40      # drop headers/fragments
TARGET_PASSAGE_WORDS = 220  # merge short paragraphs up to roughly this size


def chapter_text(item) -> tuple[str, list[str]]:
    """Return (chapter_title, list_of_paragraphs) for one EPUB document item."""
    soup = BeautifulSoup(item.get_content(), "html.parser")
    for tag in soup(["script", "style", "sup", "table"]):
        tag.decompose()
    title_tag = soup.find(["h1", "h2", "h3"])
    title = title_tag.get_text(" ", strip=True) if title_tag else item.get_name()
    paras = []
    for p in soup.find_all(["p", "blockquote"]):
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
        # drop footnote markers like [1] and MIA transcription notes
        text = re.sub(r"\[\d+\]", "", text).strip()
        if text:
            paras.append(text)
    return title, paras


def segment(paras: list[str]) -> list[str]:
    """Merge paragraphs into passages of roughly TARGET_PASSAGE_WORDS words."""
    passages, buf, buf_words = [], [], 0
    for p in paras:
        w = len(p.split())
        buf.append(p)
        buf_words += w
        if buf_words >= TARGET_PASSAGE_WORDS:
            passages.append("\n".join(buf))
            buf, buf_words = [], 0
    if buf and buf_words >= MIN_PASSAGE_WORDS:
        passages.append("\n".join(buf))
    return [p for p in passages if len(p.split()) >= MIN_PASSAGE_WORDS]


def parse_work(entry: dict) -> tuple[dict, list[dict]]:
    book = epub.read_epub(entry["local_path"])
    passages = []
    seq = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        title, paras = chapter_text(item)
        if SKIP_TITLES.search(title or ""):
            continue
        for text in segment(paras):
            seq += 1
            passages.append({
                "passage_id": f"{entry['work_id']}__p{seq:04d}",
                "work_id": entry["work_id"],
                "chapter": title[:120],
                "seq": seq,
                "text": text,
                "n_words": len(text.split()),
            })
    work_row = {
        "work_id": entry["work_id"],
        "author": entry["author"],
        "title": entry["title"],
        "year": entry.get("year"),
        "tradition": entry.get("tradition"),
        "language": "en",
        "source_url": entry["url"],
        "license": entry.get("license", "verify"),
        "sha256": entry["sha256"],
        "n_passages": len(passages),
    }
    return work_row, passages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/manifest.json")
    ap.add_argument("--out", default="data/parquet")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    seen, deduped = {}, []
    for entry in manifest:
        wid = entry["work_id"]
        if wid in seen:
            print(f"[warn] duplicate work_id '{wid}' ({entry['url']}) — skipping, "
                  f"kept {seen[wid]}")
            continue
        seen[wid] = entry["url"]
        deduped.append(entry)
    manifest = deduped
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_works, all_passages = [], []
    for entry in manifest:
        try:
            work_row, passages = parse_work(entry)
            all_works.append(work_row)
            all_passages.extend(passages)
            print(f"[ok  ] {entry['work_id']}: {len(passages)} passages")
        except Exception as exc:
            print(f"[FAIL] {entry['work_id']}: {exc}")

    pd.DataFrame(all_works).to_parquet(out / "works.parquet", index=False)
    pd.DataFrame(all_passages).to_parquet(out / "passages.parquet", index=False)
    total = sum(w["n_passages"] for w in all_works)
    print(f"\n{len(all_works)} works, {total} passages -> {out}")


if __name__ == "__main__":
    main()
