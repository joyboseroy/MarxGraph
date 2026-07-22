"""Detect (and optionally remove) near-duplicate works within the same author.

Different EPUB editions of the same book (a single-file edition vs a multi-volume
split, or two independently uploaded scans) get different work_ids and different
passage segmentation, so exact text/hash matching on passages won't catch them.
Instead this compares a normalized snippet of each work's opening text against
every other work by the same author — the actual prose should match closely even
if paragraph/chapter boundaries don't.

Default mode is report-only: nothing is deleted until you review the report and
re-run with --apply.

    python src/dedupe_works.py                      # writes duplicate_works_report.csv
    python src/dedupe_works.py --apply               # also removes the flagged duplicates
                                                      # (keeps the work with more passages
                                                      #  from each duplicate pair/group)
"""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

SNIPPET_WORDS = 400        # how much opening text to compare per work
SIMILARITY_THRESHOLD = 0.75  # SequenceMatcher ratio above this = likely same book


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower())
    return re.sub(r"[^a-z0-9 ]", "", text)


def work_snippet(passages: pd.DataFrame, work_id: str) -> str:
    sub = passages[passages.work_id == work_id].sort_values("seq")
    words, out = 0, []
    for text in sub.text:
        out.append(text)
        words += len(text.split())
        if words >= SNIPPET_WORDS:
            break
    return normalize(" ".join(out))[: SNIPPET_WORDS * 7]  # rough char cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet")
    ap.add_argument("--apply", action="store_true",
                     help="actually remove flagged duplicates (default: report only)")
    args = ap.parse_args()
    p = Path(args.parquet)

    works = pd.read_parquet(p / "works.parquet")
    passages = pd.read_parquet(p / "passages.parquet")

    snippets = {wid: work_snippet(passages, wid) for wid in works.work_id}

    rows = []
    checked = set()
    for author, grp in works.groupby("author"):
        wids = list(grp.work_id)
        for i in range(len(wids)):
            for j in range(i + 1, len(wids)):
                a, b = wids[i], wids[j]
                if not snippets[a] or not snippets[b]:
                    continue
                ratio = SequenceMatcher(None, snippets[a], snippets[b]).ratio()
                if ratio >= SIMILARITY_THRESHOLD:
                    na = int(works.loc[works.work_id == a, "n_passages"].iloc[0])
                    nb = int(works.loc[works.work_id == b, "n_passages"].iloc[0])
                    keep, drop = (a, b) if na >= nb else (b, a)
                    rows.append({"author": author, "work_a": a, "title_a":
                                 works.loc[works.work_id == a, "title"].iloc[0],
                                 "work_b": b, "title_b":
                                 works.loc[works.work_id == b, "title"].iloc[0],
                                 "similarity": round(ratio, 3),
                                 "keep": keep, "drop": drop})

    report_path = p / "duplicate_works_report.csv"
    report = pd.DataFrame(rows)
    report.to_csv(report_path, index=False)
    print(f"{len(rows)} likely-duplicate pairs found -> {report_path}")
    if not rows:
        return
    print(report[["author", "title_a", "title_b", "similarity", "drop"]].to_string(index=False))

    if not args.apply:
        print("\nReport-only run. Review the CSV, then re-run with --apply to remove them.")
        return

    drop_ids = set(report["drop"])
    print(f"\n--apply set: removing {len(drop_ids)} duplicate work(s): {sorted(drop_ids)}")

    for fname, col in [("works.parquet", "work_id"), ("passages.parquet", "work_id"),
                        ("concept_mentions.parquet", "work_id"),
                        ("claims.parquet", "work_id"), ("references.parquet", "work_id")]:
        fpath = p / fname
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        before = len(df)
        df = df[~df[col].isin(drop_ids)]
        df.to_parquet(fpath, index=False)
        print(f"  {fname}: {before} -> {len(df)} rows")

    # also prune extractions.jsonl so re-running extract_claims.py doesn't think
    # the dropped work's passages still need (or already have) extraction
    ext_path = p / "extractions.jsonl"
    if ext_path.exists():
        surviving_passages = set(pd.read_parquet(p / "passages.parquet").passage_id)
        lines = [l for l in ext_path.read_text().splitlines()
                 if l.strip() and json.loads(l)["passage_id"] in surviving_passages]
        ext_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  extractions.jsonl: pruned to {len(lines)} surviving passages")


if __name__ == "__main__":
    main()
