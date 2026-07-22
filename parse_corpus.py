"""Stage 3: LLM extraction of concept mentions, claims, and explicit references.

    export ANTHROPIC_API_KEY=...     # or GROQ_API_KEY with MARXGRAPH_BACKEND=groq
    python src/extract_claims.py --passages data/parquet/passages.parquet \
        --works data/parquet/works.parquet --out data/parquet --limit 200 --workers 8

Resumable: already-processed passage_ids in the output jsonl are skipped, so you can
run it in cheap batches (--limit) and stop/restart freely. Start with --limit 50 and
read the output before spending money on the full corpus.

Runs requests concurrently (--workers, default 8) since this is a network-bound
workload (waiting on the LLM API), not a compute-bound one — a GPU does not help
here regardless of platform; only concurrency does. Raise --workers if your
provider's rate limit allows it, lower it if you start seeing 429s pile up.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

import llm_backend

PROMPT = Path("prompts/claim_extraction.txt").read_text()


def seed_concept_block() -> str:
    seeds = yaml.safe_load(Path("config/seed_concepts.yaml").read_text())["concepts"]
    lines = [f"- {slug}: {v['label']} (aliases: {', '.join(v['aliases']) or 'none'})"
             for slug, v in seeds.items()]
    return "SEED CONCEPT LIST:\n" + "\n".join(lines)


def extract_one(passage: dict, work: dict, seed_block: str) -> dict:
    user_msg = (
        f"{seed_block}\n\n"
        f"AUTHOR: {work['author']}\nWORK: {work['title']}\nYEAR: {work['year']}\n"
        f"PASSAGE (id {passage['passage_id']}):\n{passage['text']}"
    )
    data = llm_backend.call_json(PROMPT, user_msg, max_tokens=3000)
    data["passage_id"] = passage["passage_id"]
    data["work_id"] = passage["work_id"]
    data["model"] = llm_backend.MODEL
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="data/parquet/passages.parquet")
    ap.add_argument("--works", default="data/parquet/works.parquet")
    ap.add_argument("--out", default="data/parquet")
    ap.add_argument("--limit", type=int, default=None, help="max NEW passages this run")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent requests (network-bound workload; tune to your rate limit)")
    args = ap.parse_args()

    passages = pd.read_parquet(args.passages)
    works = pd.read_parquet(args.works)
    dupes = works[works.duplicated("work_id", keep=False)]
    if not dupes.empty:
        print(f"[warn] {dupes.work_id.nunique()} duplicate work_id(s) in works.parquet; "
              f"keeping first occurrence of each")
        works = works.drop_duplicates("work_id", keep="first")
    works = works.set_index("work_id").to_dict("index")
    out = Path(args.out)
    raw_path = out / "extractions.jsonl"

    done = set()
    if raw_path.exists():
        with raw_path.open() as f:
            done = {json.loads(line)["passage_id"] for line in f if line.strip()}
    todo = passages[~passages.passage_id.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already extracted; {len(todo)} to do this run.")

    print(f"backend={llm_backend.BACKEND} model={llm_backend.MODEL} workers={args.workers}")
    seed_block = seed_concept_block()

    write_lock = threading.Lock()
    ok_count, fail_count = 0, 0

    def worker(row: dict) -> tuple[str, dict | None, str | None]:
        time.sleep(random.uniform(0, 0.4))  # desynchronize the pool to avoid bursty 429s
        work = works[row["work_id"]]
        try:
            return row["passage_id"], extract_one(row, work, seed_block), None
        except Exception as exc:
            return row["passage_id"], None, str(exc)

    todo_records = todo.to_dict("records")
    with raw_path.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, row) for row in todo_records]
        for i, fut in enumerate(as_completed(futures), 1):
            passage_id, result, err = fut.result()
            if err:
                fail_count += 1
                print(f"[warn] failed for {passage_id}: {err}")
            else:
                ok_count += 1
                with write_lock:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
            if i % 50 == 0:
                print(f"  ...{i}/{len(todo_records)} (ok={ok_count} fail={fail_count})")

    # flatten jsonl -> tidy parquet tables
    mentions, claims, references = [], [], []
    with raw_path.open() as f:
        for line in f:
            d = json.loads(line)
            base = {"passage_id": d["passage_id"], "work_id": d["work_id"], "model": d.get("model")}
            for m in d.get("concept_mentions", []):
                mentions.append({**base, **m})
            for j, c in enumerate(d.get("claims", [])):
                claims.append({**base, "claim_id": f"{d['passage_id']}__c{j}", **c,
                               "concepts": json.dumps(c.get("concepts", []))})
            for r in d.get("references", []):
                references.append({**base, **r})

    pd.DataFrame(mentions).to_parquet(out / "concept_mentions.parquet", index=False)
    pd.DataFrame(claims).to_parquet(out / "claims.parquet", index=False)
    pd.DataFrame(references).to_parquet(out / "references.parquet", index=False)
    print(f"{len(mentions)} mentions, {len(claims)} claims, {len(references)} references -> {out}")


if __name__ == "__main__":
    main()
