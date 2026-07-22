"""Stage 3 (Batch mode): extract claims via Groq's asynchronous Batch API.

Use this instead of extract_claims.py when you're hitting Groq's free-tier rate limits
(30 RPM / 1,000 RPD / 8K TPM for openai/gpt-oss-120b as of mid-2026 — check
https://console.groq.com/settings/limits for your current numbers). The Batch API
does NOT count against those limits and costs 50% less, at the cost of latency:
results land within 24 hours to 7 days, not immediately.

Workflow:
    python src/extract_claims_batch.py submit --limit 3000   # build + upload + create batch(es)
    python src/extract_claims_batch.py status                # poll until status=completed
    python src/extract_claims_batch.py collect                # download results -> extractions.jsonl
    # repeat submit for the next chunk once you've collected the previous one

Batches tracked in data/parquet/batches.json so `status`/`collect` know what's outstanding.
Only requires GROQ_API_KEY — this script is Groq-specific (Anthropic has no batch tier
with the same shape as of this writing; extract_claims.py remains the sync path for that
backend).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

GROQ_API = "https://api.groq.com/openai/v1"
PROMPT = Path("prompts/claim_extraction.txt").read_text()
CHUNK_SIZE = 1000  # Groq recommends splitting large workloads rather than one huge batch


def _headers() -> dict:
    import os
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


def _model() -> str:
    import os
    return os.environ.get("MARXGRAPH_MODEL", "openai/gpt-oss-120b")


def seed_concept_block() -> str:
    seeds = yaml.safe_load(Path("config/seed_concepts.yaml").read_text())["concepts"]
    lines = [f"- {slug}: {v['label']} (aliases: {', '.join(v['aliases']) or 'none'})"
             for slug, v in seeds.items()]
    return "SEED CONCEPT LIST:\n" + "\n".join(lines)


def load_batches(out: Path) -> list[dict]:
    p = out / "batches.json"
    return json.loads(p.read_text()) if p.exists() else []


def save_batches(out: Path, batches: list[dict]) -> None:
    (out / "batches.json").write_text(json.dumps(batches, indent=2))


def cmd_submit(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    passages = pd.read_parquet(args.passages)
    works = pd.read_parquet(args.works).drop_duplicates("work_id").set_index("work_id").to_dict("index")

    done = set()
    ext_path = out / "extractions.jsonl"
    if ext_path.exists():
        with ext_path.open() as f:
            done = {json.loads(line)["passage_id"] for line in f if line.strip()}
    in_flight = set()
    for b in load_batches(out):
        if b["status"] not in ("collected",):
            in_flight |= set(b.get("custom_ids", []))

    todo = passages[~passages.passage_id.isin(done | in_flight)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} done, {len(in_flight)} already in-flight, {len(todo)} to submit now")
    if todo.empty:
        return

    seed_block = seed_concept_block()
    model = _model()
    batches = load_batches(out)

    records = todo.to_dict("records")
    for chunk_i in range(0, len(records), CHUNK_SIZE):
        chunk = records[chunk_i:chunk_i + CHUNK_SIZE]
        jsonl_path = out / f"batch_input_{int(time.time())}_{chunk_i}.jsonl"
        custom_ids = []
        with jsonl_path.open("w") as f:
            for row in chunk:
                work = works[row["work_id"]]
                user_msg = (
                    f"{seed_block}\n\nAUTHOR: {work['author']}\nWORK: {work['title']}\n"
                    f"YEAR: {work['year']}\nPASSAGE (id {row['passage_id']}):\n{row['text']}"
                )
                body = {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 3000,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                }
                f.write(json.dumps({"custom_id": row["passage_id"], "method": "POST",
                                     "url": "/v1/chat/completions", "body": body}) + "\n")
                custom_ids.append(row["passage_id"])

        upload = requests.post(f"{GROQ_API}/files", headers=_headers(),
                                files={"file": open(jsonl_path, "rb")},
                                data={"purpose": "batch"}, timeout=120)
        upload.raise_for_status()
        file_id = upload.json()["id"]

        create = requests.post(f"{GROQ_API}/batches", headers=_headers(),
                                json={"input_file_id": file_id, "endpoint": "/v1/chat/completions",
                                      "completion_window": args.window}, timeout=60)
        create.raise_for_status()
        batch = create.json()
        batches.append({
            "batch_id": batch["id"], "input_file_id": file_id,
            "jsonl_path": str(jsonl_path), "status": "submitted",
            "custom_ids": custom_ids, "n": len(custom_ids),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[submit] {batch['id']}: {len(custom_ids)} requests, window={args.window}")

    save_batches(out, batches)


def cmd_status(args) -> None:
    out = Path(args.out)
    batches = load_batches(out)
    outstanding = [b for b in batches if b["status"] not in ("collected",)]
    if not outstanding:
        print("No outstanding batches. Submit some with `submit`.")
        return
    for b in outstanding:
        resp = requests.get(f"{GROQ_API}/batches/{b['batch_id']}", headers=_headers(), timeout=30)
        resp.raise_for_status()
        d = resp.json()
        b["status"] = d["status"]
        b["output_file_id"] = d.get("output_file_id")
        b["error_file_id"] = d.get("error_file_id")
        counts = d.get("request_counts", {})
        print(f"[{d['status']:>12}] {b['batch_id']}  "
              f"{counts.get('completed', 0)}/{counts.get('total', b['n'])} done "
              f"({counts.get('failed', 0)} failed)")
    save_batches(out, batches)


def cmd_collect(args) -> None:
    out = Path(args.out)
    batches = load_batches(out)
    ext_path = out / "extractions.jsonl"
    collected_any = False

    with ext_path.open("a") as ext_f:
        for b in batches:
            if b["status"] != "completed" or not b.get("output_file_id"):
                continue
            resp = requests.get(f"{GROQ_API}/files/{b['output_file_id']}/content",
                                 headers=_headers(), timeout=120)
            resp.raise_for_status()
            n_ok = 0
            for line in resp.text.splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                custom_id = d["custom_id"]
                if d.get("error") or d["response"]["status_code"] != 200:
                    print(f"[warn] {custom_id}: batch request failed, {d.get('error')}")
                    continue
                content = d["response"]["body"]["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(content.strip().removeprefix("```json")
                                         .removeprefix("```").removesuffix("```").strip())
                except json.JSONDecodeError:
                    print(f"[warn] {custom_id}: invalid JSON in batch result, skipping")
                    continue
                # recover work_id from the original jsonl input (custom_id == passage_id,
                # which encodes work_id as its prefix up to "__p")
                work_id = custom_id.rsplit("__p", 1)[0]
                parsed["passage_id"] = custom_id
                parsed["work_id"] = work_id
                parsed["model"] = _model()
                ext_f.write(json.dumps(parsed) + "\n")
                n_ok += 1
            print(f"[collect] {b['batch_id']}: {n_ok} results appended")
            b["status"] = "collected"
            collected_any = True

    save_batches(out, batches)
    if not collected_any:
        print("Nothing to collect yet — run `status` first, wait for status=completed.")
        return

    # rebuild tidy parquet tables from the full extractions log, same as extract_claims.py
    mentions, claims, references = [], [], []
    with ext_path.open() as f:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="data/parquet/passages.parquet")
    ap.add_argument("--works", default="data/parquet/works.parquet")
    ap.add_argument("--out", default="data/parquet")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--window", default="7d", choices=["24h", "7d"],
                    help="longer window = better chance of completion under load")

    sub.add_parser("status")
    sub.add_parser("collect")

    args = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
