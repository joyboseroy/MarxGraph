"""Stage 4: classify how each concept transformed between chronologically ordered thinkers.

For every concept, build claim sets per author, order author pairs by lineage, and ask the
model to type the transformation. Output: concept_evolution.parquet.

Resumable: (concept, earlier_author, later_author) triples already present in
concept_evolution.parquet are skipped, so a rerun after partial failures only fills gaps
instead of recomputing everything and re-spending quota on jobs that already succeeded.

    python src/type_relations.py --parquet data/parquet --workers 24
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import pandas as pd

import llm_backend

PROMPT = Path("prompts/relation_typing.txt").read_text()

# V1 lineage constraint: only type pairs that are plausibly a transmission path.
LINEAGE_PAIRS = {
    ("marx", "lenin"), ("engels", "lenin"), ("marx", "luxemburg"), ("engels", "luxemburg"),
    ("marx", "trotsky"), ("marx", "stalin"), ("marx", "mao"),
    ("lenin", "trotsky"), ("lenin", "stalin"), ("lenin", "mao"),
    ("luxemburg", "lenin"),          # contemporaneous debate, both directions matter
    ("lenin", "luxemburg"),
    ("stalin", "mao"), ("trotsky", "stalin"), ("stalin", "trotsky"),
    ("mixed_marx_engels", "lenin"), ("mixed_marx_engels", "luxemburg"),
    ("mixed_marx_engels", "trotsky"), ("mixed_marx_engels", "stalin"),
    ("mixed_marx_engels", "mao"),
}

MAX_CLAIMS_PER_SIDE = 12
MIN_CLAIMS_PER_SIDE = 2


def claim_block(df: pd.DataFrame) -> str:
    lines = []
    for _, r in df.head(MAX_CLAIMS_PER_SIDE).iterrows():
        lines.append(f"[{r.claim_id}] ({r.title}, {r.year}) {r.claim}\n"
                     f"    evidence: \"{r.evidence_span}\"")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    p = Path(args.parquet)

    claims = pd.read_parquet(p / "claims.parquet")
    works = pd.read_parquet(p / "works.parquet")
    claims = claims.merge(works[["work_id", "author", "title", "year"]], on="work_id")
    claims["concept_list"] = claims["concepts"].apply(json.loads)
    exploded = claims.explode("concept_list").rename(columns={"concept_list": "concept"})
    exploded = exploded.dropna(subset=["concept"])

    print(f"backend={llm_backend.BACKEND} model={llm_backend.MODEL} workers={args.workers}")

    evo_path = p / "concept_evolution.parquet"
    existing = pd.read_parquet(evo_path) if evo_path.exists() else pd.DataFrame(
        columns=["concept", "earlier_author", "later_author"])
    done = set(zip(existing.concept, existing.earlier_author, existing.later_author))
    if done:
        print(f"{len(done)} already typed from a previous run — will skip those")

    jobs = []  # (concept, earlier, later, user_msg)
    for concept, grp in exploded.groupby("concept"):
        authors = grp.author.unique()
        for a, b in combinations(authors, 2):
            # orient by lineage table; skip unknown pairs
            if (a, b) in LINEAGE_PAIRS:
                earlier, later = a, b
            elif (b, a) in LINEAGE_PAIRS:
                earlier, later = b, a
            else:
                continue
            if (concept, earlier, later) in done:
                continue
            e_claims = grp[grp.author == earlier]
            l_claims = grp[grp.author == later]
            if len(e_claims) < MIN_CLAIMS_PER_SIDE or len(l_claims) < MIN_CLAIMS_PER_SIDE:
                continue
            user_msg = (f"CONCEPT: {concept}\n\nEARLIER thinker: {earlier}\n"
                        f"{claim_block(e_claims)}\n\nLATER thinker: {later}\n"
                        f"{claim_block(l_claims)}")
            jobs.append((concept, earlier, later, user_msg))
    print(f"{len(jobs)} concept x lineage-pair jobs to type this run")

    def worker(job):
        concept, earlier, later, user_msg = job
        try:
            d = llm_backend.call_json(PROMPT, user_msg, max_tokens=1200)
            return concept, earlier, later, d, None
        except Exception as exc:
            return concept, earlier, later, None, str(exc)

    rows = []
    lock = threading.Lock()
    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                concept, earlier, later, d, err = fut.result()
                if err:
                    print(f"[FAIL] {concept}: {earlier}->{later}: {err}")
                    continue
                with lock:
                    rows.append({
                        "concept": concept, "earlier_author": earlier, "later_author": later,
                        "transformation": d.get("transformation"),
                        "rationale": d.get("rationale"),
                        "key_earlier_claims": json.dumps(d.get("key_earlier_claims", [])),
                        "key_later_claims": json.dumps(d.get("key_later_claims", [])),
                        "confidence": d.get("confidence"),
                        "perspective": "llm_extraction", "model": llm_backend.MODEL,
                    })
                print(f"[ok  ] {concept}: {earlier} -> {later}: {d.get('transformation')}")
                if i % 25 == 0:
                    print(f"  ...{i}/{len(jobs)}")

    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    combined.to_parquet(evo_path, index=False)
    print(f"{len(rows)} new edges this run, {len(combined)} total -> {evo_path}")
    if not jobs:
        print("Nothing left to type — all lineage pairs already covered.")


if __name__ == "__main__":
    main()
