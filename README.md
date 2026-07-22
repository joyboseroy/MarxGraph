"""Build stratified samples for human validation, per the README's protocol:
100 claims + 50 evolution edges, checked against their evidence spans by a human
annotator, to get an accuracy/kappa figure before any paper claims quality.

    python src/sample_for_validation.py
    # -> data/parquet/validation_claims.csv (100 rows)
    # -> data/parquet/validation_evolution.csv (50 rows)

Open the CSVs in a spreadsheet. For claims, read the evidence_span and judge whether
the claim is actually supported by it (not by your background knowledge of the
thinker); fill in `human_correct` (y/n) and optionally `notes`. For evolution edges,
read the rationale AND the source claims it cites (via key_earlier_claims /
key_later_claims — cross-reference validation_claims.csv or claims.parquet by
claim_id) and judge whether the transformation label is actually supported.
"""

import argparse
from pathlib import Path

import pandas as pd


def stratified_sample(df: pd.DataFrame, n: int, by: str, seed: int = 42) -> pd.DataFrame:
    """Sample ~n rows spread proportionally across the categories in `by`."""
    if len(df) <= n:
        return df.copy()
    groups = df[by].unique()
    per_group = max(1, n // len(groups))
    parts = []
    for g in groups:
        sub = df[df[by] == g]
        parts.append(sub.sample(min(per_group, len(sub)), random_state=seed))
    sampled = pd.concat(parts)
    # top up / trim to hit n as closely as possible
    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        top_up = remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, top_up])
    return sampled.sample(min(n, len(sampled)), random_state=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet")
    ap.add_argument("--n-claims", type=int, default=100)
    ap.add_argument("--n-evolution", type=int, default=50)
    args = ap.parse_args()
    p = Path(args.parquet)

    claims = pd.read_parquet(p / "claims.parquet")
    works = pd.read_parquet(p / "works.parquet")[["work_id", "author", "title"]]
    claims = claims.merge(works, on="work_id", how="left")

    claim_sample = stratified_sample(claims, args.n_claims, by="author")
    claim_out = claim_sample[["claim_id", "author", "title", "claim", "evidence_span",
                               "confidence"]].copy()
    claim_out["human_correct"] = ""   # y/n — is the claim actually supported by evidence_span?
    claim_out["notes"] = ""
    claim_out.to_csv(p / "validation_claims.csv", index=False)
    print(f"{len(claim_out)} claims -> {p / 'validation_claims.csv'}")
    print(f"  stratified by author: {claim_sample.author.value_counts().to_dict()}")

    evo = pd.read_parquet(p / "concept_evolution.parquet")
    evo_sample = stratified_sample(evo, args.n_evolution, by="transformation")
    evo_out = evo_sample[["concept", "earlier_author", "later_author", "transformation",
                           "rationale", "key_earlier_claims", "key_later_claims",
                           "confidence"]].copy()
    evo_out["human_correct"] = ""  # y/n — does the rationale + cited claims support the label?
    evo_out["notes"] = ""
    evo_out.to_csv(p / "validation_evolution.csv", index=False)
    print(f"{len(evo_out)} evolution edges -> {p / 'validation_evolution.csv'}")
    print(f"  stratified by transformation: {evo_sample.transformation.value_counts().to_dict()}")

    print("\nOnce annotated, compute accuracy with:")
    print("  python -c \"import pandas as pd; d=pd.read_csv('data/parquet/validation_claims.csv'); "
          "print((d.human_correct=='y').mean())\"")


if __name__ == "__main__":
    main()
