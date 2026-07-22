"""Quick first-look summary of concept_evolution.parquet — sanity-check the shape of
the results before building the graph or writing anything up.

    python src/summarize_evolution.py
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet")
    args = ap.parse_args()
    p = Path(args.parquet)

    evo = pd.read_parquet(p / "concept_evolution.parquet")
    claims = pd.read_parquet(p / "claims.parquet")
    works = pd.read_parquet(p / "works.parquet")

    print(f"=== Corpus ===")
    print(f"{len(works)} works, {len(claims)} claims, {len(evo)} evolution edges\n")

    print("=== Transformation type distribution ===")
    dist = evo.transformation.value_counts()
    for label, n in dist.items():
        print(f"  {label:<16} {n:>4}  ({n/len(evo)*100:.1f}%)")

    print("\n=== By lineage pair (earlier -> later) ===")
    pair_counts = evo.groupby(["earlier_author", "later_author"]).size().sort_values(ascending=False)
    for (e, l), n in pair_counts.items():
        print(f"  {e:<20} -> {l:<10} {n:>4} edges")

    print("\n=== Transformation mix per lineage pair ===")
    cross = pd.crosstab([evo.earlier_author, evo.later_author], evo.transformation)
    print(cross.to_string())

    print("\n=== Concepts with the most cross-thinker edges (contested ground) ===")
    top_concepts = evo.groupby("concept").size().sort_values(ascending=False).head(15)
    for concept, n in top_concepts.items():
        labels = evo[evo.concept == concept].transformation.value_counts().to_dict()
        print(f"  {concept:<32} {n:>2} edges  {labels}")

    print("\n=== Non-EXTENDED edges (the interesting ones) ===")
    interesting = evo[evo.transformation != "EXTENDED"].sort_values(["concept", "earlier_author"])
    for _, r in interesting.iterrows():
        print(f"  [{r.transformation:<14}] {r.concept:<28} {r.earlier_author} -> {r.later_author}"
              f"  (conf={r.confidence:.2f})")

    print(f"\n=== Confidence distribution ===")
    print(evo.confidence.describe().to_string())

    low_conf = evo[evo.confidence < 0.6]
    if len(low_conf):
        print(f"\n{len(low_conf)} edges with confidence < 0.6 (worth reviewing first):")
        print(low_conf[["concept", "earlier_author", "later_author", "transformation", "confidence"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
