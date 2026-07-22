"""Concept centrality: builds a bipartite-derived concept graph from the evolution
table (two concepts are linked if they co-occur in the same thinker's claim set)
and separately reports simple cross-lineage coverage, to distinguish concepts
that are genuinely shared vocabulary across the whole tradition from ones that
are localized to a single branch.

    pip install networkx
    python src/concept_centrality.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import networkx as nx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    p = Path(args.parquet)

    evo = pd.read_parquet(p / "concept_evolution.parquet")

    # --- Simple, transparent measure: how many distinct lineage pairs discuss
    # each concept, and how many distinct authors touch it at all ---
    print("=== Cross-lineage coverage (simplest, most interpretable measure) ===\n")
    coverage = (evo.groupby("concept")
                .agg(n_lineage_pairs=("concept", "size"),
                     authors=("earlier_author", lambda s: set(s) | set(evo.loc[s.index, "later_author"])))
                .reset_index())
    coverage["n_distinct_authors"] = coverage["authors"].apply(len)
    coverage = coverage.sort_values("n_lineage_pairs", ascending=False)
    print(coverage[["concept", "n_lineage_pairs", "n_distinct_authors"]]
          .head(args.top).to_string(index=False))

    # --- Graph-based measure: concepts as nodes, edge weight = number of
    # thinkers that co-discuss both concepts (via claims.parquet if available,
    # falls back to evolution-table co-occurrence if not) ---
    claims_path = p / "claims.parquet"
    if claims_path.exists():
        print("\n=== Betweenness centrality (via claims.parquet concept co-occurrence) ===\n")
        claims = pd.read_parquet(claims_path)
        G = nx.Graph()
        for concepts_json in claims.concepts:
            concepts = json.loads(concepts_json)
            for c in concepts:
                G.add_node(c)
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    a, b = concepts[i], concepts[j]
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1)

        print(f"Concept co-occurrence graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        centrality = nx.betweenness_centrality(G, weight="weight")
        top = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
        print(f"\nTop {args.top} concepts by betweenness centrality "
              "(bridges between otherwise-separate concept clusters):\n")
        for concept, score in top:
            print(f"  {concept:<32} {score:.4f}")

        degree = dict(G.degree(weight="weight"))
        top_degree = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
        print(f"\nTop {args.top} concepts by weighted degree "
              "(most frequently co-discussed with other concepts):\n")
        for concept, score in top_degree:
            print(f"  {concept:<32} {score}")

        out = pd.DataFrame([
            {"concept": c, "betweenness": centrality.get(c, 0), "weighted_degree": degree.get(c, 0)}
            for c in G.nodes()
        ]).sort_values("betweenness", ascending=False)
        out_path = p / "concept_centrality.csv"
        out.to_csv(out_path, index=False)
        print(f"\n-> {out_path}")
    else:
        print(f"\n{claims_path} not found -- skipping graph-based centrality, "
              "only the simple coverage measure above was computed.")


if __name__ == "__main__":
    main()
