"""Stage 5: assemble the temporal knowledge graph from the parquet tables.

Nodes: thinkers, works, concepts, claims. Edges: WROTE, CONTAINS (work->claim),
MENTIONS (claim->concept), reference edges (claim->thinker with stance), and
perspectival TRANSFORMS edges from concept_evolution.

    pip install networkx
    python src/build_graph.py --parquet data/parquet --out graph
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import networkx as nx


def safe(v, default=""):
    """GraphML can't serialize None/NaN — coerce any missing value to a safe default.
    Groq's smaller model occasionally emits a null field (e.g. evidence_span), and
    harvested works can have a missing year/title, so this needs to be applied broadly
    rather than only where a crash has already been seen."""
    if v is None:
        return default
    try:
        if isinstance(v, float) and pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet")
    ap.add_argument("--out", default="graph")
    args = ap.parse_args()
    p, out = Path(args.parquet), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    works = pd.read_parquet(p / "works.parquet")
    claims = pd.read_parquet(p / "claims.parquet")
    refs = pd.read_parquet(p / "references.parquet")
    evo = pd.read_parquet(p / "concept_evolution.parquet")

    G = nx.MultiDiGraph()

    for _, w in works.iterrows():
        G.add_node(f"thinker:{w.author}", type="thinker")
        G.add_node(f"work:{w.work_id}", type="work", title=safe(w.title, "untitled"),
                   year=int(w.year) if pd.notna(w.year) else -1,
                   tradition=safe(w.tradition, "unknown"))
        G.add_edge(f"thinker:{w.author}", f"work:{w.work_id}", relation="WROTE")

    for _, c in claims.iterrows():
        cid = f"claim:{c.claim_id}"
        G.add_node(cid, type="claim", text=safe(c.claim), evidence=safe(c.evidence_span),
                   confidence=float(safe(c.confidence, 0) or 0))
        G.add_edge(f"work:{c.work_id}", cid, relation="CONTAINS")
        for concept in json.loads(c.concepts):
            cn = f"concept:{concept}"
            if cn not in G:
                G.add_node(cn, type="concept")
            G.add_edge(cid, cn, relation="MENTIONS")

    for _, r in refs.iterrows():
        if not isinstance(r.target_thinker, str) or not r.target_thinker:
            continue
        tgt = f"thinker:{r.target_thinker.strip().lower().replace(' ', '_')}"
        if tgt not in G:
            G.add_node(tgt, type="thinker", external=True)
        G.add_edge(f"work:{r.work_id}", tgt, relation=safe(r.stance, "CITES"),
                   evidence=safe(r.evidence_span), confidence=float(safe(r.confidence, 0) or 0),
                   perspective="documentary")

    for _, e in evo.iterrows():
        G.add_edge(f"thinker:{e.earlier_author}", f"thinker:{e.later_author}",
                   relation=f"TRANSFORMS_{safe(e.transformation, 'UNKNOWN')}",
                   concept=safe(e.concept), rationale=safe(e.rationale),
                   confidence=float(safe(e.confidence, 0) or 0),
                   perspective=safe(e.perspective, "llm_extraction"), model=safe(e.model))

    nx.write_graphml(G, out / "marxgraph.graphml")
    with (out / "marxgraph.jsonl").open("w") as f:
        for n, d in G.nodes(data=True):
            f.write(json.dumps({"kind": "node", "id": n, **d}) + "\n")
        for u, v, d in G.edges(data=True):
            f.write(json.dumps({"kind": "edge", "source": u, "target": v, **d}) + "\n")

    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {out}/")


if __name__ == "__main__":
    main()
