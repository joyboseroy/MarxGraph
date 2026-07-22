"""Orthodoxy index: for each later thinker, what fraction of the concepts they
inherited were kept largely intact (PRESERVED/EXTENDED) versus genuinely
departed from (CONTESTED/REFORMULATED/REJECTED/CONTEXTUALIZED). INSUFFICIENT
edges are excluded since they represent a non-judgment, not a data point.

    python src/orthodoxy_index.py
"""

import argparse
from pathlib import Path

import pandas as pd

CONTINUITY = {"PRESERVED", "EXTENDED"}
DEPARTURE = {"CONTESTED", "REFORMULATED", "REJECTED", "CONTEXTUALIZED"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet")
    args = ap.parse_args()
    p = Path(args.parquet)

    evo = pd.read_parquet(p / "concept_evolution.parquet")
    scored = evo[evo.transformation.isin(CONTINUITY | DEPARTURE)].copy()
    scored["continuity"] = scored.transformation.isin(CONTINUITY)

    rows = []
    for author, grp in scored.groupby("later_author"):
        n = len(grp)
        n_continuity = grp.continuity.sum()
        n_departure = n - n_continuity
        rows.append({
            "thinker": author,
            "n_edges": n,
            "continuity_pct": round(100 * n_continuity / n, 1),
            "departure_pct": round(100 * n_departure / n, 1),
            "n_preserved": (grp.transformation == "PRESERVED").sum(),
            "n_extended": (grp.transformation == "EXTENDED").sum(),
            "n_contextualized": (grp.transformation == "CONTEXTUALIZED").sum(),
            "n_reformulated": (grp.transformation == "REFORMULATED").sum(),
            "n_contested": (grp.transformation == "CONTESTED").sum(),
            "n_rejected": (grp.transformation == "REJECTED").sum(),
        })

    result = pd.DataFrame(rows).sort_values("continuity_pct", ascending=False)
    print("Orthodoxy index (higher continuity_pct = kept more of what was inherited intact)\n")
    print(result.to_string(index=False))

    out_path = p / "orthodoxy_index.csv"
    result.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")

    print(f"\nExcluded {(evo.transformation == 'INSUFFICIENT').sum()} INSUFFICIENT edge(s) "
          f"from scoring (non-judgment, not a data point).")
    print("\nNote: this only scores each thinker as a RECEIVER (later_author). A thinker who")
    print("appears only as an earlier_author across all their edges (rare, but check n_edges")
    print("above) will not get a row here.")


if __name__ == "__main__":
    main()
