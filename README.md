# MarxGraph

**Released dataset:** [https://huggingface.co/datasets/joyboseroy/MarxGraph](https://huggingface.co/datasets/joyboseroy/MarxGraph) 

A temporal, perspectival knowledge graph of Marxist thought: how concepts moved and mutated
from Marx and Engels through Lenin, Luxemburg, Trotsky and Stalin to Mao. Corpus sourced
from the Marxists Internet Archive (MIA), released as linked parquet tables (Hugging Face)
plus GraphML/JSONL exports and full extraction code.

## Why EPUBs, not a scraper

MIA publishes clean volunteer-produced EPUBs for nearly every core V1 work. Downloading
~40 EPUB files replaces crawling thousands of HTML pages: cleaner text, trivially cached,
and far kinder to a volunteer-run archive. `src/download.py` uses a 5-second delay, a
descriptive User-Agent, checks robots.txt, and caches everything.

## Pipeline

```
python src/download.py --config config/works.yaml        # 1. fetch EPUBs + manifest.json
python src/parse_corpus.py                               # 2. works.parquet, passages.parquet

# choose an LLM backend for stages 3-4 (src/llm_backend.py):
export MARXGRAPH_BACKEND=groq                             # cheap Llama/GPT-OSS via Groq
export GROQ_API_KEY=...
#   -- or --
export MARXGRAPH_BACKEND=anthropic                        # default
export ANTHROPIC_API_KEY=...

python src/extract_claims.py --limit 50                  # 3. claims, mentions, references
                                                         #    (resumable; start small, inspect,
                                                         #     then remove --limit)
python src/type_relations.py                             # 4. concept_evolution.parquet
pip install networkx && python src/build_graph.py        # 5. marxgraph.graphml / .jsonl
```

Both stages 3 and 4 go through `src/llm_backend.py`, a thin `requests`-based wrapper (no
Anthropic or Groq SDK needed) so you can switch backends without touching pipeline code.
Groq's default model (`openai/gpt-oss-120b`) can move fast — Groq deprecated
`llama-3.3-70b-versatile` in 2026 — so check https://console.groq.com/docs/models before a
long run and override with `MARXGRAPH_MODEL` if needed. Since Groq's models are much smaller
than Sonnet, expect noisier claim extraction and lower confidence scores; the validation
protocol below matters even more with this backend, and it's worth spot-checking the first
`--limit 20` batch closely before scaling up.

### Hitting Groq's free-tier rate limits?

`openai/gpt-oss-120b` on Groq's free tier is capped around 30 RPM / 1,000 RPD / 8K TPM
(check current numbers at https://console.groq.com/settings/limits — these move). At that
ceiling, ~14,000 passages take hours no matter how many `--workers` you throw at
`extract_claims.py` — concurrency just makes you hit 429s faster, since it's a hard
per-account cap, not a speed problem. Neither more CPU/GPU nor Colab changes this: every
request still goes out to the same account-level limit.

Use `src/extract_claims_batch.py` instead — Groq's async Batch API, which does **not** count
against standard rate limits and costs 50% less:

```
python src/extract_claims_batch.py submit --limit 3000 --window 7d   # build, upload, create batch(es)
python src/extract_claims_batch.py status                            # poll until status=completed
python src/extract_claims_batch.py collect                           # download -> claims.parquet etc.
```

Tradeoff: results land within 24 hours to 7 days, not immediately, so this suits "let it run
overnight" rather than interactive iteration. Batches are tracked in `data/parquet/batches.json`;
`submit` automatically skips passages already done or already in-flight in another batch, so
you can call it repeatedly for successive chunks. This only works with the Groq backend —
Anthropic has no equivalent batch tier of the same shape at the time of writing.

## Dataset tables

| file | grain | key columns |
|---|---|---|
| works.parquet | one work | work_id, author, title, year, tradition, license, source_url, sha256 |
| passages.parquet | ~220-word passage | passage_id, work_id, chapter, text |
| concept_mentions.parquet | one mention | passage_id, concept, surface_form, is_new_concept |
| claims.parquet | one proposition | claim_id, claim, concepts, evidence_span, confidence |
| references.parquet | explicit citation/attack | work_id, target_thinker, stance, evidence_span |
| concept_evolution.parquet | concept x (earlier, later) author pair | transformation, rationale, perspective, confidence |

## Design commitments

1. **Documentary vs interpretive.** "Lenin wrote claim Y in 1917" is documentary.
   "Mao EXTENDS Lenin on revolutionary agency" is interpretive, and every interpretive
   edge carries `perspective`, `extraction_method`, `model`, and `confidence`. Contradictory
   edges from different perspectives are a feature: the genealogy of Marxism is contested
   (continuation vs betrayal debates), and the graph represents that contest instead of
   silently adjudicating it.
2. **Evidence spans everywhere.** Every claim and edge cites a verbatim quote and passage id.
   No span, no edge.
3. **Transformation typology.** PRESERVED / EXTENDED / REFORMULATED / CONTEXTUALIZED /
   CONTESTED / REJECTED, defined in `config/ontology.yaml` and `prompts/relation_typing.txt`.

## Validation protocol (needed before any paper claim)

Sample 100 extracted claims and 50 evolution edges, stratified by author. Two human
annotators label correctness against the evidence span; report accuracy and Cohen's kappa.
Without this the dataset is "an LLM's opinions about Lenin", which reviewers will notice.

## Licensing (read before uploading text to Hugging Face)

MIA content is free to access but **not uniformly free of copyright**. Pre-1929 originals
and MIA volunteer translations are generally public domain; some translations (notably
certain Progress Publishers / Foreign Languages Press editions used for Stalin, Mao, and
late Lenin volumes) have contested or reserved status. The manifest marks these `verify`.
Options per work: (a) confirm PD/permission from the work's MIA credits page and ship text;
(b) ship passage offsets + annotations only, with a download script that re-fetches text
from MIA at load time (the works/claims/relations tables are your annotations and are
yours to license, e.g. CC-BY-4.0). Credit MIA prominently either way.

## V1 scope

Marx/Engels, Lenin, Luxemburg, Trotsky, Stalin, Mao.
