# Validation

## Methodology

Two independent samples were drawn for human validation, following a stratified
sampling protocol (`src/sample_for_validation.py`):

- **100 claims**, stratified proportionally across the six authors in the corpus
  (Marx/Engels, Lenin, Luxemburg, Trotsky, Stalin, Mao), out of 71,572 total.
- **50 concept-evolution edges**, stratified proportionally across the seven
  transformation labels (EXTENDED, CONTEXTUALIZED, REFORMULATED, CONTESTED,
  PRESERVED, INSUFFICIENT, REJECTED), out of 446 total.

**Claims** were checked against a strict standard: is the claim's content actually
verifiable from its cited `evidence_span` alone, treating anything outside that
span (including background knowledge of the source text) as unverified. This
is a conservative standard: it flags cases where the evidence span was
truncated by an ellipsis in a way that hides the specific content the claim
asserts, even if that content is accurate.

**Evolution edges** were checked for internal coherence (does the rationale
support the assigned label) and plausibility against known Marxist intellectual
history. This is a weaker check than the claims review: the validation sample
provides only the *IDs* of the claims each rationale cites
(`key_earlier_claims`/`key_later_claims`), not their text, so word-level
cross-referencing against source evidence was not possible in this pass.

An initial automated pass (Claude, Anthropic) was run first, followed by human
adjudication (the corpus author, a domain expert in this literature) on every
case the automated pass could not resolve with confidence. This two-stage
design exists because an LLM checking another LLM's extraction is not an
independent check in the sense the validation protocol intends; both share
similar reasoning patterns and blind spots. The domain-expert pass is what
gives the final numbers below their evidentiary weight; the automated pass
functioned as a fast pre-filter that surfaced candidates for review, not as a
substitute for it.

## Results

| Sample | n | Correct | Accuracy |
|---|---|---|---|
| Claims (vs. evidence span) | 100 | 99 | **99.0%** |
| Evolution edges (rationale coherence + plausibility) | 50 | 50 | **100%** |

5 of 150 total items were flagged as uncertain by the automated pass and
resolved by human review. Of these, 4 were confirmed accurate on inspection:
in each case the claim contained specific content (a name, an attribution, a
causal detail) that had been elided by `...` in the extracted evidence span but
was in fact correct given the full source passage. This illustrates a
systematic conservatism in the evidence-span-only check: it under-counts
correct claims whose exact wording happens to fall outside the quoted span.

One item did not survive review:

> **Claim:** "Political enemies repeatedly reused old accusations, shifting
> them from right to left **without imagination**."
> **Evidence span:** "they simply switched the same old accusations about
> from one point to another, the movement being predominantly from right to
> left"

The phrase "without imagination" is an evaluative addition not present in the
source text, a documented instance of the extraction model supplying a
plausible-sounding editorial gloss beyond what the evidence supports. This is
retained in the dataset (flagged `human_correct=n` in the validation CSV)
rather than silently corrected, since it is a genuine and informative failure
mode of the pipeline, not an isolated fluke to be memory-holed.

## Caveats

- **Sample size.** 100/71,572 claims (0.14%) and 50/446 evolution edges
  (11.2%) were reviewed. These are point estimates, not corpus-wide guarantees;
  a different random sample could yield a different figure, especially for the
  claims accuracy given the small absolute sample relative to corpus size.
- **Confidence scores are not a reliable uncertainty signal.** The 446
  evolution edges have a confidence distribution with mean 0.852 and std 0.035
  (75% of all edges fall between 0.85-0.86) - implausibly narrow for a model
  genuinely discriminating between certain and uncertain judgments. Do not
  filter or weight by this column as if it reflects calibrated uncertainty.
- **Evolution-edge validation was weaker than claims validation**, for the
  reason described above (no claim text in the sample, only IDs). A rigorous
  cross-reference against full claim text has not yet been performed at scale.
- **Interpretive edges are perspectival, not objective fact.** Transformation
  labels (especially CONTESTED/REJECTED) reflect the extraction model's reading
  of the cited passages, framed as a documentary+interpretive distinction (see
  `config/ontology.yaml`) rather than a canonical verdict on contested
  historical questions (e.g., continuity vs. rupture between Lenin and
  Stalin). Different scholarly traditions describe these relationships
  differently; the dataset records one LLM-assisted reading, evidence-linked
  and human-spot-checked, not a settled adjudication.
- **A likely duplicate work was identified but not yet removed as of this
  validation pass**: Trotsky's *History of the Russian Revolution* appears to
  have been ingested twice under different work_ids (a single-file edition and
  a separately split 3-volume edition), via independent keyword matches on the
  same harvested index page. `src/dedupe_works.py` exists to detect and remove
  this class of duplicate but has not yet been run against the released
  corpus. Downstream users should be aware that, until this is applied,
  Trotsky's claims on any concept discussed in that book may be
  double-weighted relative to other authors.
