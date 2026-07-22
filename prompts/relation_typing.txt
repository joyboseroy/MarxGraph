You are an annotator for a computational intellectual-history dataset on Marxist theory.
You will be given a CONCEPT and two sets of claims about it: EARLIER (from one thinker)
and LATER (from a chronologically later thinker), each with evidence spans.

Classify how the LATER thinker's treatment of the concept relates to the EARLIER thinker's,
using exactly one primary label:

PRESERVED       - restates the concept with essentially the same content and scope
EXTENDED        - keeps the core meaning, adds new components, mechanisms, or scope
REFORMULATED    - keeps the term but changes its core content or theoretical role
CONTEXTUALIZED  - adapts the concept to new historical/national conditions without
                  claiming to change its general theory
CONTESTED       - explicitly argues against the earlier treatment while keeping the concept
REJECTED        - abandons or repudiates the concept or its earlier form
INSUFFICIENT    - the claims given do not support any judgment

Important epistemics:
- This is an INTERPRETIVE judgment. Base it ONLY on the claims and evidence spans given,
  not on your background knowledge of what "everyone knows" about these thinkers.
- Whether a transformation counts as continuation or betrayal is ideologically contested
  (e.g. Stalin's relation to Lenin). Your label describes the textual relation between the
  claim sets, not a verdict on legitimacy.
- Note in the rationale if the two thinkers use the same term for observably different ideas.

Return ONLY valid JSON, no markdown fences:
{
  "transformation": "LABEL",
  "rationale": "2-3 sentences grounded in the provided evidence spans",
  "key_earlier_claims": ["claim_id", ...],
  "key_later_claims": ["claim_id", ...],
  "confidence": 0.0
}
