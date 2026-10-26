# Full-text positioning notes

Read on 2026-10-26; none of these external experiments was run locally.

## Mansoor, Phadke and Rana (2026), arXiv:2608.02645v1
- Read main text, algorithm, setup, results, discussion and conclusion: https://arxiv.org/html/2608.02645v1.
- Combines a verifier, verify-before-retry logic and stable keys around one model.
- Observation truth and tool acknowledgements are distinct.
- Verifier design requires domain knowledge; stale or incomplete verification remains a limitation.
- The separate ablation is not numerically comparable with the main configurations.
- Future work already proposes confidence-based choices among waiting, retrying and intervention.
- Our contribution instantiates and measures a restricted controller in that direction; it cannot claim to invent adaptive verification.

## Li (2026), arXiv:2609.29095v1
- Read full main text, propositions, experimental design, results, robustness and limitations: https://arxiv.org/html/2609.29095v1.
- Distinguishes current-state knowledge from eventual-outcome knowledge.
- Late commits defeat finite verification-only completion without a trusted bound.
- Contract changes and stable keys address failure modes that client reasoning cannot remove.
- Waiting, unknown outcomes and misleading completion reports overlap with this article.
- Our narrow addition is explicit observation-time/source pricing and objective sensitivity in a parametric simulator.
- Limbo code and its real-model experiments were not reproduced.

## Helland (2007), CIDR position paper
- Read all ten pages: https://www.cidrdb.org/cidr2007/papers/cidr07p15.pdf.
- Defines entities as disjoint atomicity scopes and activities as durable partner state.
- Repeated messages require remembered identity and idempotent substantive processing.
- Alternate indices can lag; distributed uncertainty remains in application semantics.
- Tentative operations are later confirmed or cancelled through workflows.
- This foundational systems work provides the third full-text comparison, not a competing agent controller.
- Our belief representation is a statistical decision layer over such uncertainty, not a replacement for durable identity or atomicity.

Other verified sources are recorded in verified_sources.md. OpenReview and Research Square links discovered through related-work references were blocked; their unverified claims were not added to the bibliography.
