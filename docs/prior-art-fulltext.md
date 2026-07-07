# Prior-art full-text check (Task 0) — GO, with framing refinement

Date: 2026-07-06. Gating novelty check for the Judge vs. React paper. Two
competitors read in full (twice each, neutral prompts to avoid the summarizer
contamination the first survey hit); two more quick-checked.

The paper's contribution rests on three sub-results:
- **S1** — success-rate phase transition vs. the dimensionless ratio
  `r = latency ÷ action-window`, transition near `r ≈ 1`.
- **S2** — controlled **same-decision flip**: the *identical correct action*
  succeeds when blocking, fails in real-time (isolates latency from judgment).
- **S3** — **action-repeat / decision-hold** experiment: does holding a decision
  across frames rescue real-time performance? (We show it does not.)

## Findings

| Paper | S1 | S2 | S3 | Isolates on single decision? |
|---|---|---|---|---|
| **OmniGameArena** (2606.09826) | ABSENT — reports absolute action-count/score drops, "avoids normalized threshold language" | ABSENT — PDQ/LCRT scored per **episode**, "no analysis isolates individual decision points" | ABSENT — "no intervention holds one action constant across frames" | No — aggregate whole-game |
| **Win Fast or Lose Slow** (2505.19481) | ABSENT but **closest** — has an action-window/**200ms saturation** point, but latency swept via compression γ that **also degrades quality** (confounded) | ABSENT — ELO/win-rate over full matches | ABSENT | No — confounded, match-level |
| **VideoGameBench** (2505.18134) | ABSENT | ABSENT — Lite vs real-time only at aggregate completion (0.48% vs 1.6%) | ABSENT — has a timed-action interface, never tests it | Partial, aggregate |
| **Real-Time Deadlines** (2601.13206) | ABSENT — fixed budgets, ratio not swept | ABSENT — aggregate closure rates | ABSENT | Partial (strong but dialogue, aggregate) |

## Verdict: GO

None of S1/S2/S3 is fully present in any of the four. Ranked by safety:
1. **S3 — safest**, wholly uncontested; nobody runs a decision-hold intervention.
2. **S2 — safe**; pause-vs-real-time exists everywhere but only as **aggregate
   whole-game** comparisons. The **single-decision** flip is genuinely novel.
3. **S1 — open, but frame carefully** (see guardrail).

## Guardrails (must-do to stay novel)

- **S1 vs. Win Fast:** they already have the action-window/200ms concept as a
  *saturation* threshold, with latency confounded with quality. Our S1 must be a
  **success-rate collapse at `r ≈ 1` with latency (`D`) varied INDEPENDENTLY of
  judgment quality** — which our injected-delay apparatus does by construction
  (the model is fixed; `D` is a harness knob). Explicitly contrast against their
  saturation-point + quality-confounded curve. Do **not** pitch S1 as merely
  "there is a characteristic timescale equal to the action window."
- **Lead with S2's specificity**, not a generic "we isolate timing from
  judgment" claim.

## Claims to DROP as already-taken

- "First to study the latency–quality tradeoff for LLM/VLM agents" — owned by
  **Win Fast** ("the first systematic study of this trade off").
- "First to isolate temporal/timing failure from decision competence" in the
  general sense — owned at aggregate level by **Real-Time Deadlines** and
  **OmniGameArena** (PDQ/LCRT).

## Recommended lead for the paper

**S2 (single-decision same-action flip) as the conceptual hook**, backed by
**S1 (independently-varied, quality-controlled ratio collapse at `r ≈ 1`) as the
quantitative centerpiece**, with **S3 (decision-hold rescue test) as the clean,
uncontested mechanism experiment.**
