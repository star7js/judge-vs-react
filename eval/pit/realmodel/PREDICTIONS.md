# Predictions recorded before the data

Each entry is committed before the run it predicts, so the git history shows
the order.

## Aim scene, target width 2 (recorded 2026-11-21, before `aim_sweep.py --target-width 2`)

**Question.** At width 1 the Sonnet 4.6 judge's modal commit is 4.9 ticks
past the window's near edge (`δ = 0.41` of an 11.76-tick window) and its
cliff is at `r* = 0.65`. Is that lateness a fixed number of ticks or a fixed
fraction of the window?

- Fixed ticks: at width 2 (23.42-tick window, in-window ticks 39..61) the
  commit stays about 4.9 ticks in, `δ ≈ 0.21`, and the cliff moves out to
  `r ≈ 0.79`.
- Fixed fraction: `δ` stays 0.41 and the cliff stays near `r ≈ 0.59`
  (`D ≈ 13.8`).

**What the battery says** (`battery_aim_high_w2.json`, 86 frames × 3
samples, run 2026-11-21): 0/3 on ticks 39 and 41, 1/3 on tick 40, 3/3 on
every tick 42..60, 0/3 on tick 61, no FIRE outside the window. Through
`analyze.commit_first_passage` with the two-consecutive rule that is a commit
on tick 43 with probability 1.0, which is 4.7 ticks past the near edge
(`δ = 0.20`).

**Prediction for the 72-episode sweep** (delays 0, 8, 12, 13, 14, 15, 16,
17, 18, 19, 20, 22; six seeds each):

- Commits: all in-window, mode tick 43. Going by width 1, where the closed
  loop ran later than a 3-sample battery predicted, expect a spread over
  roughly ticks 41..44 rather than a single tick.
- Hits: 6/6 through `D = 18` (`r = 0.77`), 0/6 from `D = 19` (`r = 0.81`)
  for a commit on tick 43; so `r*` between 0.75 and 0.82, 54 hits of 72.
  A commit one tick later or earlier moves the last clearing delay by one.
- Every episode obeys hit iff `39 <= commit tick + D <= 61`.

**What would count against it.** A cliff at or below `D = 15` (`r = 0.64`)
would favor the fixed-fraction reading. Out-of-window commits would mean the
prompt does not transfer to the wider target, and the sweep would not test
the question.
