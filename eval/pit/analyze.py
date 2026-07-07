"""Analysis + figures for the pit-jump experiment.

Consumes the oracle sweep (`eval/pit/data/oracle_sweep.jsonl`, from
`eval.pit.run_oracle_sweep`) and, when present, VLM runs
(`eval/pit/data/vlm_runs.jsonl`, from `eval.pit.run_vlm`).

Core fits:
- `ratio(row, tick_s)` -- the dimensionless timing ratio
  `r = speed * delay_ticks / W` (`W = w_hi - w_lo`), the analytic takeoff
  window's width. `W <= 0` is a degenerate/empty window (e.g. the
  `pit_width=4` oracle configs, which the C sim marks with `w_lo=10.0,
  w_hi=-1.0`: no takeoff position clears, so the pit is unclearable) --
  `ratio()` returns `float("inf")` for those rows rather than dividing by a
  non-positive width.
- `psychometric_fit(x, y)` / `transition_fit(r, success)` -- logistic curve
  fits via `scipy.optimize.curve_fit`, recovered on synthetic data in
  `eval/pit/tests/test_fits.py` (never fit against the deterministic oracle
  itself, which has no real fit "error" to recover).

Figure writers (`fig_timing`, `fig_dissociation`, `fig_flip`, `fig_repeat`)
each write one PNG under an output directory. The oracle policy is
deterministic and ignores fog/seed (it decides from the true window, not a
rendered frame), so:
- `fig_timing` and `fig_flip` are real analyses of oracle data.
- `fig_repeat` is real oracle data too, but needs more than one `repeat`
  value to plot r* vs k; the default sweep grid only has `repeat=1`, so
  `main()` generates a small supplementary repeat sweep via the same C sim
  binary the oracle sweep uses.
- `fig_dissociation` (judgment accuracy vs fog) *cannot* be shown with
  oracle data -- the oracle ignores fog, so it would look perfect
  regardless of fog and that would misrepresent a VLM's real
  perception/decision dissociation. If no `vlm_runs.jsonl` data is
  supplied, it writes a clearly-labeled placeholder figure instead of
  fabricating an effect.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

DEFAULT_ORACLE = "eval/pit/data/oracle_sweep.jsonl"
DEFAULT_VLM = "eval/pit/data/vlm_runs.jsonl"
DEFAULT_OUT = "eval/pit/figures"
DEFAULT_REPEAT_SWEEP = "eval/pit/data/oracle_repeat_sweep.jsonl"
DEFAULT_TICK_S = 1 / 60


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Return the parsed rows of a JSONL file, or `[]` if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# ratio()
# ---------------------------------------------------------------------------

def window_width(row: dict) -> float:
    """`W = w_hi - w_lo`. Non-positive means the pit is unclearable (no
    takeoff position clears it)."""
    return row["w_hi"] - row["w_lo"]


def ratio(row: dict, tick_s: float = DEFAULT_TICK_S) -> float:
    """Dimensionless timing ratio `r = speed * delay_ticks / W`.

    `delay_ticks` comes from the oracle's `delay` field (already in ticks)
    if present and usable; otherwise falls back to a VLM row's `L_mean_s`
    (measured decision latency, in seconds), converted to ticks via
    `tick_s`. A VLM row prefers real measured latency (`L_mean_s`) over its
    config `delay` (which is normally 0 for VLM runs) when both are
    present, since `L_mean_s` is the quantity actually being studied there.

    Returns `float("inf")` when `W <= 0` (empty/degenerate window --
    unclearable regardless of timing) rather than dividing by a
    non-positive width.
    """
    w = window_width(row)
    if w <= 0:
        return float("inf")

    l_mean_s = row.get("L_mean_s")
    if l_mean_s is not None and not (isinstance(l_mean_s, float) and math.isnan(l_mean_s)):
        delay_ticks = l_mean_s / tick_s
    elif row.get("delay") is not None:
        delay_ticks = row["delay"]
    else:
        raise KeyError(
            "ratio(): row has neither a usable 'L_mean_s' nor a 'delay' field"
        )

    return row["speed"] * delay_ticks / w


# ---------------------------------------------------------------------------
# Logistic fits
# ---------------------------------------------------------------------------

_EXP_CLIP = 500.0  # avoids harmless np.exp overflow warnings far from threshold


def _increasing_logistic(x, threshold, slope):
    z = np.clip(-slope * (x - threshold), -_EXP_CLIP, _EXP_CLIP)
    return 1.0 / (1.0 + np.exp(z))


def _decreasing_logistic(r, r_star, slope):
    z = np.clip(slope * (r - r_star), -_EXP_CLIP, _EXP_CLIP)
    return 1.0 / (1.0 + np.exp(z))


def _center_bounds(x: np.ndarray) -> tuple[float, float]:
    """Loose bounds for a fitted x-intercept: the data's range padded by
    half its span (or by 1.0 if the data has no spread)."""
    spread = float(x.max() - x.min())
    pad = spread if spread > 0 else 1.0
    return float(x.min()) - pad, float(x.max()) + pad


def psychometric_fit(x, y) -> dict:
    """Logistic fit of a binary outcome `y` against a stimulus `x`, using
    the standard increasing-sigmoid convention
    `p(x) = 1 / (1 + exp(-slope * (x - threshold)))`.
    Returns `{"threshold": float, "slope": float}`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = _center_bounds(x)
    p0 = [float(np.median(x)), 1.0]
    popt, _ = curve_fit(
        _increasing_logistic, x, y, p0=p0,
        bounds=([lo, -1000.0], [hi, 1000.0]), maxfev=20000,
    )
    return {"threshold": float(popt[0]), "slope": float(popt[1])}


def transition_fit(r, success) -> dict:
    """Logistic fit of a binary outcome `success` against the timing ratio
    `r`, using the decreasing-sigmoid convention
    `p(r) = 1 / (1 + exp(slope * (r - r_star)))` (matches the physical
    picture: success is high for `r < 1` -- decision executes before the
    window closes -- and drops past `r ~ 1`).
    Returns `{"r_star": float, "slope": float}`.

    Non-finite `r` values (from `ratio()`'s `float("inf")` for empty
    windows) are dropped before fitting -- they carry no timing
    information (the config is unclearable regardless of `r`).
    """
    r = np.asarray(r, dtype=float)
    success = np.asarray(success, dtype=float)
    finite = np.isfinite(r)
    r, success = r[finite], success[finite]
    if len(r) < 2:
        raise ValueError("transition_fit(): need at least 2 finite-r rows to fit")
    lo, hi = _center_bounds(r)
    p0 = [float(np.median(r)), 5.0]
    popt, _ = curve_fit(
        _decreasing_logistic, r, success, p0=p0,
        bounds=([lo, -1000.0], [hi, 1000.0]), maxfev=20000,
    )
    return {"r_star": float(popt[0]), "slope": float(popt[1])}


# ---------------------------------------------------------------------------
# Figure writers
# ---------------------------------------------------------------------------

_PIT_WIDTH_COLORS = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}


def fig_timing(oracle_rows: list[dict], out_path: str, vlm_rows: list[dict] | None = None,
                tick_s: float = DEFAULT_TICK_S) -> dict:
    """Headline figure: P(clear) vs the dimensionless ratio r, pooled across
    pit_width (each has a different window width W, so the same delay
    sweep lands at a different r) -- the transitions should collapse near
    r ~ 1. Real-model VLM points (if any) are overlaid on the same r axis.

    Excludes rows with an empty/degenerate window (W <= 0, e.g.
    pit_width=4) from the fit -- they are unclearable at any delay and
    would only distort the transition. Returns the fit dict plus the
    excluded-row count, and prints both.
    """
    excluded = [row for row in oracle_rows if window_width(row) <= 0]
    usable = [row for row in oracle_rows if window_width(row) > 0]

    r = np.array([ratio(row, tick_s) for row in usable])
    y = np.array([row["cleared"] for row in usable], dtype=float)
    fit = transition_fit(r, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    for pw in sorted({row["pit_width"] for row in usable}):
        mask = np.array([row["pit_width"] == pw for row in usable])
        ax.scatter(
            r[mask], y[mask] + np.random.default_rng(pw).uniform(-0.02, 0.02, mask.sum()),
            s=18, alpha=0.5, color=_PIT_WIDTH_COLORS.get(pw, "gray"),
            label=f"pit_width={pw}",
        )

    r_curve = np.linspace(0, max(r.max(), fit["r_star"] * 1.5, 2.0), 400)
    ax.plot(r_curve, _decreasing_logistic(r_curve, fit["r_star"], fit["slope"]),
            color="black", lw=2, label="pooled logistic fit")
    ax.axvline(fit["r_star"], color="black", ls="--", lw=1,
               label=f"r* = {fit['r_star']:.2f}")

    if vlm_rows:
        # A model's own latency L places it at r(L) on this axis; whether it
        # clears in REAL TIME is read off the ORACLE curve at that r (left of r*
        # clears, right of r* fails). We deliberately do NOT scatter the VLM's
        # `cleared` here: run_vlm episodes execute at delay=0 (to measure L and
        # the delay-0 judgment), so their `cleared` does not reflect their own
        # latency and would misleadingly sit at y=1 past r*. Plot each model's
        # r(L) position as a vertical marker instead.
        from collections import defaultdict
        by_model = defaultdict(list)
        for row in vlm_rows:
            rr = ratio(row, tick_s)
            if np.isfinite(rr):
                by_model[row.get("model", "model")].append(rr)
        for i, (name, rs) in enumerate(sorted(by_model.items())):
            rbar = float(np.mean(rs))
            ax.axvline(rbar, color=f"C{i}", ls=":", lw=1.5,
                       label=f"{name} (r={rbar:.2f})")

    ax.set_xlabel("r = speed * delay_ticks / W")
    ax.set_ylabel("cleared")
    ax.set_title(
        f"Timing collapse across pit widths (excluded {len(excluded)} "
        f"empty-window rows, W<=0)"
    )
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(
        f"[fig_timing] r_star={fit['r_star']:.4f} slope={fit['slope']:.4f}; "
        f"excluded {len(excluded)}/{len(oracle_rows)} rows with W<=0 "
        f"(empty/degenerate window)"
    )
    return {**fit, "excluded_empty_window_rows": len(excluded)}


def fig_aim(aim_rows: list[dict], out_path: str, tick_s: float = DEFAULT_TICK_S) -> dict:
    """Generality test: the same r = travel/window collapse, on a *different*
    task. The sweeping-aim episodes emit the pit's schema with speed<-omega and
    w_lo/w_hi<-angular window, so `ratio()` gives r = omega*delay/(2h) with no
    special-casing. If clear/fail collapses near r~1 across target widths just
    like the pit, the timing law is not specific to translational jumps."""
    r = np.array([ratio(row, tick_s) for row in aim_rows])
    y = np.array([row["cleared"] for row in aim_rows], dtype=float)
    finite = np.isfinite(r)
    r, y = r[finite], y[finite]
    fit = transition_fit(r, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    widths = sorted({row["pit_width"] for row in aim_rows})
    for wid in widths:
        mask = np.array([row["pit_width"] == wid for row in aim_rows])[finite]
        # width_id -> target angular window width in degrees, for the legend
        w0 = next(row for row in aim_rows if row["pit_width"] == wid)
        deg = (w0["w_hi"] - w0["w_lo"]) * 180.0 / np.pi
        ax.scatter(
            r[mask], y[mask] + np.random.default_rng(wid).uniform(-0.02, 0.02, mask.sum()),
            s=20, alpha=0.6, color=_PIT_WIDTH_COLORS.get(wid, "gray"),
            label=f"target width {deg:.1f}°",
        )
    r_curve = np.linspace(0, max(r.max(), fit["r_star"] * 1.5, 2.0), 400)
    ax.plot(r_curve, _decreasing_logistic(r_curve, fit["r_star"], fit["slope"]),
            color="black", lw=2, label="pooled logistic fit")
    ax.axvline(fit["r_star"], color="black", ls="--", lw=1,
               label=f"r* = {fit['r_star']:.2f}")
    ax.set_xlabel("r = omega * delay_ticks / (2h)  [angular travel / window]")
    ax.set_ylabel("cleared (shot hit the target)")
    ax.set_title("Generality: the r~1 law on a second task\n"
                 "(sweeping-aim; rotate through an angular window, fire once)")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_aim] r_star={fit['r_star']:.4f} slope={fit['slope']:.4f} "
          f"(n={len(r)} configs, {len(widths)} target widths)")
    return {**fit, "n": len(r)}


def fig_dissociation(vlm_rows: list[dict] | None, out_path: str,
                      tick_s: float = DEFAULT_TICK_S) -> dict:
    """Judgment accuracy vs fog -- can only be shown with VLM data (the
    oracle ignores fog entirely, so an oracle-only version would show a
    perfect, fog-invariant curve and misrepresent a real perception
    effect). If no VLM rows are available, writes a clearly-labeled
    placeholder figure instead of fabricating a dissociation.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    if not vlm_rows:
        ax.text(
            0.5, 0.5,
            "Awaiting VLM runs (eval/pit/data/vlm_runs.jsonl not found or empty).\n\n"
            "The oracle policy ignores fog by construction (it decides from the\n"
            "true window, not a rendered frame), so oracle data cannot show a\n"
            "perception-driven dissociation between judgment and timing.\n"
            "This figure will populate once real VLM episodes are recorded.",
            ha="center", va="center", fontsize=11, wrap=True,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title("Perception/dissociation (placeholder -- no VLM data yet)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[fig_dissociation] no VLM data available -- wrote placeholder to {out_path}")
        return {"placeholder": True}

    fogs = sorted({row["fog"] for row in vlm_rows})
    for fog in fogs:
        rows = [row for row in vlm_rows if row["fog"] == fog]
        r = np.array([ratio(row, tick_s) for row in rows])
        y = np.array([row["cleared"] for row in rows], dtype=float)
        order = np.argsort(r)
        ax.plot(r[order], y[order], marker="o", label=f"fog={fog}")

    ax.set_xlabel("r = speed * L_ticks / W")
    ax.set_ylabel("cleared")
    ax.set_title("Judgment/timing dissociation by fog (VLM data)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig_dissociation] wrote real VLM-data figure to {out_path}")
    return {"placeholder": False, "fogs": fogs}


def fig_flip(oracle_rows: list[dict], out_path: str, tick_s: float = DEFAULT_TICK_S) -> dict:
    """Same-decision flip: among decisions that were correct at commit time
    (`in_window == 1`, i.e. a zero-delay takeoff from that exact
    `decision_x` clears), P(clear) at `delay == 0` vs. P(clear) once the
    delay is large enough to blow the window (`r > 1`) -- the *same*
    decision flips from clearing to falling purely because of when it was
    executed.
    """
    def _dedupe_configs(rows):
        # The oracle emits identical rows across fog/seed (it ignores them), so
        # dedupe to unique (pit_width, delay, repeat) configs. n then reflects
        # independent configs, not the 60x fog/seed replication; proportions are
        # unchanged (the duplicates are identical).
        seen, uniq = set(), []
        for row in rows:
            key = (row["pit_width"], row["delay"], row["repeat"])
            if key not in seen:
                seen.add(key)
                uniq.append(row)
        return uniq

    usable = [row for row in oracle_rows if window_width(row) > 0 and row["in_window"] == 1]
    zero_delay = _dedupe_configs([row for row in usable if row["delay"] == 0])
    late_delay = _dedupe_configs([row for row in usable if ratio(row, tick_s) > 1.0])

    if not zero_delay or not late_delay:
        raise ValueError(
            f"fig_flip(): need both delay=0 (got {len(zero_delay)}) and "
            f"r>1 (got {len(late_delay)}) in-window rows to compare"
        )

    p_zero = float(np.mean([row["cleared"] for row in zero_delay]))
    p_late = float(np.mean([row["cleared"] for row in late_delay]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        ["delay = 0\n(n=%d)" % len(zero_delay), "delay > window (r>1)\n(n=%d)" % len(late_delay)],
        [p_zero, p_late], color=["#1b9e77", "#d95f02"],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(cleared | decision was in-window)")
    ax.set_title("Same decision, flipped outcome\n(oracle-correct decisions)")
    for i, v in enumerate([p_zero, p_late]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[fig_flip] P(clear|delay=0)={p_zero:.3f} P(clear|r>1)={p_late:.3f}")
    return {"p_delay_zero": p_zero, "p_delay_late": p_late}


def fig_repeat(rows: list[dict], out_path: str, tick_s: float = DEFAULT_TICK_S) -> dict:
    """r* vs. action-repeat k. `repeat` gates how often the policy is
    consulted; a larger repeat adds an implicit polling delay on top of the
    explicit `delay`, so the critical *explicit* delay needed to blow the
    window should shrink as k grows -- i.e. r* should decrease with k.
    Needs more than one distinct `repeat` value in `rows` (the default
    oracle sweep only has repeat=1; `main()` generates a small
    supplementary repeat sweep for this figure).
    """
    usable = [row for row in rows if window_width(row) > 0]
    repeats = sorted({row["repeat"] for row in usable})
    if len(repeats) < 2:
        raise ValueError(
            f"fig_repeat(): need >=2 distinct repeat values, got {repeats}"
        )

    ks, r_stars = [], []
    skipped = []
    for k in repeats:
        group = [row for row in usable if row["repeat"] == k]
        r = np.array([ratio(row, tick_s) for row in group])
        y = np.array([row["cleared"] for row in group], dtype=float)
        if len(set(y.tolist())) < 2:
            skipped.append(k)
            continue
        try:
            fit = transition_fit(r, y)
        except (RuntimeError, ValueError):
            skipped.append(k)
            continue
        ks.append(k)
        r_stars.append(fit["r_star"])

    if not ks:
        raise ValueError("fig_repeat(): no repeat group produced a usable fit")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ks, r_stars, marker="o", color="#1b9e77")
    ax.set_xlabel("action-repeat k")
    ax.set_ylabel("r*")
    ax.set_title("Timing threshold r* vs. action-repeat k")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if skipped:
        print(f"[fig_repeat] skipped repeat values with no usable fit: {skipped}")
    print(f"[fig_repeat] r* by k: {dict(zip(ks, r_stars))}")
    return {"k": ks, "r_star": r_stars, "skipped": skipped}


# ---------------------------------------------------------------------------
# Supplementary repeat sweep (the default oracle grid only has repeat=1)
# ---------------------------------------------------------------------------

def ensure_repeat_sweep(binary: str, out_path: str) -> list[dict]:
    """Return rows spanning multiple `repeat` values for `fig_repeat`,
    generating them via the real C sim binary if `out_path` doesn't already
    exist. Small, fast grid: one pit_width, one fog value, a delay sweep,
    and repeat in {1,2,4,8,16}. Real sim output, not fabricated.
    """
    existing = load_jsonl(out_path)
    if existing:
        return existing

    from eval.pit.config import grid
    from eval.pit.run_oracle_sweep import run_sweep

    configs = grid(
        pit_width=[2], fog=[0.0], delay=list(range(0, 41, 2)),
        repeat=[1, 2, 4, 8, 16], seed=[0],
    )
    run_sweep(binary, configs, out_path)
    return load_jsonl(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Map raw judge strings to short display names + a capability sort order. Any
# judge not listed sorts last, labelled by its raw string.
_JUDGE_DISPLAY = {
    "claude-haiku-4-5 (blind subagent)": ("Haiku 4.5", 0),
    "claude-sonnet (blind subagent)": ("Sonnet", 1),
    "claude-opus (blind subagent)": ("Opus", 2),
    "qwen2.5-vl-7b (local, Ollama)": ("Qwen2.5-VL 7B\n(local)", 3),
}


def fig_perception(rows: list[dict], out_path: str) -> dict:
    """Perception axis: real (blind) vision models' jump/wait judgment accuracy
    vs. fog, one clustered pair of bars per model. `rows` come from
    `perception_judgments.jsonl` — each a frame the judge saw with only the
    image, scored `correct` against the ground-truth `in_window` (a jump from
    that spot clears). Fog obscures the pit's distance, so judgment (at zero
    delay) degrades — the perception half of the perception-vs-timing
    dissociation. Capable models (Opus, Sonnet) read the depth cue when it is
    present and collapse toward chance under fog; a weaker model that never
    reads the cue sits at floor in both conditions."""
    # accuracy[judge][fog] = list of correct flags
    acc: dict[str, dict[float, list[int]]] = {}
    for row in rows:
        acc.setdefault(row["judge"], {}).setdefault(row["fog"], []).append(int(row["correct"]))

    fogs = sorted({row["fog"] for row in rows})
    judges = sorted(acc, key=lambda j: _JUDGE_DISPLAY.get(j, (j, 99))[1])
    labels = [_JUDGE_DISPLAY.get(j, (j, 99))[0] for j in judges]

    x = np.arange(len(judges))
    width = 0.8 / max(len(fogs), 1)
    colors = {min(fogs): "#1b9e77", max(fogs): "#d95f02"}
    fig, ax = plt.subplots(figsize=(7, 5))
    result: dict[str, dict[float, float]] = {}
    for k, f in enumerate(fogs):
        vals, ns = [], []
        for j in judges:
            flags = acc[j].get(f, [])
            vals.append(float(np.mean(flags)) if flags else 0.0)
            ns.append(len(flags))
        offset = (k - (len(fogs) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=colors.get(f, "#7570b3"),
                      label=f"clear (fog={f:.0f})" if f == min(fogs) else f"foggy (fog={f:.0f})")
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
        for lbl, f2, v in zip(labels, [f] * len(judges), vals):
            result.setdefault(lbl, {})[f2] = v

    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\n(n={sum(len(acc[j].get(f, [])) for f in fogs)})"
                        for lb, j in zip(labels, judges)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("judgment accuracy (jump/wait vs. true window)")
    ax.set_title("Perception axis: vision judgment vs. fog\n(blind vision models, zero delay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print("[fig_perception] accuracy by judge/fog: " + "; ".join(
        f"{lb} " + ",".join(f"fog{f:.0f}={result[lb][f]:.0%}" for f in fogs)
        for lb in [labels[i] for i in range(len(judges))]))
    return result


def fig_latency(latency_rows: list[dict], out_path: str, *,
                w_lo: float = 8.921, w_hi: float = 9.999, speed: float = 0.07,
                tick_s: float = DEFAULT_TICK_S, r_star: float = 1.1686) -> dict:
    """Real local-VLM latency vs. the task's time budget. A single decision's
    dimensionless cost is exactly r = L / T_window, where the action window is
    open for T_window = (w_hi - w_lo)/speed * tick_s seconds. The fitted collapse
    r* (from fig_timing) sets the *maximum tolerable* latency r**T_window; any L
    above it fails in real time. `latency_rows` come from `vlm_latency.jsonl`
    (per-decision `latency_s` measured against a local Ollama vision model)."""
    t_window = (w_hi - w_lo) / speed * tick_s   # seconds the window is open
    tolerable = r_star * t_window               # max latency that still clears

    # Group per-decision latencies by model (one strip + median line each).
    def _tag(model: str) -> str:
        return "API" if "API" in model else ("local" if "local" in model else "")

    def _short(model: str) -> str:
        name = model.split(" (")[0].replace(":11b", "").replace("-7b", "")
        t = _tag(model)
        return f"{name}\n({t})" if t else name

    by_model: dict[str, list[float]] = {}
    for row in latency_rows:
        by_model.setdefault(row.get("model", "local VLM"), []).append(float(row["latency_s"]))
    models = sorted(by_model, key=lambda m: np.median(by_model[m]))  # fastest first, slowest last
    # API points blue-ish, local points orange-ish, so the two regimes read apart.
    api_c, local_c = "#3767b0", "#d95f02"

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(models) + 1.8))
    rng = np.random.default_rng(0)
    yticks, ylabels = [], []
    result: dict[str, dict] = {}
    for i, model in enumerate(models):
        lat = sorted(by_model[model])
        med = float(np.median(lat))
        r_med = med / t_window
        y = i + 1
        color = api_c if _tag(model) == "API" else local_c
        ax.scatter(lat, y + rng.uniform(-0.16, 0.16, len(lat)), s=24, alpha=0.6,
                   color=color, zorder=3)
        ax.plot([med, med], [y - 0.30, y + 0.30], color=color, lw=2.5, zorder=4)
        ax.text(med * 1.09, y + 0.05, f"median {med:.1f}s → r≈{r_med:.0f}",
                color=color, fontsize=8, va="center")
        yticks.append(y)
        ylabels.append(_short(model))
        result[model] = {"median_L_s": med, "mean_L_s": float(np.mean(lat)), "r_median": r_med}

    # Time-budget reference lines (shared across models).
    ax.axvline(tick_s, color="#666", ls=":", lw=1.2, label=f"1 sim tick = {tick_s*1000:.0f}ms")
    ax.axvline(t_window, color="#1b9e77", ls="--", lw=1.5,
               label=f"action window open = {t_window*1000:.0f}ms")
    ax.axvline(tolerable, color="black", ls="-.", lw=1.5,
               label=f"max tolerable (r*={r_star:.2f}) = {tolerable*1000:.0f}ms")
    ax.set_xscale("log")
    alllat = [v for vs in by_model.values() for v in vs]
    ax.set_xlim(tick_s * 0.5, max(alllat) * 1.9)
    ax.set_ylim(0.4, len(models) + 0.9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("time per decision (s, log scale)")
    r_lo = min(result[m]["r_median"] for m in models)
    r_hi = max(result[m]["r_median"] for m in models)
    ax.set_title(f"Every real VLM is {r_lo:.0f}–{r_hi:.0f}x too slow to act in the window\n"
                 f"(r = L / window; window open only {t_window*1000:.0f}ms, "
                 f"max tolerable {tolerable*1000:.0f}ms — none clear it)")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    result["_task"] = {"t_window_s": t_window, "tolerable_L_s": tolerable, "r_star": r_star}
    print("[fig_latency] " + "; ".join(
        f"{_short(m).split(chr(10))[0]}: median={result[m]['median_L_s']:.2f}s r={result[m]['r_median']:.1f}"
        for m in models))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", default=DEFAULT_ORACLE)
    parser.add_argument("--vlm", default=DEFAULT_VLM)
    parser.add_argument("--perception", default="eval/pit/data/perception_judgments.jsonl")
    parser.add_argument("--latency", default="eval/pit/data/vlm_latency.jsonl")
    parser.add_argument("--aim", default="eval/pit/data/aim_sweep.jsonl")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--repeat-sweep", default=DEFAULT_REPEAT_SWEEP)
    parser.add_argument("--binary", default=os.environ.get("KEEN_BIN", "raycaster/build/keen-raycaster"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_rows = load_jsonl(args.oracle)
    if not oracle_rows:
        raise SystemExit(
            f"No rows found at {args.oracle!r}. Generate it first: "
            f"uv run python -m eval.pit.run_oracle_sweep"
        )
    vlm_rows = load_jsonl(args.vlm)

    fig_timing(oracle_rows, out_dir / "fig_timing.png", vlm_rows=vlm_rows)
    fig_dissociation(vlm_rows, out_dir / "fig_dissociation.png")
    fig_flip(oracle_rows, out_dir / "fig_flip.png")

    perception_rows = load_jsonl(args.perception)
    if perception_rows:
        fig_perception(perception_rows, out_dir / "fig_perception.png")
    else:
        print(f"[fig_perception] skipped -- no data at {args.perception!r}")

    latency_rows = load_jsonl(args.latency)
    if latency_rows:
        fig_latency(latency_rows, out_dir / "fig_latency.png")
    else:
        print(f"[fig_latency] skipped -- no data at {args.latency!r}")

    aim_rows = load_jsonl(args.aim)
    if aim_rows:
        fig_aim(aim_rows, out_dir / "fig_aim.png")
    else:
        print(f"[fig_aim] skipped -- no data at {args.aim!r}")

    try:
        repeat_rows = ensure_repeat_sweep(args.binary, args.repeat_sweep)
        fig_repeat(repeat_rows, out_dir / "fig_repeat.png")
    except (FileNotFoundError, OSError) as exc:
        print(f"[fig_repeat] skipped -- could not run sim binary {args.binary!r}: {exc}")

    print(f"Done: figures written to {out_dir}")


if __name__ == "__main__":
    main()
