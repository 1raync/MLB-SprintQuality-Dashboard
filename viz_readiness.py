"""Step 3 figure: self-baseline readiness trend for the batter's running output.

Two stacked panels (see PROJECT_BRIEF.md sec 3.4):
  A. Per-qualifying-run Sprint Speed (ft/s) over the month vs the player's OWN baseline, with
     the rolling baseline and the green/amber/red traffic-light bands (mean, -1 SD, -2 SD).
     This is the "is he running like himself?" view, in interpretable ft/s.
  B. Per-GAME readiness: the composite output index (z vs baseline) as a traffic light, marker
     style = n_qualifying confidence, with prior-7-day baserunning load overlaid so the
     "after lots of load" context is visible alongside the output.

Pulls live numbers from readiness.py (no stale CSV). Bands are tunable conventions, NOT
validated cut-offs (brief sec 3.4).

Run: python viz_readiness.py  ->  figures/readiness_trend.png
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import readiness as rd

LIGHT = {"green": "tab:green", "amber": "tab:orange", "red": "tab:red", "no-read": "0.7"}


def _date_jitter(dates: pd.Series) -> np.ndarray:
    """Deterministic small x-offset so multiple runs on one game-date don't overplot."""
    dates = pd.to_datetime(dates)
    x = dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    out = x.copy()
    for d in np.unique(x):
        idx = np.where(x == d)[0]
        out[idx] = d + (np.arange(len(idx)) - (len(idx) - 1) / 2) * 0.16
    return out


def main():
    os.makedirs("figures", exist_ok=True)
    runs = rd.build_run_table()
    games = rd.build_game_table(runs)

    q = runs[runs["qualifying"]].sort_values(["game_date", "play_id"]).reset_index(drop=True)
    nq = runs[~runs["qualifying"]]
    mu, sd = q["F1_peak1s_ftps"].mean(), q["F1_peak1s_ftps"].std(ddof=1)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.25, 1])

    # ---------- Panel A: per-run Sprint Speed vs self-baseline ----------
    qx = _date_jitter(q["game_date"]); nqx = _date_jitter(nq["game_date"])
    # traffic-light bands (self-baseline): green >= mu-1SD, amber mu-2SD..mu-1SD, red < mu-2SD
    axA.axhspan(mu - sd, mu + 3 * sd, color="tab:green", alpha=0.07)
    axA.axhspan(mu - 2 * sd, mu - sd, color="tab:orange", alpha=0.09)
    axA.axhspan(mu - 5 * sd, mu - 2 * sd, color="tab:red", alpha=0.08)
    axA.axhline(mu, color="0.35", lw=1.3, label=f"baseline mean ({mu:.1f} ft/s)")
    axA.axhline(mu - sd, color="0.6", lw=0.9, ls="--")
    axA.axhline(mu - 2 * sd, color="0.6", lw=0.9, ls=":")
    axA.axhline(27, color="tab:blue", lw=1.0, ls="-.", alpha=0.7, label="MLB avg 27 ft/s")

    # rolling baseline over qualifying runs (trailing ROLL_N), in chronological order
    axA.plot(qx, q["F1_rolling_baseline"], color="crimson", lw=1.8,
             label=f"rolling baseline (trailing {rd.ROLL_N} runs)")
    sc = axA.scatter(qx, q["F1_peak1s_ftps"], c=q["effort_score"], cmap="viridis",
                     vmin=0.5, vmax=1.0, s=70, edgecolor="0.2", zorder=5, label="qualifying run")
    axA.scatter(nqx, nq["F1_peak1s_ftps"], marker="x", c="0.6", s=40, alpha=0.7,
                zorder=4, label="non-qualifying (sub-max)")
    cb = fig.colorbar(sc, ax=axA, pad=0.01); cb.set_label("effort score")

    axA.set_ylabel("Sprint Speed — peak 1-s (ft/s)")
    axA.set_title("A. Running output vs the player's own baseline (each qualifying run; "
                  "green/amber/red = within 1 / 1–2 / >2 SD below his mean)", fontsize=11)
    axA.set_ylim(min(runs["F1_peak1s_ftps"].min() - 1, mu - 2.5 * sd), mu + 2.2 * sd)
    axA.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.95)

    # ---------- Panel B: per-game readiness + prior-7d load ----------
    g = games.copy()
    gx = pd.to_datetime(g["game_date"]).map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    axB.axhspan(rd.GREEN_Z, 4, color="tab:green", alpha=0.07)
    axB.axhspan(rd.RED_Z, rd.GREEN_Z, color="tab:orange", alpha=0.09)
    axB.axhspan(-5, rd.RED_Z, color="tab:red", alpha=0.08)
    axB.axhline(0, color="0.35", lw=1.3)
    axB.axhline(rd.GREEN_Z, color="0.6", lw=0.9, ls="--")
    axB.axhline(rd.RED_Z, color="0.6", lw=0.9, ls=":")

    for _, r in g.iterrows():
        x = pd.Timestamp(r["game_date"]).toordinal()
        z = r["output_index_z"]
        col = LIGHT[r["readiness"]]
        if pd.isna(z):
            axB.scatter(x, 0, marker="s", facecolor="none", edgecolor="0.6", s=80, zorder=5)
            axB.annotate("no-read\n(0 qual runs)", (x, 0), fontsize=6.5, ha="center",
                         va="bottom", color="0.5")
            continue
        # filled = ok confidence; hollow = single-observation (n_qual < 2)
        ok = r["confidence"] == "ok"
        axB.scatter(x, z, marker="o", s=130, zorder=6,
                    facecolor=col if ok else "none", edgecolor=col, linewidths=2,
                    alpha=0.95 if ok else 0.9)
        axB.annotate(f"n={r['n_qual']}", (x, z), fontsize=7, ha="center",
                     va="bottom" if z >= 0 else "top",
                     xytext=(0, 6 if z >= 0 else -6), textcoords="offset points")
    axB.plot(gx, g["output_index_z"].to_numpy(dtype=float), color="0.4", lw=1.0, alpha=0.5, zorder=3)

    # prior-7-day baserunning load on a twin axis (context for "after lots of load")
    axL = axB.twinx()
    axL.bar(gx, g["load_prior7d_ft"], width=0.6, color="0.5", alpha=0.18, zorder=1)
    axL.set_ylabel("prior-7d baserunning load (ft, path)", color="0.45")
    axL.tick_params(axis="y", labelcolor="0.45")

    axB.set_ylabel("game output index (z vs baseline)")
    axB.set_ylim(-3, 4)
    axB.set_title("B. Per-game readiness (sprint speed + burst, effort-weighted) with recent "
                  "load — hollow = single-run (low confidence)", fontsize=11)

    # shared date ticks
    all_ord = sorted(pd.to_datetime(runs["game_date"]).map(pd.Timestamp.toordinal).unique())
    for ax in (axA, axB):
        ax.set_xticks(all_ord)
        ax.set_xticklabels([pd.Timestamp.fromordinal(int(o)).strftime("%m-%d") for o in all_ord],
                           rotation=45, fontsize=8)
        ax.grid(True, axis="x", ls=":", alpha=0.35)
    axB.set_xlabel("game date (2025)")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=LIGHT[c], markersize=10,
                      label=c) for c in ("green", "amber", "red")]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.4", markersize=10,
                       markeredgecolor="0.4", label="filled = ≥2 qual runs"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markersize=10,
                       markeredgecolor="0.4", label="hollow = single run")]
    axB.legend(handles=handles, loc="lower left", fontsize=8, ncol=2, framealpha=0.95)

    fig.suptitle("Self-baseline readiness trend — single batter, May 2025 "
                 "(decision support, not an automated benching rule)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = "figures/readiness_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"baseline: {mu:.2f} ± {sd:.2f} ft/s (qualifying runs, n={len(q)})  "
          f"amber<{mu-sd:.1f}  red<{mu-2*sd:.1f}")


if __name__ == "__main__":
    main()
