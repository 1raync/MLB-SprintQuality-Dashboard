"""Step 3: per-game physical-output & readiness from the batter's max-effort runs.

Rolls the per-run Sprint Speed (F1) and burst (F2) up to a per-GAME readout of whether the
batter is performing as physically well -- normalized to his OWN baseline and gated to
max-effort runs (effort_score from cluster_runs) so that play context (an easy out he jogs
out) doesn't masquerade as fatigue. Also computes a home-to-first proxy, but reports it as a
CONFIRMATORY metric only (see the redundancy correlation printed in the report).

Why effort-gating is the whole game (see PROJECT_BRIEF.md sec 3.3-3.4): a per-game mean over
ALL runs is dominated by which plays happened, not by fitness -- coast-out groundballs sit at
11-17 ft/s next to 28 ft/s sprints. We compare like-for-like by keeping only runs the player
was plausibly going all-out on (effort_score >= QUAL_EFFORT).

Why home-to-first is demoted: true Statcast Home-to-First needs the bat-on-ball CONTACT
timestamp (not in this data); the best proxy is "time to cover 90 ft from the first step" =
the 90-ft running split. But that proxy is ~burst + sprint speed integrated over 90 ft, so it
is redundant with F1+F2 (the report prints |r|), and the distinct, fatigue-sensitive part of
true H2F (reaction/initiation from contact) is exactly the part we cannot measure.

Sample reality: ~1 max-effort run per game (3 of 11 games have zero near-max runs), so a
per-game value is ONE noisy observation. Every game read carries an n_qual confidence flag,
and the trustworthy signal is the rolling trend across runs, not any single game.

Run: python readiness.py  ->  per_run_output.csv + game_readiness.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import load_data as ld
import run_features as rf
import cluster_runs as cr

# --- DECIDED constants (tunable; logged in the writeup) ------------------------------------
# DECIDED: physical output = F1 Sprint Speed (primary) + F2 burst (secondary). Both are
#   first-derivative/integral metrics that are reliable at 30 fps. Home-to-first is computed
#   but kept CONFIRMATORY (redundant; true H2F uncomputable). Accel/decel EVENT COUNTS are
#   excluded -- the 2nd derivative is noise-dominated at 30 fps (see brief sec 3.3 callout).
# DECIDED: qualifying (max-effort) run = effort_score >= 0.5. Rationale: separates all-out runs
#   from leg-outs/trots so a speed drop reflects the athlete, not the play. Rejected: a fixed
#   %-of-Vmax gate (ignores the run's shape) and using all runs (effort-context confounded).
QUAL_EFFORT = 0.5
ACUTE_DAYS = 7              # "recent load" window entering a game (baserunning load only)
ROLL_N = 5                 # trailing qualifying runs for the rolling baseline trend
H2F_DIST_FT = 90.0         # home-to-first distance (one base path)
# DECIDED: traffic light on the baseline z-score (self-referenced, CV-style). Bands are a
#   TUNABLE convention (brief sec 3.4), not validated cut-offs: green >= -0.5 SD, amber
#   -1..-0.5, red <= -1 SD. Always shown WITH the n_qual confidence flag.
GREEN_Z, RED_Z = -0.5, -1.0


def home_to_first_proxy(play_df: pd.DataFrame) -> float:
    """Time (s) to cover the first 90 ft of CoM path from the first step -- the 90-ft running
    split, a proxy for (not equal to) Statcast Home-to-First. NaN if the run never reaches 90 ft.
    Uses cumulative path length so a slightly curved baseline still counts 90 ft of running."""
    t, pos, speed = rf.com_speed(play_df)
    xy_raw = play_df.sort_values("t_sec")[["MidHip_x", "MidHip_y"]].to_numpy()
    s = rf.segment_run(t, xy_raw, speed)["start"]
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos[s:], axis=0), axis=1))])
    idx = int(np.searchsorted(cum, H2F_DIST_FT))
    return float(t[s + idx] - t[s]) if idx < len(cum) else np.nan


def hsr_distance(play_df: pd.DataFrame, thr_ftps: float) -> float:
    """High-speed-running distance (ft): CoM path covered while smoothed speed exceeds thr_ftps."""
    t, pos, speed = rf.com_speed(play_df)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return float(step[speed[:-1] > thr_ftps].sum())


def time_to_90(play_df: pd.DataFrame) -> float:
    """Time (s) from first step to first reaching 90% of the run's peak CoM speed - the
    acceleration-phase duration. Always defined (the peak lies inside the run window)."""
    t, pos, speed = rf.com_speed(play_df)
    xy_raw = play_df.sort_values("t_sec")[["MidHip_x", "MidHip_y"]].to_numpy()
    seg = rf.segment_run(t, xy_raw, speed)
    s, e = seg["start"], seg["end"]
    w = speed[s:e + 1]
    reach = np.where(w >= 0.9 * float(w.max()))[0]
    return float(t[s + reach[0]] - t[s]) if len(reach) else np.nan


def build_run_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-run table for the captured runs: F1/F2, effort_score, home-to-first proxy, and each
    metric z-scored against the player's own qualifying-run baseline."""
    if df is None:
        df = ld.load_reshaped()
    feats = rf.build_feature_table(df)
    runs, X, _ = cr.build_matrix(feats)               # is_run subset + scaled effort-shape matrix

    # effort_score (GMM posterior of the max-effort component) -- reuse cluster_runs, no figures
    k, _ = cr.select_k(X)
    f1 = runs["F1_peak1s_ftps"].to_numpy()
    wl = cr.order_tiers_by_speed(cr.ward_labels(X, k), f1)
    gmm = cr.gmm_fit(X, k)
    runs["effort_score"], _ = cr.effort_scores(X, gmm, wl, f1)

    runs["h2f_90ft_s"] = [home_to_first_proxy(df[df["play_id"] == pid]) for pid in runs["play_id"]]
    runs["game_date"] = pd.to_datetime(runs["game_date"])
    runs["qualifying"] = runs["effort_score"] >= QUAL_EFFORT

    # high-speed-running distance per run (> 75% of his Vmax) - feeds the prior-7d workload
    vmax = float(runs["F1_peak1s_ftps"].max())
    runs["hsr_ft"] = [hsr_distance(df[df["play_id"] == pid], 0.75 * vmax) for pid in runs["play_id"]]
    runs["t90_s"] = [time_to_90(df[df["play_id"] == pid]) for pid in runs["play_id"]]

    # Baseline = mean/SD over the player's QUALIFYING runs (his max-effort norm). Full-sample
    # (not expanding) for a stable descriptive baseline given small N; rolling trend added below.
    q = runs[runs["qualifying"]]
    for col, lo_is_good in [("F1_peak1s_ftps", False), ("F2_burst_ft", False), ("h2f_90ft_s", True),
                            ("hsr_ft", False), ("t90_s", True)]:
        mu, sd = q[col].mean(), q[col].std(ddof=1)
        z = (runs[col] - mu) / sd
        # orient so + = physically better (faster speed/burst/HSR; SHORTER H2F & time-to-90%)
        runs[col + "_z"] = -z if lo_is_good else z

    # composite output index = mean of the 4 effort axes (H2F excluded: confirmatory/redundant)
    runs["output_index_z"] = runs[["F1_peak1s_ftps_z", "F2_burst_ft_z",
                                    "hsr_ft_z", "t90_s_z"]].mean(axis=1)

    # rolling baseline trend over qualifying runs in chronological order (trailing ROLL_N)
    runs = runs.sort_values(["game_date", "play_id"]).reset_index(drop=True)
    qmask = runs["qualifying"]
    roll = (runs.loc[qmask, "F1_peak1s_ftps"]
            .rolling(ROLL_N, min_periods=2).mean())
    runs["F1_rolling_baseline"] = roll
    return runs


def _light(z: float) -> str:
    if np.isnan(z):
        return "no-read"
    return "green" if z >= GREEN_Z else ("red" if z <= RED_Z else "amber")


def _load_light(load: float, lo: float, hi: float) -> str:
    """Load indicator vs his own tertiles: low = green (fresh), mid = amber, high = red."""
    return "green" if load <= lo else ("red" if load >= hi else "amber")


def _readiness_rollup(lights) -> str:
    """Next-day readiness from the indicator lights (load, sprint, burst, time-to-90):
    a red plus any other flag, or two reds -> red (Not Recovered); two amber -> amber
    (Borderline); a single flag -> green. (A lone red staying green is the literal rule; flip
    the `A >= 2` line to `(A + R) >= 2 or R >= 1` if a single red should read amber.)"""
    A = sum(1 for x in lights if x == "amber")
    R = sum(1 for x in lights if x == "red")
    if R >= 2 or (R >= 1 and A >= 1):
        return "red"
    if A >= 2:
        return "amber"
    return "green"


def build_game_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-game next-day readiness. Each game's effort-weighted metrics (sprint, burst, time-to-90)
    and the prior-7-day high-speed-running load are classified green/amber/red vs his own baseline;
    the readiness light rolls those 4 indicators up (_readiness_rollup). Days-rest is context, not
    in the roll-up. Output also carries the composite z (for context) and an n_qual confidence flag."""
    runs = runs.sort_values("game_date")
    # workload entering a game = high-speed-running distance (>75% Vmax) over the prior 7 days
    per_game_load = (runs.groupby("game_date")
                     .agg(total_run_dist=("hsr_ft", "sum"), n_runs=("play_id", "size")))
    dates = list(per_game_load.index)
    rows = []
    for i, d in enumerate(dates):
        g = runs[runs["game_date"] == d]
        q = g[g["qualifying"]]
        w = q["effort_score"]
        def wmean(col):
            return float(np.average(q[col], weights=w)) if len(q) and w.sum() > 0 else np.nan
        # high-speed-running distance ENTERING this game: sum over the prior ACUTE_DAYS (excl. today)
        prior = per_game_load[(per_game_load.index >= d - pd.Timedelta(days=ACUTE_DAYS))
                              & (per_game_load.index < d)]
        rows.append(dict(
            game_date=d.date(),
            n_runs=int(len(g)), n_qual=int(len(q)),
            sprint_ftps=wmean("F1_peak1s_ftps"), burst_ft=wmean("F2_burst_ft"), t90_s=wmean("t90_s"),
            F1_z=wmean("F1_peak1s_ftps_z"), burst_z=wmean("F2_burst_ft_z"),
            hsr_z=wmean("hsr_ft_z"), t90_z=wmean("t90_s_z"), h2f_z=wmean("h2f_90ft_s_z"),
            output_index_z=wmean("output_index_z"),
            confidence=("low(n<2)" if len(q) < 2 else "ok"),
            days_rest=(np.nan if i == 0 else (d - dates[i - 1]).days),
            load_prior7d_ft=float(prior["total_run_dist"].sum()),
            runs_prior7d=int(prior["n_runs"].sum()),
        ))
    df = pd.DataFrame(rows)

    # readiness = roll-up of 4 indicator lights: prior-7d load (his own tertiles) + sprint, burst,
    # time-to-90% (vs baseline). Days-rest is deliberately NOT counted (it's schedule context).
    loads = df["load_prior7d_ft"].to_numpy(dtype=float)
    lo, hi = float(np.quantile(loads, 1 / 3)), float(np.quantile(loads, 2 / 3))
    light, n_amber, n_red = [], [], []
    for _, r in df.iterrows():
        if np.isnan(r["output_index_z"]):          # no qualifying run -> can't assess
            light.append("no-read"); n_amber.append(0); n_red.append(0); continue
        inds = [_load_light(r["load_prior7d_ft"], lo, hi),
                _light(r["F1_z"]), _light(r["burst_z"]), _light(r["t90_z"])]
        light.append(_readiness_rollup(inds))
        n_amber.append(sum(1 for x in inds if x == "amber"))
        n_red.append(sum(1 for x in inds if x == "red"))
    df["readiness"], df["n_amber"], df["n_red"] = light, n_amber, n_red
    return df


def main():
    runs = build_run_table()
    q = runs[runs["qualifying"]]
    Vmax = runs["F1_peak1s_ftps"].max()

    print("=== readiness.py - Step 3: per-game physical output & readiness ===\n")
    print(f"captured runs: {len(runs)}   qualifying (effort_score>={QUAL_EFFORT}): {len(q)}   "
          f"Vmax (F1) = {Vmax:.1f} ft/s")

    # --- redundancy check: are the 3 requested metrics independent? (justifies demoting H2F) ---
    h = q.dropna(subset=["h2f_90ft_s"])
    r_fb = q["F1_peak1s_ftps"].corr(q["F2_burst_ft"])
    r_fh = h["F1_peak1s_ftps"].corr(h["h2f_90ft_s"])
    r_bh = h["F2_burst_ft"].corr(h["h2f_90ft_s"])
    print("\n--- metric redundancy across qualifying runs (Pearson r) ---")
    print(f"  Sprint speed (F1) vs burst (F2):      r = {r_fb:+.2f}")
    print(f"  Sprint speed (F1) vs home-to-first:   r = {r_fh:+.2f}  (neg = faster -> shorter time)")
    print(f"  burst (F2) vs home-to-first:          r = {r_bh:+.2f}")
    print(f"  -> Sprint speed & burst are {'near-independent' if abs(r_fb) < 0.4 else 'correlated'} "
          f"(|r|={abs(r_fb):.2f}): distinct facets (top-end vs acceleration) -> KEEP BOTH.")
    print(f"  -> Home-to-first is {'collinear with sprint speed' if abs(r_fh) > 0.7 else 'partly independent'} "
          f"(r={r_fh:+.2f}) -> redundant; report as CONFIRMATORY only, not a third signal.")

    games = build_game_table(runs)
    print("\n--- per-game readiness (output_index_z = mean of F1_z & burst_z, effort-weighted) ---")
    show = games[["game_date", "n_runs", "n_qual", "F1_z", "burst_z", "h2f_z",
                  "output_index_z", "readiness", "confidence", "days_rest",
                  "load_prior7d_ft", "runs_prior7d"]]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(show.round(2).to_string(index=False))

    n_lowconf = int((games["confidence"] != "ok").sum())
    print(f"\n[sample caveat] {n_lowconf}/{len(games)} games have <2 qualifying runs -> single-obs "
          f"reads; trust the rolling trend, not one game.")

    # does recent baserunning load track lower output? (descriptive; n is tiny)
    gg = games.dropna(subset=["output_index_z"])
    if len(gg) >= 4:
        r = gg["load_prior7d_ft"].corr(gg["output_index_z"])
        print(f"[load vs output] corr(prior-7d run load, output_index_z) = {r:+.2f} "
              f"over {len(gg)} games with a read (descriptive only; baserunning load is a partial proxy).")

    runs.to_csv("per_run_output.csv", index=False)
    games.to_csv("game_readiness.csv", index=False)
    print(f"\nwrote per_run_output.csv ({len(runs)} runs) and game_readiness.csv ({len(games)} games)")


if __name__ == "__main__":
    main()
