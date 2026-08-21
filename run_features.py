"""Step 2a: per-run physical-output features for the batter's plays.

For every play we segment the run out of the box and compute a small set of kinematic
features from the smoothed center-of-mass (MidHip) trajectory. These features feed the
effort/run-type clustering (cluster_runs.py) and, later, the Sprint Speed / workload steps.

Pipeline per play (see PROJECT_BRIEF.md sec 3.5): FILTER (zero-lag Butterworth) -> DIFFERENTIATE
(np.gradient on the measured t_sec, never an assumed 1/30) -> segment -> feature scalars.

Two segmentation gates with DISTINCT jobs (don't conflate them):
  LEAVE_BOX_FT   trims the run START within a play (skips the swing).
  RUN_MIN_DISP_FT gates IS-THIS-A-RUN. It must be large: no-contact plays still displace
                  ~13-23 ft (swing + recoil step), so a 3 ft gate would call all 50 plays runs.

Units are FEET / ft per second throughout (see CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

import load_data as ld

# --- Trusted diamond geometry (mirrors viz_all_runs.py; viz files left untouched) ---
SIDE = 90.0
HOME = np.array([0.0, 0.0])
B1 = SIDE * np.array([np.cos(np.radians(45)), np.sin(np.radians(45))])    # (63.6, 63.6)
B2 = np.array([0.0, SIDE * np.sqrt(2)])                                   # (0, 127.3)
B3 = SIDE * np.array([np.cos(np.radians(135)), np.sin(np.radians(135))])  # (-63.6, 63.6)
BASES = {"1B": B1, "2B": B2, "3B": B3}

# --- Segmentation / feature constants (DECIDED; tunable, logged in the writeup) ---
# DECIDED: RUN_MIN_DISP_FT=40 ft is the is-this-a-run gate. Rationale: no-contact plays peak at
#   ~23 ft of CoM displacement, real runs at ~100+ ft, so 40 sits in the empty gap. Rejected:
#   reusing LEAVE_BOX_FT=3 as the run gate -> would mislabel all 50 plays as runs.
LEAVE_BOX_FT = 3.0            # CoM is this far from stance => the run has started (skip the swing)
RUN_MIN_DISP_FT = 40.0       # max CoM displacement above this => a real run out of the box
REACH_FT = 6.0               # CoM within this of a bag => "reached" that base
FIRST_STEP_SPEED_FTPS = 3.0  # speed gate, combined with LEAVE_BOX_FT, to pin the first-step frame
STANCE_WINDOW_S = 0.5        # median CoM over the first 0.5 s = pre-movement stance origin

# DECIDED: Butterworth cutoff fc=6 Hz, order 2 (filtfilt => effective 4th order, zero phase).
#   Rationale: CoM speed is low-frequency (stride bounce ~2-2.5 Hz); 6 Hz keeps it and kills
#   keypoint jitter, well under the 15 Hz Nyquist of 30 fps. Chosen by residual analysis
#   (residual_analysis() below). Rejected: higher cutoffs pass jitter into the derivative.
BUTTER_CUTOFF_HZ = 6.0
BUTTER_ORDER = 2

BURST_S = 1.5                # acceleration window: feet covered in first 1.5 s (Statcast "burst")
SUSTAIN_FRAC = 0.9           # "near peak" = >= 90% of F1
PEAK_WINDOW_S = 1.0          # Statcast Sprint Speed window
TERMINAL_WINDOW_S = 1.0      # window before arrival for braking decel + slide check
SLIDE_Z_DROP_FT = 1.3        # CoM (MidHip_z) drop vs running baseline suggesting a slide/dive


# --------------------------------------------------------------------------------------
# Smoothing + differentiation
# --------------------------------------------------------------------------------------
def _sampling_rate(t: np.ndarray) -> float:
    """Effective fps from the measured timestamps (handles the small ~30 fps jitter)."""
    return 1.0 / float(np.median(np.diff(t)))


def _butter_lowpass(sig: np.ndarray, fs: float, fc: float = BUTTER_CUTOFF_HZ,
                    order: int = BUTTER_ORDER) -> np.ndarray:
    """Zero-lag (forward-backward) Butterworth low-pass. filtfilt => no phase shift, so peak
    speed stays time-aligned with position."""
    n = len(sig)
    if n < 2 * (order + 1):           # too short to filter; return as-is (won't happen here)
        return sig.astype(float).copy()
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        padlen = n - 1
    return filtfilt(b, a, sig, padlen=padlen)


def com_speed(play_df: pd.DataFrame, fc: float = BUTTER_CUTOFF_HZ):
    """Smoothed horizontal CoM (MidHip) trajectory and speed for one play.

    Returns (t, pos_filt[N,2], speed[N]) in seconds, feet, ft/s. Filter the positions, then
    differentiate with np.gradient over the actual t_sec spacing.
    """
    d = play_df.sort_values("t_sec")
    t = d["t_sec"].to_numpy()
    xy = d[["MidHip_x", "MidHip_y"]].to_numpy()
    fs = _sampling_rate(t)
    pos = np.column_stack([_butter_lowpass(xy[:, 0], fs, fc),
                           _butter_lowpass(xy[:, 1], fs, fc)])
    vel = np.gradient(pos, t, axis=0)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    return t, pos, speed


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------
def _furthest_base(xy: np.ndarray):
    """Name of the furthest base the CoM passed within REACH_FT of, or None."""
    reached = [name for name in ("1B", "2B", "3B")
               if np.linalg.norm(xy - BASES[name], axis=1).min() <= REACH_FT]
    return reached[-1] if reached else None


def segment_run(t: np.ndarray, xy_raw: np.ndarray, speed: np.ndarray) -> dict:
    """Find the run window within a play.

    start = first frame past LEAVE_BOX_FT AND above the first-step speed gate (skips the swing).
    end   = closest approach to the furthest base reached, else the furthest-displacement frame.
    is_run = max CoM displacement from stance exceeds RUN_MIN_DISP_FT.
    """
    stance = np.median(xy_raw[t <= t[0] + STANCE_WINDOW_S], axis=0)
    disp = np.linalg.norm(xy_raw - stance, axis=1)
    max_disp = float(disp.max())

    moving = (disp > LEAVE_BOX_FT) & (speed > FIRST_STEP_SPEED_FTPS)
    start = int(np.argmax(moving)) if moving.any() else 0

    base = _furthest_base(xy_raw)
    end = (int(np.argmin(np.linalg.norm(xy_raw - BASES[base], axis=1)))
           if base is not None else int(np.argmax(disp)))
    if end <= start:                     # degenerate (no-run) -> use the whole remaining clip
        end = len(t) - 1
    return dict(is_run=max_disp > RUN_MIN_DISP_FT, start=start, end=end,
                stance=stance, max_disp=max_disp, base=base)


# --------------------------------------------------------------------------------------
# Feature primitives
# --------------------------------------------------------------------------------------
def peak_1s_speed(t: np.ndarray, pos: np.ndarray, win_s: float = PEAK_WINDOW_S) -> float:
    """Statcast Sprint Speed form: max over rolling <=win_s windows of
    ||pos[j] - pos[i]|| / (t[j] - t[i]).  Displacement-over-elapsed-time is robust to per-frame
    jitter (unlike max of an instantaneous derivative)."""
    best = 0.0
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + win_s, side="right")) - 1
        if j > i:
            best = max(best, np.linalg.norm(pos[j] - pos[i]) / (t[j] - t[i]))
    return best


def _burst(t: np.ndarray, pos: np.ndarray, start: int, win_s: float = BURST_S) -> float:
    """Straight-line CoM displacement in the first win_s after the first step (Statcast burst)."""
    j = int(np.searchsorted(t, t[start] + win_s, side="right")) - 1
    j = min(max(j, start), len(t) - 1)
    return float(np.linalg.norm(pos[j] - pos[start]))


def play_features(play_df: pd.DataFrame) -> dict:
    """All per-play features. Effort-axis features (F1,F2,F3,F5,F7) feed clustering; the rest
    are descriptors (NOT clustered) or annotations (slide_flag)."""
    d = play_df.sort_values("t_sec").reset_index(drop=True)
    t, pos, speed = com_speed(d)
    xy_raw = d[["MidHip_x", "MidHip_y"]].to_numpy()
    z = d["MidHip_z"].to_numpy()
    seg = segment_run(t, xy_raw, speed)
    s, e = seg["start"], seg["end"]

    # F1 peak 1-s sprint speed (ft/s) -- the primary intensity axis, computed over the whole play
    f1 = peak_1s_speed(t, pos)
    # F2 burst: feet covered in first 1.5 s off the first step (acceleration axis)
    f2 = _burst(t, pos, s)
    # F3 straightness: net / path over the run window (1 = straight to base, <1 = curved/rounding)
    win = pos[s:e + 1]
    path_len = float(np.sum(np.linalg.norm(np.diff(win, axis=0), axis=1))) if len(win) > 1 else 0.0
    net_disp = float(np.linalg.norm(win[-1] - win[0])) if len(win) > 1 else 0.0
    f3 = net_disp / path_len if path_len > 0 else np.nan
    # F5 sustain fraction: share of the run window held at >= 90% of F1 (max-effort hold vs coast)
    win_speed = speed[s:e + 1]
    f5 = float(np.mean(win_speed >= SUSTAIN_FRAC * f1)) if f1 > 0 and len(win_speed) else np.nan
    # F7 braking deceleration near arrival (ft/s^2, reported as a positive magnitude)
    accel = np.gradient(speed, t)
    term = (t >= t[e] - TERMINAL_WINDOW_S) & (t <= t[e])
    f7 = float(-accel[term].min()) if term.any() else np.nan

    # slide/dive annotation (heuristic, low reliability at 30 fps -- kept OUT of the cluster vector)
    base_z = float(np.median(z[s:e + 1])) if e > s else float(np.median(z))
    term_z_min = float(z[term].min()) if term.any() else float(z[-3:].min())
    z_drop = base_z - term_z_min
    slide = bool(z_drop > SLIDE_Z_DROP_FT and z[e] < base_z - 1.0)

    return {
        "play_id": d["play_id"].iloc[0],
        "pitch_result": d["pitch_result"].iloc[0],
        "game_date": d["frame_timestamp"].iloc[0].date(),
        "is_run": bool(seg["is_run"]),
        # --- clustering vector (effort axes) ---
        "F1_peak1s_ftps": f1,
        "F2_burst_ft": f2,
        "F3_straightness": f3,
        "F5_sustain_frac": f5,
        "F7_brake_decel_ftps2": f7,
        # --- annotation ---
        "slide_flag": slide,
        "z_drop_ft": z_drop,
        # --- descriptors (NOT clustered; kept for interpretation/validation) ---
        "base_reached": seg["base"],
        "max_disp_ft": seg["max_disp"],
        "path_len_ft": path_len,
        "net_disp_ft": net_disp,
        "run_dur_s": float(t[e] - t[s]),
    }


def build_feature_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-play feature table for all plays. `is_run` marks the 29 real runs that clustering uses."""
    if df is None:
        df = ld.load_reshaped()
    rows = [play_features(df[df["play_id"] == pid]) for pid in sorted(df["play_id"].unique())]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Filter-cutoff justification (residual analysis) + Savitzky-Golay cross-check
# --------------------------------------------------------------------------------------
def residual_analysis(df: pd.DataFrame, cutoffs=range(3, 13)) -> pd.DataFrame:
    """RMS of (raw - filtered) MidHip position vs cutoff, averaged over run plays. Pick the
    cutoff at the knee where the residual stops dropping like noise (Winter's method)."""
    runs = [pid for pid in sorted(df["play_id"].unique())
            if build_one_is_run(df, pid)]
    out = []
    for fc in cutoffs:
        res = []
        for pid in runs:
            d = df[df["play_id"] == pid].sort_values("t_sec")
            t = d["t_sec"].to_numpy()
            fs = _sampling_rate(t)
            for ax in ("MidHip_x", "MidHip_y"):
                raw = d[ax].to_numpy()
                res.append(np.sqrt(np.mean((raw - _butter_lowpass(raw, fs, fc)) ** 2)))
        out.append({"cutoff_hz": fc, "rms_residual_ft": float(np.mean(res))})
    return pd.DataFrame(out)


def build_one_is_run(df: pd.DataFrame, pid: str) -> bool:
    d = df[df["play_id"] == pid]
    t, _, speed = com_speed(d)
    xy = d.sort_values("t_sec")[["MidHip_x", "MidHip_y"]].to_numpy()
    return segment_run(t, xy, speed)["is_run"]


def savgol_cross_check(play_df: pd.DataFrame) -> float:
    """Peak 1-s speed via Savitzky-Golay smoothing instead of Butterworth -- a filter-robustness
    check. Should land within a few tenths of a ft/s of com_speed's F1."""
    d = play_df.sort_values("t_sec")
    t = d["t_sec"].to_numpy()
    win = max(5, int(round(0.4 * _sampling_rate(t))) | 1)   # ~0.4 s, odd
    pos = np.column_stack([savgol_filter(d["MidHip_x"].to_numpy(), win, 3),
                           savgol_filter(d["MidHip_y"].to_numpy(), win, 3)])
    return peak_1s_speed(t, pos)


if __name__ == "__main__":
    df = ld.load_reshaped()
    feats = build_feature_table(df)

    print("=== run_features.py - per-play feature table ===\n")
    n_run = int(feats["is_run"].sum())
    hip = feats["pitch_result"].str.startswith("hit_into_play")
    print(f"plays: {len(feats)}   is_run=True: {n_run}   (29 hit_into_play* plays)")
    print(f"is_run vs hit_into_play* agreement: {int((feats['is_run'] == hip).sum())}/50")
    miss = feats[hip & ~feats["is_run"]]
    for _, r in miss.iterrows():
        print(f"  excluded as no-captured-run: {r['play_id']} {r['pitch_result']} "
              f"(CoM travel {r['max_disp_ft']:.0f} ft, {r['run_dur_s']:.1f} s) -> likely truncated clip")

    runs = feats[feats["is_run"]]
    print("\n--- F1 peak 1-s sprint speed (ft/s), runs only ---")
    print(f"mean {runs['F1_peak1s_ftps'].mean():.2f}  median {runs['F1_peak1s_ftps'].median():.2f}  "
          f"max {runs['F1_peak1s_ftps'].max():.2f}  (brief: mean ~24.2, max ~28.9)")
    fast = runs.loc[runs["F1_peak1s_ftps"].idxmax()]
    slow = feats.loc[feats["F1_peak1s_ftps"].idxmin()]
    print(f"fastest run : {fast['play_id']} {fast['pitch_result']}  F1={fast['F1_peak1s_ftps']:.2f} ft/s")
    print(f"slowest play: {slow['play_id']} {slow['pitch_result']}  F1={slow['F1_peak1s_ftps']:.2f} ft/s "
          f"(no-run swing-lunge)")
    print(f"ceiling check: max F1 {feats['F1_peak1s_ftps'].max():.1f} ft/s < 41 ft/s human limit: "
          f"{feats['F1_peak1s_ftps'].max() < 41}")
    print(f"Savitzky-Golay cross-check on fastest run: F1={savgol_cross_check(df[df['play_id']==fast['play_id']]):.2f} "
          f"ft/s (Butterworth {fast['F1_peak1s_ftps']:.2f})")
    print(f"slide_flag count: {int(feats['slide_flag'].sum())}")

    print("\n--- residual analysis (RMS raw-filtered MidHip, ft) ---")
    print(residual_analysis(df).to_string(index=False))

    print("\n--- effort-axis features, runs only (describe) ---")
    cols = ["F1_peak1s_ftps", "F2_burst_ft", "F3_straightness", "F5_sustain_frac", "F7_brake_decel_ftps2"]
    with pd.option_context("display.width", 200):
        print(runs[cols].describe().round(2).to_string())

    feats.to_csv("run_features.csv", index=False)
    print(f"\nwrote run_features.csv  ({len(feats)} plays x {feats.shape[1]} cols)")
