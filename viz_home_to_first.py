"""Methodology figure: home-to-first CoM paths vs the trusted first base.

Validates that "first base = 90 ft from the origin at 45 deg" (the corner of a 90-ft
square, trusting origin = home plate) is consistent with where the player's center of mass
(MidHip) actually travels. For every home-to-first run we overlay the CoM path from the box
to the bag and mark the closest approach to the trusted 1B, so the accuracy is visible.

Run: python viz_home_to_first.py   ->   figures/home_to_first_accuracy.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle

import load_data as ld

# --- Trusted diamond geometry (90-ft square, home at origin, 1B at 45 deg) ---
SIDE = 90.0
HOME = np.array([0.0, 0.0])
B1 = SIDE * np.array([np.cos(np.radians(45)), np.sin(np.radians(45))])   # (63.6, 63.6)
B2 = np.array([0.0, SIDE * np.sqrt(2)])                                  # (0, 127.3)
B3 = SIDE * np.array([np.cos(np.radians(135)), np.sin(np.radians(135))])  # (-63.6, 63.6)

LEAVE_BOX_FT = 3.0      # path starts once CoM is this far from the stance (skips the swing)
REACH_FT = 6.0          # "reached a base" if CoM passes within this of the bag
STANCE_NEAR_HOME = 15.0  # require a captured start near home to draw a home->first path

# safe single (no_out) vs out-at-first -- both are valid home->first runs for this check
COLORS = {"hit_into_play_no_out": "#1f77b4", "hit_into_play": "#ff7f0e"}
LABELS = {"hit_into_play_no_out": "safe single (no_out)", "hit_into_play": "out at first"}


def home_to_first_runs(df):
    """Return [(play_id, pitch_result, path_xy, arrival_xy, arrival_err)] for runs whose
    destination was first base, with a captured start near home."""
    out = []
    for pid in sorted(df.play_id.unique()):
        d = df[df.play_id == pid].reset_index(drop=True)
        if not d.pitch_result.iloc[0].startswith("hit_into_play"):
            continue
        xy = d[["MidHip_x", "MidHip_y"]].to_numpy()
        d1 = np.linalg.norm(xy - B1, axis=1)
        d2 = np.linalg.norm(xy - B2, axis=1)
        d3 = np.linalg.norm(xy - B3, axis=1)
        # reached 1B, did not continue to 2B/3B, and the stance was captured near home
        if d1.min() > REACH_FT or d2.min() <= REACH_FT or d3.min() <= REACH_FT:
            continue
        if np.linalg.norm(xy[0] - HOME) > STANCE_NEAR_HOME:
            continue
        disp0 = np.linalg.norm(xy - xy[0], axis=1)
        start = int(np.argmax(disp0 > LEAVE_BOX_FT))   # first step out of the box
        end = int(np.argmin(d1))                        # closest approach to 1B (arrival)
        out.append((pid, d.pitch_result.iloc[0], xy[start:end + 1], xy[end], float(d1[end])))
    return out


def main():
    df = ld.load_reshaped()
    runs = home_to_first_runs(df)
    errs = np.array([r[4] for r in runs])
    n_safe = sum(1 for r in runs if r[1] == "hit_into_play_no_out")

    fig, ax = plt.subplots(figsize=(9, 9))

    # reference geometry: home->1B foul line and the home->1B leg of the diamond
    ax.plot([HOME[0], B1[0]], [HOME[1], B1[1]], "--", color="0.6", lw=1.2,
            zorder=1, label="home->1B reference (90 ft, 45 deg)")
    ax.plot([B1[0], B2[0]], [B1[1], B2[1]], ":", color="0.8", lw=1, zorder=1)  # hint of 1B->2B

    # CoM paths, colored by safe/out
    seen = set()
    for pid, pr, path, arr, err in runs:
        lbl = LABELS[pr] if pr not in seen else None
        seen.add(pr)
        ax.plot(path[:, 0], path[:, 1], "-", color=COLORS[pr], lw=1.2, alpha=0.55,
                zorder=2, label=lbl)
        ax.plot(*arr, "o", color=COLORS[pr], ms=4, alpha=0.9, zorder=3)

    # markers: home plate (pentagon) and the trusted 1B bag (rotated square)
    ax.add_patch(RegularPolygon(HOME, 5, radius=2.2, orientation=np.pi,
                                facecolor="white", edgecolor="black", lw=1.5, zorder=4))
    ax.plot(*B1, "s", color="crimson", ms=12, mec="black", mew=1.2, zorder=5,
            label="trusted 1B (90 ft, 45 deg)")
    ax.annotate("home plate\n(origin)", HOME, textcoords="offset points", xytext=(8, -22),
                fontsize=9)
    ax.annotate("trusted 1B\n(63.6, 63.6)", B1, textcoords="offset points", xytext=(10, -2),
                fontsize=9, color="crimson")

    # stats box
    txt = (f"home-to-first runs: n = {len(runs)}  ({n_safe} safe singles, "
           f"{len(runs) - n_safe} outs at first)\n"
           f"CoM closest approach to trusted 1B:\n"
           f"  mean {errs.mean():.1f} ft   median {np.median(errs):.1f} ft   "
           f"max {errs.max():.1f} ft\n"
           f"(paths funnel through the bag to ~1.5 ft; residual =\n"
           f" CoM-over-bag vs bag-corner + keypoint noise)")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95))

    ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)")
    ax.set_title("Home-to-first CoM (MidHip) paths vs the trusted first base\n"
                 "Phillies HP QA take-home -- coordinate-system validation", fontsize=12)
    ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(-8, 74); ax.set_ylim(-8, 74)
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)

    # zoom inset around 1B to show the arrival cluster vs the trusted bag
    axins = ax.inset_axes([0.62, 0.08, 0.34, 0.34])
    for pid, pr, path, arr, err in runs:
        axins.plot(path[:, 0], path[:, 1], "-", color=COLORS[pr], lw=1, alpha=0.5)
        axins.plot(*arr, "o", color=COLORS[pr], ms=5, alpha=0.9)
    axins.plot(*B1, "s", color="crimson", ms=12, mec="black", mew=1.2)
    axins.add_patch(Circle(B1, errs.mean(), fill=False, ec="crimson", ls="--", lw=1))
    axins.set_xlim(B1[0] - 7, B1[0] + 7); axins.set_ylim(B1[1] - 7, B1[1] + 7)
    axins.set_aspect("equal"); axins.set_title("zoom: arrival vs 1B", fontsize=8)
    axins.tick_params(labelsize=7)
    ax.indicate_inset_zoom(axins, edgecolor="0.5")

    os.makedirs("figures", exist_ok=True)
    out = "figures/home_to_first_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"n={len(runs)} home-to-first runs ({n_safe} safe singles); "
          f"closest approach to trusted 1B: mean {errs.mean():.2f}, "
          f"median {np.median(errs):.2f}, max {errs.max():.2f} ft")


if __name__ == "__main__":
    main()
