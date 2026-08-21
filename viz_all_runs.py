"""Methodology figure(s): all 29 run plays on the trusted 90-ft diamond.

Every hit_into_play* run's CoM (MidHip) path is overlaid on the trusted diamond (home at
origin, bases at the 90-ft-square corners). Outs (hit_into_play) are grouped into one light
color (less significant); safe hits are colored by how far the runner got. This validates
the coordinate system at all four bases: paths to 1B/2B/3B/home converge on the trusted bags.

Solid = run to the base reached; dashed = the next ~1.5 s (overrun / deceleration after the
base). Emits two views from the identical plot: the full diamond and a home-to-first zoom.

Run: python viz_all_runs.py
  -> figures/all_runs_diamond.png
  -> figures/all_runs_home_to_first_zoom.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D

import load_data as ld

# Trusted diamond: 90-ft square, home at origin, 1B at 45 deg
SIDE = 90.0
HOME = np.array([0.0, 0.0])
B1 = SIDE * np.array([np.cos(np.radians(45)), np.sin(np.radians(45))])    # (63.6, 63.6)
B2 = np.array([0.0, SIDE * np.sqrt(2)])                                   # (0, 127.3)
B3 = SIDE * np.array([np.cos(np.radians(135)), np.sin(np.radians(135))])  # (-63.6, 63.6)

LEAVE_BOX_FT = 3.0     # path starts once CoM is this far from the stance (skips the swing)
REACH_FT = 6.0         # "reached a base" if CoM passes within this of the bag
HOME_RETURN_FT = 12.0  # ended this close to home => completed the circuit (scored)
AFTER_SEC = 1.5        # how much running to show after reaching the base (overrun)

# color class -> (color, legend label). Outs are grouped + de-emphasized; safe hits by base.
COLOR = {
    "scored": ("crimson",    "scored / inside-the-park"),
    "triple": ("tab:purple", "triple (safe, 3B)"),
    "double": ("tab:green",  "double (safe, 2B)"),
    "single": ("tab:blue",   "single (safe, 1B)"),
    "out":    ("0.6",        "out (hit_into_play)"),
}
LEGEND_ORDER = ["scored", "triple", "double", "single", "out"]


def extract_runs(df):
    """For every run play return its color class, path to the base reached, the overrun path
    (~AFTER_SEC after), arrival point, and closest-approach distance to each base."""
    runs = []
    for pid in sorted(df.play_id.unique()):
        d = df[df.play_id == pid].reset_index(drop=True)
        pr = d.pitch_result.iloc[0]
        if not pr.startswith("hit_into_play"):
            continue
        xy = d[["MidHip_x", "MidHip_y"]].to_numpy()
        t = d["t_sec"].to_numpy()
        d1 = np.linalg.norm(xy - B1, axis=1)
        d2 = np.linalg.norm(xy - B2, axis=1)
        d3 = np.linalg.norm(xy - B3, axis=1)
        r1, r2, r3 = d1.min() <= REACH_FT, d2.min() <= REACH_FT, d3.min() <= REACH_FT
        ended_home = np.linalg.norm(xy[-1] - HOME) <= HOME_RETURN_FT

        if r3 and ended_home:
            geo, end = "scored", len(xy) - 1
        elif r3:
            geo, end = "triple", int(np.argmin(d3))
        elif r2:
            geo, end = "double", int(np.argmin(d2))
        elif r1:
            geo, end = "single", int(np.argmin(d1))
        else:
            geo, end = "noreach", int(np.argmax(np.linalg.norm(xy - xy[0], axis=1)))

        cls = "out" if pr == "hit_into_play" else geo   # outs grouped; safe hits by distance

        disp0 = np.linalg.norm(xy - xy[0], axis=1)
        start = int(np.argmax(disp0 > LEAVE_BOX_FT)) if (disp0 > LEAVE_BOX_FT).any() else 0
        after = int(np.searchsorted(t, t[end] + AFTER_SEC, side="right"))
        runs.append(dict(pid=pid, cls=cls, to_base=xy[start:end + 1], over=xy[end:after],
                         arrival=xy[end], d1=d1.min(), d2=d2.min(), d3=d3.min(),
                         r1=r1, r2=r2, r3=r3))
    return runs


def render(runs, counts, stats_txt, out_path, subtitle, xlim, ylim, stats_loc):
    """Draw the diamond + all run paths. xlim/ylim None => autoscale to the data."""
    fig, ax = plt.subplots(figsize=(9.5, 9.5))

    diamond = np.array([HOME, B1, B2, B3, HOME])
    ax.plot(diamond[:, 0], diamond[:, 1], "-", color="0.8", lw=1.2, zorder=1)
    for pt, name in [(B1, "1B"), (B2, "2B"), (B3, "3B")]:
        ax.plot(*pt, "s", color="black", ms=9, mfc="white", mew=1.4, zorder=6)
        ax.annotate(f"{name}\n({pt[0]:.0f}, {pt[1]:.0f})", pt, textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.add_patch(RegularPolygon(HOME, 5, radius=2.6, orientation=np.pi,
                                facecolor="white", edgecolor="black", lw=1.5, zorder=6))
    ax.annotate("home (origin)", HOME, textcoords="offset points", xytext=(8, -20), fontsize=9)

    for r in sorted(runs, key=lambda r: r["cls"] != "out"):   # outs first (background)
        color = COLOR[r["cls"]][0]
        is_out = r["cls"] == "out"
        a, lw, z = (0.4, 1.0, 2) if is_out else (0.75, 1.5, 4)
        ax.plot(r["to_base"][:, 0], r["to_base"][:, 1], "-", color=color, lw=lw, alpha=a, zorder=z)
        if len(r["over"]) > 1:
            ax.plot(r["over"][:, 0], r["over"][:, 1], "--", color=color, lw=lw * 0.8,
                    alpha=a * 0.8, zorder=z - 1)
            ax.plot(*r["over"][-1], "x", color=color, ms=4, alpha=a, zorder=z)
        ax.plot(*r["arrival"], "o", color=color, ms=4, alpha=min(a + 0.2, 1.0), zorder=z + 1)

    sx, sy, sha, sva = stats_loc
    ax.text(sx, sy, stats_txt, transform=ax.transAxes, va=sva, ha=sha, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95))

    ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)")
    ax.set_title(f"All 29 runs on the trusted 90-ft diamond (CoM / MidHip paths, with overrun)\n"
                 f"{subtitle}", fontsize=12)
    ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.5)

    if xlim is None:
        pts = np.vstack([r["to_base"] for r in runs] + [r["over"] for r in runs if len(r["over"])]
                        + [B1[None], B2[None], B3[None], HOME[None]])
        ax.set_xlim(pts[:, 0].min() - 6, pts[:, 0].max() + 6)
        ax.set_ylim(pts[:, 1].min() - 6, pts[:, 1].max() + 8)
    else:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    handles = [Line2D([0], [0], color=COLOR[c][0], lw=2.5) for c in LEGEND_ORDER if counts[c]]
    labels = [f"{COLOR[c][1]}  (n={counts[c]})" for c in LEGEND_ORDER if counts[c]]
    handles.append(Line2D([0], [0], color="0.4", ls="--", lw=1.2))
    labels.append(f"overrun (~{AFTER_SEC:.1f} s after base)")
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, framealpha=0.95, title="run outcome / reached")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    df = ld.load_reshaped()
    runs = extract_runs(df)
    counts = {c: sum(1 for r in runs if r["cls"] == c) for c in COLOR}

    def conv(key, rk):
        vals = [r[key] for r in runs if r[rk]]
        return len(vals), float(np.mean(vals)), float(np.max(vals))
    n1, m1, x1 = conv("d1", "r1"); n2, m2, x2 = conv("d2", "r2"); n3, m3, x3 = conv("d3", "r3")
    stats_txt = ("CoM closest approach to each trusted bag (mean / max):\n"
                 f"  1B  n={n1}:  {m1:.1f} / {x1:.1f} ft\n"
                 f"  2B  n={n2}:  {m2:.1f} / {x2:.1f} ft\n"
                 f"  3B  n={n3}:  {m3:.1f} / {x3:.1f} ft\n"
                 f"solid = run to base;  dashed = next {AFTER_SEC:.1f} s (overrun), x = stop")

    os.makedirs("figures", exist_ok=True)
    # full diamond (autoscale), stats box in the empty interior
    render(runs, counts, stats_txt, "figures/all_runs_diamond.png",
           "outs grouped (light gray); safe hits colored by base reached",
           xlim=None, ylim=None, stats_loc=(0.17, 0.50, "left", "center"))
    # home-to-first zoom (same plot, tighter window), stats box in the empty lower-right
    render(runs, counts, stats_txt, "figures/all_runs_home_to_first_zoom.png",
           "zoomed to home -> first base (all 29 runs, same colors)",
           xlim=(-6, 92), ylim=(-6, 86), stats_loc=(0.97, 0.03, "right", "bottom"))

    print("class counts:", counts)
    print(f"convergence  1B {m1:.2f}/{x1:.2f}  2B {m2:.2f}/{x2:.2f}  3B {m3:.2f}/{x3:.2f} ft (mean/max)")


if __name__ == "__main__":
    main()
