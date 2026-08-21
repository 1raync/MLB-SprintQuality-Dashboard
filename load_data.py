"""Step 1 of the HP QA solution: load and reshape the batter joint-position data.

The parquet ships in long format -- one row per (play, frame, joint) with the x/y/z
coordinates buried in a nested `joint` struct. Almost every downstream metric needs either
an across-time view (a joint's position over consecutive frames -> velocity, sprint speed)
or an across-joint view (several joints at one instant -> bone length, center of mass, gait
events). Neither is workable in long format, so we flatten the struct and pivot to one row
per (play, frame) with named landmark columns, sorted by time within each play.

Later steps import `load_reshaped()`. Run this file directly for a verification report.

Units are FEET (x/y = field plane, z = vertical height). See PROJECT_BRIEF.md sec 2.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DEFAULT_PARQUET = "batter_running_workload.parquet"
DEFAULT_WIDE_PATH = "batter_running_wide.parquet"

# Canonical id -> name map for all 29 landmarks, verified against the data.
# ids 0-24 are the OpenPose BODY_25 keypoint set (exact ordering); ids 25-28 are the four
# hand landmarks (thumbs/pinkies) appended on top. Used to assert the format hasn't drifted.
BODY_25_JOINTS: dict[int, str] = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel",
    25: "LThumb", 26: "LPinky", 27: "RThumb", 28: "RPinky",
}
N_JOINTS = len(BODY_25_JOINTS)  # 29

KEY_COLS = ["play_id", "pitch_result", "frame_timestamp", "frame_idx", "t_sec"]
AXES = ("x", "y", "z")


def _check_integrity(long_df: pd.DataFrame) -> None:
    """Raise AssertionError if the long-format data doesn't match the expected schema.

    These guards belong here, at the one place the raw data enters the pipeline: a bad frame
    or a format change should fail loudly now, not surface as a plausible-but-wrong velocity
    three steps downstream.
    """
    # 1. id -> name mapping matches the canonical BODY_25(+hands) set exactly.
    found = (
        long_df[["id", "name"]]
        .drop_duplicates()
        .set_index("id")["name"]
        .to_dict()
    )
    assert found == BODY_25_JOINTS, (
        "joint id->name map does not match the expected BODY_25(+hands) set; "
        f"got {sorted(found.items())}"
    )

    # 2. Exactly N_JOINTS joints per (play, frame): no missing landmarks, no duplicates.
    per_frame = long_df.groupby(["play_id", "frame_timestamp"], sort=False).size()
    bad = per_frame[per_frame != N_JOINTS]
    assert bad.empty, f"{len(bad)} frame(s) do not have exactly {N_JOINTS} joints"

    # 3. No missing coordinates.
    n_nan = int(long_df[["x", "y", "z"]].isna().sum().sum())
    assert n_nan == 0, f"found {n_nan} NaN coordinate value(s)"

    # 4. Row-count round trip: total rows == frames * joints (no stray/dropped rows).
    n_frames = per_frame.shape[0]
    assert len(long_df) == n_frames * N_JOINTS, (
        f"row count {len(long_df)} != frames {n_frames} * joints {N_JOINTS}"
    )


def load_reshaped(path: str = DEFAULT_PARQUET, *, validate: bool = True) -> pd.DataFrame:
    """Load the parquet and return wide format: one row per (play, frame).

    Columns: play_id, pitch_result, frame_timestamp, frame_idx, t_sec, then 87 coordinate
    columns named `{Joint}_{x|y|z}` (e.g. ``MidHip_x``). Rows are sorted by
    (play_id, frame_timestamp) so time is monotonic within each play -- a precondition for
    any differencing downstream.
    """
    raw = pd.read_parquet(path)

    # Flatten the nested `joint` struct into its own columns and re-attach the keys.
    joints = pd.json_normalize(raw["joint"])  # -> id, name, x, y, z
    long_df = pd.concat(
        [raw[["play_id", "pitch_result", "frame_timestamp"]].reset_index(drop=True),
         joints.reset_index(drop=True)],
        axis=1,
    )

    if validate:
        _check_integrity(long_df)

    # Pivot to wide: one row per (play, frame), a column per joint x axis.
    wide = long_df.pivot(
        index=["play_id", "frame_timestamp"],
        columns="name",
        values=list(AXES),
    )
    # Flatten the ("x", "MidHip") MultiIndex columns to "MidHip_x".
    wide.columns = [f"{joint}_{axis}" for axis, joint in wide.columns]

    # Order the coordinate columns by the canonical joint order, axis-major per joint.
    coord_cols = [f"{name}_{axis}" for name in BODY_25_JOINTS.values() for axis in AXES]
    wide = wide[coord_cols].reset_index()

    # Re-attach pitch_result (constant within a play) and sort by time within each play.
    pr = long_df.groupby("play_id", sort=False)["pitch_result"].first()
    wide["pitch_result"] = wide["play_id"].map(pr)
    wide = wide.sort_values(["play_id", "frame_timestamp"]).reset_index(drop=True)

    # Time-axis helpers downstream steps need. dt comes from timestamps (the source of truth),
    # not an assumed 30 fps -- the nominal rate has small jitter (see PROJECT_BRIEF.md sec 2.5).
    grp = wide.groupby("play_id", sort=False)
    wide["frame_idx"] = grp.cumcount()
    t0 = grp["frame_timestamp"].transform("first")
    wide["t_sec"] = (wide["frame_timestamp"] - t0).dt.total_seconds()

    return wide[KEY_COLS + coord_cols]


def save_wide(df: pd.DataFrame, path: str = DEFAULT_WIDE_PATH) -> None:
    """Persist the reshaped frame so later steps can skip the pivot. Opt-in (see --save)."""
    df.to_parquet(path, index=False)
    print(f"wrote {path}  ({len(df):,} rows x {df.shape[1]} cols)")


def _joint_xyz(df: pd.DataFrame, joint: str) -> np.ndarray:
    return df[[f"{joint}_x", f"{joint}_y", f"{joint}_z"]].to_numpy()


def _verify_report(df: pd.DataFrame) -> None:
    """Print the human-readable checks described in the plan's verification section."""
    n_coord = len(BODY_25_JOINTS) * len(AXES)
    print("=== load_data.py - verification report ===\n")

    print(f"wide shape:           {df.shape}   (expected (16761, {len(KEY_COLS) + n_coord}))")
    n_frames = len(df)
    print(f"round-trip:           {n_frames} frames x {N_JOINTS} joints = "
          f"{n_frames * N_JOINTS:,}  (raw long rows = 486,069)")
    print(f"coordinate columns:   {n_coord}  (29 joints x 3 axes)")
    print(f"key columns:          {KEY_COLS}")
    print("\n[OK] integrity checks passed (id->name map, 29 joints/frame, no NaN, row count)")

    print("\n--- head: keys + MidHip ---")
    cols = ["play_id", "pitch_result", "frame_idx", "t_sec",
            "MidHip_x", "MidHip_y", "MidHip_z"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df[cols].head(6).to_string(index=False))

    print("\n--- plays per pitch_result (expect 17/15/9/3/3/2/1) ---")
    per_play = df.groupby("play_id", sort=False)["pitch_result"].first()
    print(per_play.value_counts().to_string())
    print(f"total plays: {per_play.shape[0]}")

    # Reshape-correctness proof: a bone length is an across-joint computation. If the pivot
    # is correct, LAnkle and LKnee for the same frame now sit on the same row, so the shank
    # length is a one-row subtraction -- and it should equal the ~1.50 ft from the brief.
    one = df[df["play_id"] == "play_0336"]
    shank = np.linalg.norm(_joint_xyz(one, "LAnkle") - _joint_xyz(one, "LKnee"), axis=1)
    print("\n--- reshape proof: play_0336 left shank (ankle->knee) ---")
    print(f"mean {shank.mean():.3f} ft  std {shank.std():.3f}  "
          f"(brief: ~1.50 ft; ~18 in -> units are feet, across-joint math works on one row)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load & reshape the batter joint data.")
    parser.add_argument("--path", default=DEFAULT_PARQUET, help="input parquet path")
    parser.add_argument("--save", action="store_true",
                        help=f"also write reshaped wide parquet to {DEFAULT_WIDE_PATH}")
    args = parser.parse_args()

    wide = load_reshaped(args.path, validate=True)
    _verify_report(wide)
    if args.save:
        print()
        save_wide(wide)
