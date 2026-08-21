"""Step 2b: effort / run-type clustering of the batter's captured runs.

Clusters the real runs (run_features.is_run) by their movement profile to separate
max-effort sprints from leg-outs and trots, and emits a continuous effort score [0,1] that
downstream steps (Sprint Speed, workload, readiness) use to weight/qualify each run.

Design notes (see the plan / PROJECT_BRIEF.md):
  - Cluster on EFFORT-SHAPE axes only (F1 peak speed, F2 burst, F3 straightness, F5 sustain,
    F7 braking). Outcome proxies (path length, base reached, duration) are deliberately
    excluded -- including them would re-cluster by which base he reached, and then validating
    against pitch_result would be circular.
  - N is small (~28). Tiers are a defensible stratification of THIS player's runs, not latent
    classes. NEGATIVE-RESULT PROTOCOL: if cross-method agreement (ARI) is low AND leave-one-out
    stability is poor, we report the continuous effort score only, not discrete tiers.

Run: python cluster_runs.py  ->  run_clusters.csv + figures/cluster_*.png
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

import load_data as ld
import run_features as rf

# Effort-shape feature vector (NOT outcome proxies). F2/F7 are right-skewed -> log first.
CLUSTER_FEATURES = ["F1_peak1s_ftps", "F2_burst_ft", "F3_straightness",
                    "F5_sustain_frac", "F7_brake_decel_ftps2"]
LOG_FEATURES = ["F2_burst_ft", "F7_brake_decel_ftps2"]
K_RANGE = (2, 3, 4)
RANDOM_STATE = 0   # fixed (Math.random/np seed) for reproducibility
TIER_NAMES = {2: ["sub-max", "max-effort"],
              3: ["trot", "leg-out", "max-effort"],
              4: ["trot", "easy", "leg-out", "max-effort"]}


# --------------------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------------------
def build_matrix(feats: pd.DataFrame):
    """Return (runs_df, X_scaled, scaler) for the captured runs only."""
    runs = feats[feats["is_run"]].dropna(subset=CLUSTER_FEATURES).reset_index(drop=True)
    raw = runs[CLUSTER_FEATURES].copy()
    for c in LOG_FEATURES:
        raw[c] = np.log(raw[c])
    scaler = RobustScaler()                 # median/IQR -- robust to the F7 heavy tail
    X = scaler.fit_transform(raw.to_numpy())
    return runs, X, scaler


# --------------------------------------------------------------------------------------
# Clustering methods (all return integer labels 0..k-1)
# --------------------------------------------------------------------------------------
def ward_labels(X, k):
    return fcluster(linkage(X, method="ward"), t=k, criterion="maxclust") - 1


def kmeans_labels(X, k):
    return KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit_predict(X)


def gmm_fit(X, k):
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=10,
                           random_state=RANDOM_STATE).fit(X)


# --------------------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------------------
def select_k(X):
    """Triangulate k via silhouette (Ward + KMeans) and GMM BIC. Return (k_star, table)."""
    rows = []
    for k in K_RANGE:
        wl, kl = ward_labels(X, k), kmeans_labels(X, k)
        rows.append(dict(
            k=k,
            silhouette_ward=silhouette_score(X, wl),
            silhouette_kmeans=silhouette_score(X, kl),
            gmm_bic=gmm_fit(X, k).bic(X),
        ))
    tbl = pd.DataFrame(rows)
    tbl["silhouette_mean"] = tbl[["silhouette_ward", "silhouette_kmeans"]].mean(axis=1)
    # primary: best mean silhouette; BIC reported alongside (lower = better) as a tie-breaker
    k_star = int(tbl.loc[tbl["silhouette_mean"].idxmax(), "k"])
    return k_star, tbl


# --------------------------------------------------------------------------------------
# Effort score + tier ordering
# --------------------------------------------------------------------------------------
def order_tiers_by_speed(labels, f1):
    """Relabel clusters 0..k-1 in ascending mean F1 so the top label = max effort."""
    order = pd.Series(f1).groupby(labels).mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])


def effort_scores(X, gmm, ordered_ref_labels, f1):
    """Two continuous effort scores in [0,1]:
       - GMM posterior of the max-effort component (component with highest mean F1);
       - normalized inverse distance to the max-effort centroid (no-GMM reproducible analogue).
    """
    post = gmm.predict_proba(X)
    comp_f1 = pd.Series(f1).groupby(gmm.predict(X)).mean()
    max_comp = int(comp_f1.idxmax())
    score_gmm = post[:, max_comp]

    max_centroid = X[ordered_ref_labels == ordered_ref_labels.max()].mean(axis=0)
    d = np.linalg.norm(X - max_centroid, axis=1)
    score_dist = 1.0 - (d - d.min()) / (d.max() - d.min())
    return score_gmm, score_dist


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def loo_stability(X, k):
    """Leave-one-out co-assignment stability for Ward at k. Returns the mean agreement between
    the full-data co-clustering and each LOO co-clustering, over all evaluable pairs."""
    n = len(X)
    full = ward_labels(X, k)
    co_full = full[:, None] == full[None, :]
    agree, total = 0, 0
    for i in range(n):
        keep = np.arange(n) != i
        sub = ward_labels(X[keep], k)
        co_sub = sub[:, None] == sub[None, :]
        co_full_sub = co_full[np.ix_(keep, keep)]
        iu = np.triu_indices(n - 1, k=1)
        agree += int((co_sub[iu] == co_full_sub[iu]).sum())
        total += len(iu[0])
    return agree / total


def cross_method_ari(X, k):
    wl, kl, gl = ward_labels(X, k), kmeans_labels(X, k), gmm_fit(X, k).predict(X)
    return {
        "ward_kmeans": adjusted_rand_score(wl, kl),
        "ward_gmm": adjusted_rand_score(wl, gl),
        "kmeans_gmm": adjusted_rand_score(kl, gl),
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_dendrogram(X, runs, k):
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"{p.split('_')[1]} {pr[:14]}" for p, pr in zip(runs["play_id"], runs["pitch_result"])]
    dendrogram(linkage(X, method="ward"), labels=labels, leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_title(f"Ward dendrogram of the {len(runs)} captured runs (leaves = play + outcome)\n"
                 f"effort-shape features; cut at k={k}", fontsize=11)
    ax.set_ylabel("Ward linkage distance")
    fig.savefig("figures/cluster_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(runs, tier, k):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(tier / max(tier.max(), 1))
    for ax, (xc, yc) in zip(axes, [("F1_peak1s_ftps", "F2_burst_ft"),
                                   ("F1_peak1s_ftps", "F5_sustain_frac")]):
        ax.scatter(runs[xc], runs[yc], c=colors, s=70, edgecolor="0.3")
        for _, r in runs.iterrows():
            ax.annotate(r["play_id"].split("_")[1], (r[xc], r[yc]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(xc); ax.set_ylabel(yc); ax.grid(True, ls=":", alpha=0.5)
    axes[0].set_title("Effort tiers (green = max effort)")
    fig.suptitle(f"Run-effort clusters in feature space (k={k})", fontsize=12)
    fig.savefig("figures/cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_diamond(df, runs, tier, k):
    """Cluster-colored CoM paths on the trusted diamond (reuses run_features geometry)."""
    fig, ax = plt.subplots(figsize=(9, 9))
    diamond = np.array([rf.HOME, rf.B1, rf.B2, rf.B3, rf.HOME])
    ax.plot(diamond[:, 0], diamond[:, 1], "-", color="0.8", lw=1.2, zorder=1)
    for pt, nm in [(rf.B1, "1B"), (rf.B2, "2B"), (rf.B3, "3B")]:
        ax.plot(*pt, "s", color="black", ms=8, mfc="white", zorder=6)
        ax.annotate(nm, pt, textcoords="offset points", xytext=(6, 6), fontsize=9)
    cmap = plt.get_cmap("RdYlGn")
    for (_, r), tr in zip(runs.iterrows(), tier):
        d = df[df["play_id"] == r["play_id"]]
        t, pos, speed = rf.com_speed(d)
        seg = rf.segment_run(t, d.sort_values("t_sec")[["MidHip_x", "MidHip_y"]].to_numpy(), speed)
        path = pos[seg["start"]:seg["end"] + 1]
        ax.plot(path[:, 0], path[:, 1], "-", color=cmap(tr / max(tier.max(), 1)),
                lw=1.6, alpha=0.8, zorder=3)
    ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlabel("x (ft)"); ax.set_ylabel("y (ft)")
    ax.set_title(f"Effort-clustered CoM paths on the diamond (k={k}, green = max effort)", fontsize=11)
    fig.savefig("figures/cluster_diamond.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_speed_curves(df, runs, tier, k):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("RdYlGn")
    for t_label in sorted(set(tier)):
        # one representative run per tier: the median-F1 member
        members = runs[tier == t_label]
        rep = members.iloc[(members["F1_peak1s_ftps"] - members["F1_peak1s_ftps"].median()).abs().argmin()]
        d = df[df["play_id"] == rep["play_id"]]
        t, pos, speed = rf.com_speed(d)
        seg = rf.segment_run(t, d.sort_values("t_sec")[["MidHip_x", "MidHip_y"]].to_numpy(), speed)
        s, e = seg["start"], seg["end"]
        ax.plot(t[s:e + 1] - t[s], speed[s:e + 1], color=cmap(t_label / max(tier.max(), 1)),
                lw=2, label=f"{TIER_NAMES[k][t_label]} (e.g. {rep['play_id']}, F1={rep['F1_peak1s_ftps']:.1f})")
    ax.set_xlabel("time since first step (s)"); ax.set_ylabel("CoM speed (ft/s)")
    ax.axhline(27, ls=":", color="0.5", lw=1); ax.text(0.1, 27.3, "MLB avg 27 ft/s", fontsize=8, color="0.4")
    ax.set_title(f"Representative speed-vs-time per effort tier (k={k})", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    fig.savefig("figures/cluster_speed_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_residual(df):
    res = rf.residual_analysis(df)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res["cutoff_hz"], res["rms_residual_ft"], "o-")
    ax.axvline(rf.BUTTER_CUTOFF_HZ, ls="--", color="crimson",
               label=f"chosen fc = {rf.BUTTER_CUTOFF_HZ:.0f} Hz")
    ax.set_xlabel("Butterworth cutoff (Hz)"); ax.set_ylabel("RMS(raw - filtered) MidHip (ft)")
    ax.set_title("Residual analysis: filter-cutoff selection", fontsize=11)
    ax.legend(); ax.grid(True, ls=":", alpha=0.5)
    fig.savefig("figures/cluster_residual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    os.makedirs("figures", exist_ok=True)
    df = ld.load_reshaped()
    feats = rf.build_feature_table(df)
    runs, X, _ = build_matrix(feats)
    print(f"=== cluster_runs.py - effort clustering of {len(runs)} captured runs ===\n")

    k, ktbl = select_k(X)
    print("--- model selection (k by mean silhouette; GMM BIC as tie-breaker) ---")
    print(ktbl.round(3).to_string(index=False))
    print(f"chosen k = {k}\n")

    f1 = runs["F1_peak1s_ftps"].to_numpy()
    wl = order_tiers_by_speed(ward_labels(X, k), f1)          # headline method = Ward
    kl = order_tiers_by_speed(kmeans_labels(X, k), f1)
    gmm = gmm_fit(X, k)
    gl = order_tiers_by_speed(gmm.predict(X), f1)
    runs["tier"] = wl
    runs["tier_name"] = [TIER_NAMES[k][t] for t in wl]
    runs["tier_kmeans"], runs["tier_gmm"] = kl, gl

    score_gmm, score_dist = effort_scores(X, gmm, wl, f1)
    runs["effort_score"] = score_gmm
    runs["effort_score_dist"] = score_dist

    print("--- tier sizes + mean features (Ward, ordered by F1) ---")
    summ = runs.groupby("tier_name").agg(
        n=("play_id", "size"), F1=("F1_peak1s_ftps", "mean"), F2=("F2_burst_ft", "mean"),
        F3=("F3_straightness", "mean"), F5=("F5_sustain_frac", "mean"),
        F7=("F7_brake_decel_ftps2", "mean"), effort=("effort_score", "mean"))
    print(summ.round(2).to_string())

    print("\n--- external concordance: tier x pitch_result ---")
    print(pd.crosstab(runs["tier_name"], runs["pitch_result"]).to_string())

    ari = cross_method_ari(X, k)
    loo = loo_stability(X, k)
    print(f"\n--- stability ---")
    print(f"cross-method ARI: " + "  ".join(f"{kk}={vv:.2f}" for kk, vv in ari.items())
          + f"   mean={np.mean(list(ari.values())):.2f}")
    print(f"leave-one-out Ward co-assignment stability: {loo:.1%}")
    print(f"silhouette (Ward, k={k}): {silhouette_score(X, wl):.3f}")
    print(f"effort_score corr (GMM vs distance): {np.corrcoef(score_gmm, score_dist)[0,1]:.3f}")

    mean_ari = float(np.mean(list(ari.values())))
    if mean_ari < 0.5 or loo < 0.80:
        print("\n[NEGATIVE-RESULT PROTOCOL] cross-method ARI and/or LOO stability are weak ->\n"
              "  discrete tiers are NOT robust at this N; report the CONTINUOUS effort_score\n"
              "  (a gradient), not hard tiers, as the primary downstream signal.")
    else:
        print(f"\n[OK] tiers are stable (mean ARI {mean_ari:.2f}, LOO {loo:.0%}); both the discrete\n"
              "  tier and the continuous effort_score are usable downstream.")

    fig_residual(df); fig_dendrogram(X, runs, k); fig_scatter(runs, wl, k)
    fig_diamond(df, runs, wl, k); fig_speed_curves(df, runs, wl, k)

    out_cols = ["play_id", "pitch_result", "game_date", "tier", "tier_name", "effort_score",
                "effort_score_dist", "tier_kmeans", "tier_gmm"] + CLUSTER_FEATURES + \
               ["base_reached", "max_disp_ft", "z_drop_ft", "slide_flag"]
    runs[out_cols].to_csv("run_clusters.csv", index=False)
    print(f"\nwrote run_clusters.csv ({len(runs)} runs) and figures/cluster_*.png")


if __name__ == "__main__":
    main()
