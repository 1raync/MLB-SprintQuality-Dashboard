"""Step 3 (interactive): self-contained per-game readiness dashboard (HTML).

Generates a single, offline, clickable HTML file from the live pipeline. Click through games;
each game shows all of its run stats and whether the player is in DECLINE vs. his OWN baseline,
judged from that game's max-effort runs. Replaces the static figures/readiness_trend.png with an
interactive overview -> detail-on-demand view (Shneiderman's mantra).

It does NOT recompute physical output: it embeds readiness.build_run_table() /
build_game_table() outputs verbatim (same pattern as viz_readiness.py) so the HTML, the CSVs, and
the PNG can never disagree. The only NEW per-run math here (reusing run_features.com_speed /
segment_run) is: mean launch acceleration (m/s^2, Vmax-anchored per PROJECT_BRIEF.md sec 3.3),
high-speed-running distance (ft, > 75% Vmax), the run's speed-vs-time curve, and the
single/double/triple/out/scored outcome (reused from viz_all_runs.extract_runs).

Design follows the researched best practices: Okabe-Ito colorblind-safe traffic lights WITH text
labels (never color alone), self-baseline reference lines, tabular numerals, units in every header,
and an explicit "decision support, not an automated benching rule" framing.

Run: python build_dashboard.py  ->  figures/readiness_dashboard.html
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import load_data as ld
import run_features as rf
import readiness as rd
import viz_all_runs as va

FT_PER_M = 3.280839895
OUT = "figures/readiness_dashboard.html"
VENDOR = "figures/vendor/echarts.min.js"
CDN = "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"
MLB_AVG_FTPS = 27.0
MAX_CURVE_PTS = 120
ACCEL_CAP_MS2 = 10.0          # plausibility cap (PROJECT_BRIEF.md sec 3.3): never show impossible accel


def num(x, d=2):
    """Round to d dp; return None for NaN/inf so JSON stays valid and JS renders an em dash."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if np.isfinite(x) else None


def downsample(t, v, max_pts=MAX_CURVE_PTS):
    """Thin a speed curve to <= max_pts, always keeping the first, last, and peak frames so the
    shape and Vmax are preserved."""
    n = len(t)
    if n <= max_pts:
        idx = list(range(n))
    else:
        k = int(np.ceil(n / max_pts))
        keep = set(range(0, n, k)) | {0, n - 1, int(np.argmax(v))}
        idx = sorted(keep)
    return [round(float(t[i]), 3) for i in idx], [round(float(v[i]), 2) for i in idx]


def per_run_curves(df, runs):
    """Per-run downsampled CoM speed-vs-time curve (keyed by play_id), plus Vmax and the HSR
    threshold. Reuses run_features.com_speed/segment_run so it matches the rest of the pipeline.
    (Acceleration is now the time-to-90% metric computed in readiness.py; HSR per run too.)"""
    Vmax = float(runs["F1_peak1s_ftps"].max())          # 28.87 ft/s
    hsr_thr = 0.75 * Vmax                                 # 21.65 ft/s (PROJECT_BRIEF.md sec 3.3)
    curves = {}
    for pid in runs["play_id"]:
        d = df[df["play_id"] == pid]
        t, pos, speed = rf.com_speed(d)
        xy_raw = d.sort_values("t_sec")[["MidHip_x", "MidHip_y"]].to_numpy()
        seg = rf.segment_run(t, xy_raw, speed)
        s, e = seg["start"], seg["end"]
        tc, vc = downsample(t[s:e + 1] - t[s], speed[s:e + 1])
        curves[pid] = {"t": tc, "v": vc}
    return curves, Vmax, hsr_thr


def load_tag(load, lo, hi):
    if load is None:
        return "n/a"
    return "low" if load <= lo else ("high" if load >= hi else "mid")


VLABEL = {"green": "Within baseline", "amber": "Borderline", "red": "Not Recovered", "no-read": "No read"}


def game_verdict(g, lo, hi):
    """Verdict from the next-day readiness roll-up over 4 indicators (prior-7d load + sprint, burst,
    time-to-90%, each vs his baseline). Days-rest is shown as context, not part of the roll-up."""
    level = g["readiness"]
    if level == "no-read":
        return {"label": VLABEL[level], "level": level,
                "reasons": ["No max-effort run this game - physical output can't be assessed."]}
    ll = "green" if g["load_prior7d_ft"] <= lo else ("red" if g["load_prior7d_ft"] >= hi else "amber")
    inds = [("prior-7d load", ll), ("sprint speed", rd._light(g["F1_z"])),
            ("burst", rd._light(g["burst_z"])), ("time-to-90%", rd._light(g["t90_z"]))]
    flagged = [f"{nm} ({l})" for nm, l in inds if l in ("amber", "red")]
    reasons = ["Flagged: " + ", ".join(flagged) + "." if flagged
               else "All 4 indicators within baseline."]
    reasons.append(f"Composite output {g['output_index_z']:+.2f} SD (context).")
    reasons.append(f"{int(g['n_qual'])} qualifying run(s) - {g['confidence']}.")
    if pd.notna(g["days_rest"]) and g["days_rest"] <= 1:
        reasons.append(f"Short rest ({int(g['days_rest'])}d) - context, not in the score.")
    if g["confidence"] != "ok":
        reasons.append("Single-observation read (n_qual<2) - corroborate with the trend.")
    return {"label": VLABEL[level], "level": level, "reasons": reasons}


def build_data():
    df = ld.load_reshaped()
    runs = rd.build_run_table(df)
    games = rd.build_game_table(runs)
    curves, Vmax, hsr_thr = per_run_curves(df, runs)

    # baseline computed live, identically to viz_readiness.py (qualifying runs, ddof=1)
    q = runs[runs["qualifying"]]
    mu, sd = float(q["F1_peak1s_ftps"].mean()), float(q["F1_peak1s_ftps"].std(ddof=1))

    # outcome class (single/double/triple/scored/out) reused from viz_all_runs
    outcome = {r["pid"]: r["cls"] for r in va.extract_runs(df)}

    # effort tier name from the clustering step, if available
    tier = {}
    if os.path.exists("run_clusters.csv"):
        rc = pd.read_csv("run_clusters.csv")
        tier = dict(zip(rc["play_id"], rc["tier_name"]))

    runs = runs.copy()
    runs["iso"] = pd.to_datetime(runs["game_date"]).dt.strftime("%Y-%m-%d")

    run_rows = []
    for _, r in runs.iterrows():
        pid = r["play_id"]
        run_rows.append({
            "play_id": pid, "game_date": r["iso"],
            "outcome": outcome.get(pid, "-"),
            "tier_name": tier.get(pid, "max-effort" if r["qualifying"] else "sub-max"),
            "qualifying": bool(r["qualifying"]), "effort_score": num(r["effort_score"], 2),
            "F1_peak1s_ftps": num(r["F1_peak1s_ftps"], 2), "F1_z": num(r["F1_peak1s_ftps_z"], 2),
            "F2_burst_ft": num(r["F2_burst_ft"], 1), "burst_z": num(r["F2_burst_ft_z"], 2),
            "output_index_z": num(r["output_index_z"], 2),
            "run_decline_level": rd._light(r["F1_peak1s_ftps_z"]),
            "t90_s": num(r["t90_s"], 2), "t90_z": num(r["t90_s_z"], 2),
            "hsr_ft": num(r["hsr_ft"], 1), "hsr_z": num(r["hsr_ft_z"], 2),
            "h2f_90ft_s": num(r["h2f_90ft_s"], 2), "h2f_z": num(r["h2f_90ft_s_z"], 2),
            "straightness": num(r["F3_straightness"], 2), "sustain": num(r["F5_sustain_frac"], 2),
            "brake_decel": num(r["F7_brake_decel_ftps2"], 1), "max_disp_ft": num(r["max_disp_ft"], 0),
            "path_len_ft": num(r["path_len_ft"], 0), "run_dur_s": num(r["run_dur_s"], 2),
            "base_reached": (r["base_reached"] if pd.notna(r["base_reached"]) else None),
            "speed_curve": curves[pid],
        })

    ids_by_game = {iso: sub["play_id"].tolist()
                   for iso, sub in runs.groupby("iso")}

    loads = games["load_prior7d_ft"].to_numpy(dtype=float)
    lo, hi = float(np.quantile(loads, 1 / 3)), float(np.quantile(loads, 2 / 3))

    game_rows = []
    for _, g in games.iterrows():
        iso = pd.Timestamp(g["game_date"]).strftime("%Y-%m-%d")
        game_rows.append({
            "date": iso, "n_runs": int(g["n_runs"]), "n_qual": int(g["n_qual"]),
            "sprint_ftps": num(g["sprint_ftps"], 1), "burst_ft": num(g["burst_ft"], 1),
            "t90_s": num(g["t90_s"], 2),
            "F1_z": num(g["F1_z"], 2), "burst_z": num(g["burst_z"], 2), "h2f_z": num(g["h2f_z"], 2),
            "hsr_z": num(g["hsr_z"], 2), "t90_z": num(g["t90_z"], 2),
            "n_amber": int(g["n_amber"]), "n_red": int(g["n_red"]),
            "output_index_z": num(g["output_index_z"], 2),
            "readiness": g["readiness"], "confidence": g["confidence"],
            "days_rest": (None if pd.isna(g["days_rest"]) else int(g["days_rest"])),
            "load_prior7d_ft": num(g["load_prior7d_ft"], 0), "runs_prior7d": int(g["runs_prior7d"]),
            "verdict": game_verdict(g, lo, hi),
            "run_play_ids": ids_by_game.get(iso, []),
        })

    data = {
        "meta": {"player": "single minor-league batter", "season": "May 2025",
                 "n_runs": len(run_rows), "n_games": len(game_rows),
                 "units": {"speed": "ft/s", "distance": "ft", "time": "s", "accel": "m/s^2"},
                 "framing": "Decision support, not an automated benching rule.",
                 "source": ["readiness.py", "run_features.py", "viz_all_runs.py", "cluster_runs.py"]},
        "baseline": {"metric": "Sprint Speed (peak 1-s, ft/s)", "n_qual": int(len(q)),
                     "mu": num(mu, 2), "sd": num(sd, 2),
                     "green_floor": num(mu + rd.GREEN_Z * sd, 2), "red_floor": num(mu + rd.RED_Z * sd, 2),
                     "Vmax": num(Vmax, 2), "hsr_threshold": num(hsr_thr, 2), "mlb_avg_ftps": MLB_AVG_FTPS,
                     "load_lo": num(lo, 0), "load_hi": num(hi, 0),
                     "constants": {"GREEN_Z": rd.GREEN_Z, "RED_Z": rd.RED_Z, "QUAL_EFFORT": rd.QUAL_EFFORT,
                                   "ROLL_N": rd.ROLL_N, "BUTTER_CUTOFF_HZ": rf.BUTTER_CUTOFF_HZ}},
        "validation": {
            "n_lowconf_games": int((games["confidence"] != "ok").sum()),
            "notes": [
                "~1 max-effort run per game - trust the rolling trend, not a single game.",
                "Composite = equal-weighted mean of 4 self-baseline z-scores (sprint speed, burst, high-speed distance, time-to-90%); same SD bands.",
                "High-speed distance partly reflects run length (a double covers more than a single), so read it as output, not pure fitness - and it also feeds the prior-7d load, which couples that load with the composite.",
                "Speed from the MidHip center-of-mass proxy (not a full de Leva CoM); 30 fps band-limits fine acceleration timing.",
                "Home-to-first is a 90-ft running-split proxy and is collinear with sprint speed (confirmatory only).",
            ]},
        "games": game_rows, "runs": run_rows,
    }
    return data


# ------------------------------------------------------------------------------------------
# HTML template (single inlined <style> + app <script>; __ECHARTS_BLOCK__ and __DATA__ injected)
# ------------------------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Readiness Dashboard - single batter, May 2025</title>
<style>
  :root{
    --green:#009E73; --amber:#E69F00; --red:#D7263D; --noread:#9aa0a6;
    --bg:#f6f7f9; --card:#ffffff; --ink:#1f2430; --muted:#6b7280; --line:#e5e7eb; --accent:#0072B2;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       font-variant-numeric:tabular-nums;font-size:14px;line-height:1.45}
  .wrap{max-width:1160px;margin:0 auto;padding:20px}
  header h1{font-size:21px;margin:0 0 2px}
  header p{margin:0;color:var(--muted);font-size:13px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
        box-shadow:0 1px 3px rgba(0,0,0,0.05);padding:18px;margin:16px 0}
  h2{font-size:15px;font-weight:600;margin:0 0 10px}
  h3{font-size:14px;font-weight:600;margin:2px 0 8px;color:var(--muted)}
  .kpi-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .kpi .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  .kpi .val{font-size:24px;font-weight:600;margin-top:4px}
  .kpi .sub{font-size:11px;color:var(--muted);margin-top:2px}
  .kpi.good{background:rgba(0,158,115,0.10);border-color:rgba(0,158,115,0.40)}
  .kpi.warn{background:rgba(230,159,0,0.10);border-color:rgba(230,159,0,0.40)}
  .kpi.bad{background:rgba(215,38,61,0.12);border-color:rgba(215,38,61,0.45)}
  .kpi.noread{background:var(--card)}
  @media(max-width:820px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
  .selector{padding:12px 16px}
  .sel-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px}
  .chips{display:flex;flex-wrap:wrap;gap:8px}
  .chip{display:flex;align-items:center;gap:7px;cursor:pointer;background:#fff;border:1px solid var(--line);
        border-radius:999px;padding:6px 12px;font-size:12px;color:var(--ink);transition:all .12s}
  .chip:hover{border-color:#c3c8d0;background:#fafbfc}
  .chip.sel{border-color:var(--accent);background:#eaf4fb;box-shadow:0 0 0 3px rgba(0,114,178,.22);
            font-weight:600;transform:translateY(-1px)}
  .chip .dot{width:11px;height:11px;border-radius:50%}
  .chip .cd{font-weight:600} .chip .cl{font-size:10px;letter-spacing:.03em}
  .verdict{border-left:6px solid var(--noread);background:#fbfcfd;border-radius:8px;padding:12px 16px;margin-bottom:14px}
  .verdict .vhead{font-size:18px;font-weight:700;display:flex;align-items:center;gap:10px}
  .verdict .vchip{width:13px;height:13px;border-radius:50%;display:inline-block}
  .verdict ul{margin:8px 0 6px;padding-left:18px;color:#374151}
  .verdict ul li{margin:2px 0}
  .verdict .vfoot{font-size:12px;color:var(--muted);font-style:italic}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
  th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
  th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;font-weight:600}
  td.n,th.n{text-align:right}
  tr.submax{opacity:.55} tr.submax td{font-style:normal}
  .z{color:var(--muted);font-size:11px}
  .ddot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
  .badge{display:inline-block;color:#fff;border-radius:5px;padding:1px 7px;font-size:11px;font-weight:600}
  .tablewrap{overflow-x:auto}
  footer{color:var(--muted);font-size:12px}
  footer ul{margin:6px 0 0;padding-left:18px}
  .legend{font-size:11px;color:var(--muted);margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Running readiness - single batter, May 2025</h1>
    <p>Self-baseline traffic light on physical output during runs out of the box. Decision support, not an automated benching rule.</p>
  </header>

  <section class="card selector">
    <div class="sel-label">Game day — click to inspect (or use ← / → keys)</div>
    <div class="chips" id="chips"></div>
  </section>

  <section class="kpi-grid">
    <div class="kpi" id="card-readiness"><div class="lab">Next-day readiness</div><div class="val" id="kpi-readiness">-</div><div class="sub" id="kpi-readiness-sub">-</div></div>
    <div class="kpi" id="card-load"><div class="lab">Prior-7d load (ft)</div><div class="val" id="kpi-load">-</div><div class="sub" id="kpi-load-sub">-</div></div>
    <div class="kpi" id="card-accel"><div class="lab">Time to 90% vel (s)</div><div class="val" id="kpi-accel">-</div><div class="sub" id="kpi-accel-sub">-</div></div>
    <div class="kpi" id="card-sprint"><div class="lab">Sprint Speed (ft/s)</div><div class="val" id="kpi-sprint">-</div><div class="sub" id="kpi-sprint-sub">-</div></div>
    <div class="kpi" id="card-burst"><div class="lab">Burst (ft)</div><div class="val" id="kpi-burst">-</div><div class="sub" id="kpi-burst-sub">-</div></div>
    <div class="kpi" id="card-rest"><div class="lab">Days rest</div><div class="val" id="kpi-rest">-</div><div class="sub">since prev game</div></div>
  </section>

  <section class="card">
    <h2>Season overview - prior-7-day running load (click a point, or pick a day at the top)</h2>
    <div id="overview" style="height:380px"></div>
    <div class="legend">Y-axis = prior-7-day high-speed-running distance (ft; CoM path above 75% of his max speed). Zones (his own tertiles): green = low / fresh, amber = mid, red = high (most fatigue exposure). Dot color = the load zone. Click a point to inspect that day (readiness verdict + per-run detail appear below).</div>
  </section>

  <section class="card">
    <div class="verdict" id="verdict"></div>
    <h3>Per-run detail - <span id="detail-date"></span></h3>
    <div class="tablewrap">
    <table id="runTable">
      <thead><tr>
        <th>Outcome</th><th>Tier</th><th class="n">Effort</th><th class="n">Sprint (ft/s)</th>
        <th class="n">Burst (ft)</th><th class="n">Time→90% (s)</th><th class="n">HSR (ft)</th>
        <th class="n">H2F (s)</th><th class="n">Straight</th><th class="n">Sustain</th><th>Run status</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    </div>
    <div class="legend">Sprint shows (z vs baseline). HSR = distance above 75% of his max speed. Time→90% = seconds from first step to 90% of the run's peak speed (acceleration phase). H2F (home-to-first proxy) is confirmatory - collinear with sprint speed. Sub-max runs are shown for context but excluded from the verdict (an easy-out jog isn't fatigue).</div>
  </section>
</div>

__ECHARTS_BLOCK__
<script>const DATA = __DATA__;</script>
<script>
const C={green:"#009E73",amber:"#E69F00",red:"#D7263D","no-read":"#9aa0a6"};
const LAB={green:"WITHIN BASELINE",amber:"BORDERLINE",red:"NOT RECOVERED","no-read":"NO READ"};
const OC={single:"#0072B2",double:"#009E73",triple:"#7B3FA0",scored:"#D55E00",out:"#9aa0a6",noreach:"#9aa0a6","-":"#9aa0a6"};
const B=DATA.baseline, GZ=B.constants.GREEN_Z, RZ=B.constants.RED_Z;
const runsById={}; DATA.runs.forEach(r=>runsById[r.play_id]=r);
let sel=0, ov, dc;
const f=(x,d=2)=>(x===null||x===undefined||Number.isNaN(x))?"—":Number(x).toFixed(d);
const fz=(x,d=2)=>{if(x===null||x===undefined||Number.isNaN(x))return "—";const v=Number(x);return (v>0?"+":"")+v.toFixed(d)+" SD";};
function light(z){if(z===null||z===undefined||Number.isNaN(z))return 'no-read';return z>=GZ?'green':(z<=RZ?'red':'amber');}
function tintOf(level){return level==='green'?'good':level==='amber'?'warn':level==='red'?'bad':'noread';}
function setCard(id,cls){document.getElementById(id).className='kpi '+cls;}
function loadTint(x){if(x===null||x===undefined)return 'noread';return x<=B.load_lo?'good':(x>=B.load_hi?'bad':'warn');}

function initOverview(){
  ov=echarts.init(document.getElementById('overview'));
  const g=DATA.games;
  const LT2C={good:C.green,warn:C.amber,bad:C.red,noread:C['no-read']};
  const loads=g.map(x=>x.load_prior7d_ft);
  const ymax=Math.max(60,Math.ceil(Math.max.apply(null,loads)/50)*50+30);
  const pts=g.map(x=>({value:x.load_prior7d_ft,symbolSize:14,
      itemStyle:{color:LT2C[loadTint(x.load_prior7d_ft)]}}));
  ov.setOption({
    grid:{left:16,right:26,top:28,bottom:16,containLabel:true},
    tooltip:{trigger:'axis',formatter:p=>{const x=g[p[0].dataIndex];
      const tag=x.load_prior7d_ft<=B.load_lo?'low':(x.load_prior7d_ft>=B.load_hi?'high':'mid');
      return `<b>${x.date}</b><br>prior-7d high-speed dist: ${f(x.load_prior7d_ft,0)} ft (${tag})`;}},
    xAxis:{type:'category',data:g.map(x=>x.date.slice(5)),axisLabel:{rotate:45,fontSize:11}},
    yAxis:{type:'value',name:'prior-7d high-speed distance (ft)',nameGap:44,min:0,max:ymax,nameTextStyle:{fontSize:11}},
    series:[
      {name:'load',type:'line',data:pts,z:6,lineStyle:{color:'#b8bdc6',width:1.4},
       markArea:{silent:true,data:[
         [{yAxis:0,itemStyle:{color:'rgba(0,158,115,0.07)'}},{yAxis:B.load_lo}],
         [{yAxis:B.load_lo,itemStyle:{color:'rgba(230,159,0,0.10)'}},{yAxis:B.load_hi}],
         [{yAxis:B.load_hi,itemStyle:{color:'rgba(215,38,61,0.12)'}},{yAxis:ymax}]]},
       markLine:{silent:true,symbol:'none',label:{position:'end',fontSize:10},data:[
         {yAxis:B.load_lo,name:'low',lineStyle:{type:'dashed',color:'#aab'}},
         {yAxis:B.load_hi,name:'high',lineStyle:{type:'dashed',color:'#aab'}}]}}
    ]
  });
  ov.on('click',p=>{if(p.seriesName==='load')selectGame(p.dataIndex);});
}

function updateKPIs(g){
  // 1. next-day readiness (roll-up of the 4 indicator cards below; days-rest is context)
  const rk=document.getElementById('kpi-readiness'); rk.textContent=LAB[g.readiness]; rk.style.color=C[g.readiness];
  document.getElementById('kpi-readiness-sub').textContent=g.date;
  setCard('card-readiness', tintOf(g.readiness));
  // 2. prior-7d high-speed-running load (low=fresh/good, high=more fatigue exposure/bad)
  document.getElementById('kpi-load').textContent=f(g.load_prior7d_ft,0);
  setCard('card-load', loadTint(g.load_prior7d_ft));
  // 3. time to 90% velocity (game effort-weighted), tinted by its z
  document.getElementById('kpi-accel').textContent=g.t90_s==null?'—':f(g.t90_s,2);
  document.getElementById('kpi-accel-sub').textContent=g.t90_z==null?'no qual run':fz(g.t90_z,1);
  setCard('card-accel', g.t90_z==null?'noread':tintOf(light(g.t90_z)));
  // 4. Sprint Speed (game effort-weighted), tinted by its z
  document.getElementById('kpi-sprint').textContent=g.sprint_ftps==null?'—':f(g.sprint_ftps,1);
  document.getElementById('kpi-sprint-sub').textContent=g.F1_z==null?'no qual run':fz(g.F1_z,1);
  setCard('card-sprint', g.F1_z==null?'noread':tintOf(light(g.F1_z)));
  // 5. Burst (game effort-weighted), tinted by its z
  document.getElementById('kpi-burst').textContent=g.burst_ft==null?'—':f(g.burst_ft,1);
  document.getElementById('kpi-burst-sub').textContent=g.burst_z==null?'no qual run':fz(g.burst_z,1);
  setCard('card-burst', g.burst_z==null?'noread':tintOf(light(g.burst_z)));
  // 6. days rest - red if 1, amber if 2, green if >2 (context only, not in readiness)
  const dr=g.days_rest;
  document.getElementById('kpi-rest').textContent=(dr==null?'—':dr);
  setCard('card-rest', dr==null?'noread':(dr<=1?'bad':(dr===2?'warn':'good')));
}

function updateVerdict(g){
  const v=g.verdict,el=document.getElementById('verdict');
  el.style.borderLeftColor=C[v.level];
  el.innerHTML=`<div class="vhead" style="color:${C[v.level]}">${v.label}`+
    `<span class="vchip" style="background:${C[v.level]}"></span></div>`+
    `<ul>${v.reasons.map(r=>`<li>${r}</li>`).join('')}</ul>`+
    `<div class="vfoot">Check-engine light for HP staff - combine with context; not an automated benching rule.</div>`;
  document.getElementById('detail-date').textContent=g.date;
}

function updateTable(g){
  const tb=document.querySelector('#runTable tbody'); tb.innerHTML='';
  g.run_play_ids.map(p=>runsById[p]).filter(Boolean)
    .sort((a,b)=>(b.F1_peak1s_ftps||0)-(a.F1_peak1s_ftps||0)).forEach(r=>{
    const tr=document.createElement('tr'); if(!r.qualifying)tr.className='submax';
    const oc=`<span class="badge" style="background:${OC[r.outcome]||'#9aa0a6'}">${r.outcome||'—'}</span>`;
    const st=`<span class="ddot" style="background:${C[r.run_decline_level]}"></span>${r.run_decline_level}`;
    tr.innerHTML=`<td>${oc}</td><td>${r.tier_name||'—'}</td>`+
      `<td class="n">${f(r.effort_score)}</td>`+
      `<td class="n">${f(r.F1_peak1s_ftps,1)} <span class="z">(${fz(r.F1_z,1)})</span></td>`+
      `<td class="n">${f(r.F2_burst_ft,1)}</td>`+
      `<td class="n">${f(r.t90_s,2)}</td>`+
      `<td class="n">${f(r.hsr_ft,0)}</td><td class="n">${f(r.h2f_90ft_s,2)}</td>`+
      `<td class="n">${f(r.straightness)}</td><td class="n">${f(r.sustain,2)}</td><td>${st}</td>`;
    tb.appendChild(tr);
  });
}

function buildChips(){
  const c=document.getElementById('chips'); c.innerHTML='';
  DATA.games.forEach((g,i)=>{
    const b=document.createElement('button'); b.className='chip'; b.dataset.idx=i;
    b.innerHTML=`<span class="dot" style="background:${g.confidence==='ok'?C[g.readiness]:'#fff'};`+
      `border:2px solid ${C[g.readiness]}"></span><span class="cd">${g.date.slice(5)}</span>`+
      `<span class="cl" style="color:${C[g.readiness]}">${LAB[g.readiness]}</span>`;
    b.addEventListener('click',()=>selectGame(i)); c.appendChild(b);
  });
}

function selectGame(i){
  if(i<0||i>=DATA.games.length)return; sel=i; const g=DATA.games[i];
  document.querySelectorAll('.chip').forEach((el,j)=>el.classList.toggle('sel',j===i));
  if(ov){ov.dispatchAction({type:'downplay',seriesName:'load'});
          ov.dispatchAction({type:'highlight',seriesName:'load',dataIndex:i});}
  updateKPIs(g); updateVerdict(g); updateTable(g);
  history.replaceState(null,'','#game='+g.date);
}

function start(){
  initOverview(); buildChips();
  document.getElementById('kpi-load-sub').textContent='dist ran >'+f(B.hsr_threshold,1)+' ft/s, past 7 days';
  let i=0; const m=location.hash.match(/game=(\d{4}-\d{2}-\d{2})/);
  if(m){const k=DATA.games.findIndex(g=>g.date===m[1]); if(k>=0)i=k;}
  selectGame(i);
  window.addEventListener('keydown',e=>{
    if(e.key==='ArrowRight')selectGame(Math.min(sel+1,DATA.games.length-1));
    if(e.key==='ArrowLeft')selectGame(Math.max(sel-1,0));});
  window.addEventListener('resize',()=>{ov&&ov.resize();dc&&dc.resize();});
}
start();
</script>
</body>
</html>
"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = build_data()

    # JSON, hardened against accidental </script> breakage
    payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")

    if os.path.exists(VENDOR):
        with open(VENDOR, encoding="utf-8") as fh:
            echarts_block = "<script>\n" + fh.read() + "\n</script>"
        mode = f"inlined ({os.path.getsize(VENDOR) // 1024} KB, offline)"
    else:
        echarts_block = f'<script src="{CDN}"></script>'
        mode = "CDN fallback (needs internet)"
        print(f"WARNING: {VENDOR} not found - using {mode}.", file=sys.stderr)

    html = TEMPLATE.replace("__ECHARTS_BLOCK__", echarts_block).replace("__DATA__", payload)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)

    b = data["baseline"]
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.2f} MB)  echarts: {mode}")
    print(f"games: {data['meta']['n_games']}  runs: {data['meta']['n_runs']}  "
          f"low-confidence games: {data['validation']['n_lowconf_games']}")
    print(f"baseline: {b['mu']:.2f} ± {b['sd']:.2f} ft/s (n={b['n_qual']})  "
          f"amber<{b['green_floor']:.1f}  red<{b['red_floor']:.1f}  "
          f"Vmax {b['Vmax']:.1f}  HSR>{b['hsr_threshold']:.1f} ft/s")


if __name__ == "__main__":
    main()
