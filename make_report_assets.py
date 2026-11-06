"""Build the report's figures (PDF) and LaTeX tables from the versioned result summaries.

Every number in report/tables/*.tex and report/figures/*.pdf is read from results/final_run/
(fk_run/summary.json for the final method B-FK; suite/summary.json, analysis/summary.json, memory/summary.json,
grid_*/grid.json, gru_*/gru.json for the URDF study), so the report cannot
drift from the code. The hold-trace and memory-trace figures need per-step signals; they are extracted once
from a full run (--artifacts) into results/final_run/analysis/hold_trace.csv and
results/final_run/memory/trace_<scene>.csv, which are versioned.

Usage: python make_report_assets.py [--results results/final_run] [--artifacts artifacts/final_run]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated default categorical slots 1-3 (blue, orange, aqua); markers give a second encoding for print.
STYLE = {"B": dict(color="#2a78d6", marker="o", label="B-FK (ours)"),
         "expert": dict(color="#eb6834", marker="s", label="Expert"),
         "v3": dict(color="#1baf7a", marker="^", label="v3 (baseline)")}
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e6ea"
COL_W = 3.5  # IEEE single column, inches
TRACE_SCENE = "scene_2026_0008"
MEMORY_SCENE = "scene_2026_0012"  # fresh scene where L0 and L2 fail and L1 and L3 succeed (typical of Table II)
LEVEL_STYLE = {"L0": dict(color="#8a929c", label="L0 plain MLP", lw=0.6, zorder=1),
               "L1": dict(color="#1baf7a", label="L1 latch only", lw=1.2, zorder=3),
               "L2": dict(color="#eb6834", label="L2 hidden state only", lw=1.2, zorder=2),
               "L3": dict(color="#2a78d6", label="L3 = B-FK", lw=1.4, zorder=4)}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def pct(x):
    return f"{100 * x:.1f}"


def ci(w):
    return f"{100 * w[0]:.1f}--{100 * w[1]:.1f}"


def pval(p):
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


# ---------------------------------------------------------------- figures
def fk_condition_rows(fk):
    """fk_run/summary.json robustness and OOD entries in the row format of suite/summary.json."""
    rows = []
    for key, prefix in (("robustness", "robust_"), ("ood", "ood_")):
        for c, r in fk[key].items():
            rate = lambda x: dict(rate=x["success"] / x["scenes"], wilson95=x["wilson95"])
            rows.append(dict(condition=prefix + c, scenes=r["B_FK"]["scenes"], B=rate(r["B_FK"]), expert=rate(r["expert"]),
                             B_vs_expert=r["B_FK"]["paired"]))
    return rows


def fig_conditions(suite, out):
    """Robustness and OOD: success rate with Wilson CI, B-FK vs expert, one row per condition."""
    names = {"robust_noise003": "Noise $\\sigma$=0.003", "robust_noise010": "Noise $\\sigma$=0.01",
             "robust_torque080": "Arm torque 80%", "robust_pushee30": "EE push 30 N",
             "robust_pushbase20": "Base push 20 N", "ood_far_target": "OOD: far target",
             "ood_high_target": "OOD: high target", "ood_low_target": "OOD: low target",
             "ood_wide_start": "OOD: wide start"}
    rows = [r for r in suite["conditions"] if r["condition"] in names]
    rows.sort(key=lambda r: list(names).index(r["condition"]))
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    y = np.arange(len(rows))[::-1]
    for k, (c, dy) in enumerate((("B", 0.17), ("expert", -0.17))):
        rate = np.array([100 * r[c]["rate"] for r in rows])
        lo = rate - np.array([100 * r[c]["wilson95"][0] for r in rows])
        hi = np.array([100 * r[c]["wilson95"][1] for r in rows]) - rate
        lo, hi = np.clip(lo, 0, None), np.clip(hi, 0, None)  # float round-off at 0% / 100%
        ax.errorbar(rate, y + dy, xerr=[lo, hi], fmt=STYLE[c]["marker"], color=STYLE[c]["color"],
                    ms=4.5, elinewidth=1.2, capsize=0, label=STYLE[c]["label"],
                    markeredgecolor="white", markeredgewidth=0.6)
    for yi, r in zip(y, rows):
        p = r["B_vs_expert"]["mcnemar_p"]
        if p < 0.05:
            ax.text(101.5, yi, f"p={p:.0e}".replace("e-0", "e-"), va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [names[r["condition"]] + f" (n={r['scenes']})" for r in rows])
    ax.set_xlim(20, 112)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlabel("1 cm success rate (%), 95% Wilson CI")
    ax.axhline(y[4] - 0.5, color=MUTED, lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.0))
    fig.savefig(out / "conditions.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig_learning_curve(an, out):
    lc = sorted(an["learning_curve"].values(), key=lambda r: r["train_trajectories"])
    n = np.array([r["train_trajectories"] for r in lc])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.55), gridspec_kw=dict(wspace=0.42))
    rate = np.array([100 * r["success"] / r["scenes"] for r in lc])
    lo = rate - np.array([100 * r["wilson95"][0] for r in lc])
    hi = np.array([100 * r["wilson95"][1] for r in lc]) - rate
    a1.fill_between(n, rate - lo, rate + hi, color=STYLE["B"]["color"], alpha=0.15, lw=0)
    a1.plot(n, rate, color=STYLE["B"]["color"], marker="o", ms=4, markeredgecolor="white", markeredgewidth=0.6)
    for x, r in zip(n, lc):
        a1.annotate(f"{r['success']}", (x, 100 * r["success"] / r["scenes"]), textcoords="offset points",
                    xytext=(0, -10 if x < 200 else 5), ha="center", fontsize=6.5, color=INK)
    a1.set_xscale("log")
    a1.set_xticks(n, [str(v) for v in n])
    a1.minorticks_off()
    a1.set_ylim(60, 103)
    a1.set_xlabel("training trajectories")
    a1.set_ylabel("validation success (%)")
    for key, lab, ls in (("validation_wheel_loss", "wheel", "-"), ("validation_arm_loss", "arm", "--"),
                         ("validation_stop_loss", "stop", ":")):
        a2.plot(n, [r[key] for r in lc], color=INK, ls=ls, lw=1.3, label=lab)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xticks(n, [str(v) for v in n])
    a2.minorticks_off()
    a2.set_xlabel("training trajectories")
    a2.set_ylabel("validation loss")
    a2.legend(frameon=False, handlelength=2.0, ncol=3, loc="lower left", bbox_to_anchor=(-0.05, 1.0),
               columnspacing=0.8)
    fig.savefig(out / "learning_curve.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def hold_trace(results, artifacts, fk_rollouts):
    """Per-step EE error in TRACE_SCENE during the hold test: B-FK (fk_rollouts/hold_L3), v3 and expert
    (URDF study). Extracted once into results/final_run/analysis/hold_trace.csv."""
    path = results / "analysis" / "hold_trace.csv"
    if artifacts and (artifacts / "analysis").is_dir() and (fk_rollouts / "hold_L3").is_dir():
        cols = {}
        for c in ("B", "v3", "expert"):
            folder = ((fk_rollouts / "hold_L3") if c == "B" else (artifacts / "analysis" / f"hold_{c}")) / "rollouts" / TRACE_SCENE
            err = np.load(folder / "rollout.npz")["ee_error"]
            res = json.loads((folder / "result.json").read_text())
            n_hold = int(round(res["hold"]["completed_s"] * 1000))
            cols[c] = err[-(n_hold + 3000):]  # 3 s before success + hold
        length = min(len(v) for v in cols.values())
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_rel_success_s", *cols])
            for i in range(length):
                w.writerow([f"{(i - (length - 2001)) / 1000:.3f}", *(f"{cols[c][len(cols[c]) - length + i]:.6f}" for c in cols)])
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def fig_hold(trace, out):
    fig, ax = plt.subplots(figsize=(COL_W, 1.45))
    for c in ("expert", "v3", "B"):
        ax.plot(trace["t_rel_success_s"], 1000 * trace[c], color=STYLE[c]["color"], lw=1.2,
                label=STYLE[c]["label"] + (" Kd=10" if c == "v3" else ""))
    ax.axhline(10, color=MUTED, lw=0.8, ls="--")
    ax.text(-1.45, 11, "1 cm", fontsize=6.5, color=MUTED)
    ax.axvspan(0, 2, color=GRID, alpha=0.5, lw=0)
    ax.text(1.0, 44, "hold window", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xlim(-1.5, 2)
    ax.set_ylim(0, 50)
    ax.set_xlabel("time relative to each controller's first 1 cm success (s)")
    ax.set_ylabel("EE error (mm)")
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig(out / "hold_trace.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def memory_trace(results, artifacts, fk_rollouts):
    """Per-level wheel command and EE error in MEMORY_SCENE (fresh set), every 10 ms; extracted once."""
    path = results / "memory" / f"trace_{MEMORY_SCENE}.csv"
    if artifacts and (artifacts / "memory").is_dir() and (fk_rollouts / "test_L3").is_dir():
        folders = {"L0": artifacts / "direct" / "fresh_mlp", "L1": fk_rollouts / "test_L1",
                   "L2": fk_rollouts / "test_L2", "L3": fk_rollouts / "test_L3"}
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "t_s", "max_abs_wheel_cmd", "ee_error_m", "done", "success_1cm"])
            for lvl, folder in folders.items():
                d = np.load(folder / "rollouts" / MEMORY_SCENE / "rollout.npz")
                ok = json.loads((folder / "rollouts" / MEMORY_SCENE / "result.json").read_text())["success_1cm"]
                wheel = np.abs(d["actions"][:, :4]).max(axis=1)
                for k in range(0, len(wheel), 10):
                    w.writerow([lvl, f"{k / 1000:.2f}", f"{wheel[k]:.5f}", f"{d['ee_error'][k]:.6f}",
                                int(d["done"][k]), int(ok)])
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for lvl in LEVEL_STYLE:
        r = [x for x in rows if x["level"] == lvl]
        out[lvl] = dict(t=np.array([float(x["t_s"]) for x in r]), wheel=np.array([float(x["max_abs_wheel_cmd"]) for x in r]),
                        error=np.array([float(x["ee_error_m"]) for x in r]), success=r[0]["success_1cm"] == "1")
    return out


def fig_memory_trace(trace, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.35), sharex=True, gridspec_kw=dict(hspace=0.12))
    for lvl, st in LEVEL_STYLE.items():
        d = trace[lvl]
        kw = dict(color=st["color"], lw=st["lw"], zorder=st["zorder"])
        a1.plot(d["t"], d["wheel"], **kw, label=st["label"])
        a2.plot(d["t"], np.maximum(d["error"], 1e-4), **kw)
        if d["success"]:
            a2.plot(d["t"][-1], d["error"][-1], marker="o", ms=4, color=st["color"], markeredgecolor="white",
                    markeredgewidth=0.6, zorder=5)
    a1.axhline(0.05, color=MUTED, lw=0.7, ls="--")
    a1.text(39.5, 0.12, "0.05 rad/s", ha="right", fontsize=6.5, color=MUTED)
    a1.set_yscale("symlog", linthresh=0.05)
    a1.set_ylim(0, 4)
    a1.set_yticks([0, 0.05, 0.5, 3], ["0", "0.05", "0.5", "3"])
    a1.set_ylabel("max |wheel cmd|\n(rad/s)")
    a1.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=1.0)
    a2.axhline(0.01, color=MUTED, lw=0.7, ls="--")
    a2.text(39.5, 0.0045, "1 cm", ha="right", fontsize=6.5, color=MUTED)
    a2.set_yscale("log")
    a2.set_ylim(1e-3, 1e2)
    a2.set_xlim(0, 40)
    a2.set_ylabel("EE error (m)")
    a2.set_xlabel(f"time (s), fresh scene {MEMORY_SCENE.replace('scene_', '')}; dot: 1 cm success")
    fig.savefig(out / "memory_trace.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig_fit_vs_success(an, mem, fk, results, out):
    """Open-loop validation wheel loss (same definition for every GRU) vs closed-loop validation success.
    Filled: URDF end-effector position (B's learning curve and seeds, L2); open: learned FK (B-FK, L2)."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    pts = []  # (loss, success, label, colour, marker)
    for frac, r in an["learning_curve"].items():
        if float(frac) < 1:
            pts.append((r["validation_wheel_loss"], r["success"], f"B, {r['train_trajectories']} traj.", STYLE["B"]["color"], "o"))
    offsets = {"B, 486 traj.": (0, -11), "L2 (no stop head)": (6, -3)}
    for seed, r in an["seeds"].items():
        pts.append((r["validation_wheel_loss"], r["success"], None, STYLE["B"]["color"], "o"))
    l2 = json.loads((results / "memory" / "gru_nostop" / "gru.json").read_text())["metrics"]
    pts.append((l2["validation_wheel_loss"], mem["validation"]["L2"]["success"], "L2 (no stop head)",
                LEVEL_STYLE["L2"]["color"], "s"))
    g_fk = json.loads((results / "fk_run" / "gru_B_fk" / "gru.json").read_text())["metrics"]
    l2_fk = json.loads((results / "fk_run" / "memory" / "gru_nostop_fk" / "gru.json").read_text())["metrics"]
    for x, y, c, m in ((g_fk["validation_wheel_loss"], fk["validation"]["B_FK"]["success"], STYLE["B"]["color"], "o"),
                       (l2_fk["validation_wheel_loss"], fk["memory"]["validation_L2"]["success"], LEVEL_STYLE["L2"]["color"], "s")):
        ax.plot(x, 100 * y / 208, marker=m, ms=4.5, ls="none", markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.1)
    for x, y, lab, c, m in pts:
        ax.plot(x, 100 * y / 208, marker=m, ms=4.5, color=c, ls="none", markeredgecolor="white", markeredgewidth=0.6)
        if lab:
            dx, dy = offsets.get(lab, (6, -3))
            ax.annotate(lab, (x, 100 * y / 208), textcoords="offset points", xytext=(dx, dy), fontsize=6.5, color=INK,
                        ha="center" if dx == 0 else "left")
    seeds = [r["validation_wheel_loss"] for r in an["seeds"].values()]
    ax.text(0.98, 0.08, "filled: URDF FK\nopen: learned FK", transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
    ax.annotate("B, 973 traj., 3 seeds", (max(seeds), 100 * 207 / 208), textcoords="offset points", xytext=(6, -3),
                fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.3], ["0.05", "0.1", "0.2", "0.3"])
    ax.minorticks_off()
    ax.set_xlim(0.045, 0.6)
    ax.set_ylim(15, 108)
    ax.set_xlabel("open-loop validation wheel loss (normalised MSE)", labelpad=2)
    fig.subplots_adjust(right=0.97)
    ax.set_ylabel("val. success (%)")
    fig.savefig(out / "fit_vs_success.pdf", metadata={"CreationDate": None})
    plt.close(fig)


# ---------------------------------------------------------------- tables
def table_main(suite, an, fk, results, out):
    """Fresh test: B-FK (final), B with the URDF (same design, analytic FK and g(q)) and the expert."""
    fresh = next(r for r in suite["conditions"] if r["condition"] == "fresh")
    hold, f, h = an["hold"], fk["fresh"], fk["hold"]
    m = suite["models"]
    cols = [("B_FK", r"\textbf{B-FK (ours)}"), ("B", "B, URDF FK"), ("expert", "Expert")]
    val = dict(B_FK=f["B_FK"], B=fresh["B"], expert=fresh["expert"])
    held = dict(B_FK=h["B_FK"]["held_within_1cm"], B=hold["B"]["held_within_1cm"], expert=hold["expert"]["held_within_1cm"])
    fk_params = fk["fk_model"]["parameters"]
    gru_params = m["B"]["gru_parameters"] + m["B"]["park_mlp_parameters"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", " & " + " & ".join(c for _, c in cols) + r" \\", r"\midrule"]
    lines.append("1 cm success / 300 & " + " & ".join(
        (r"\textbf{%d}" if k == "B_FK" else "%d") % val[k]["success"] for k, _ in cols) + r" \\")
    lines.append(r"\quad rate (\%) & " + " & ".join(pct(val[k]["success"] / 300) for k, _ in cols) + r" \\")
    lines.append(r"\quad 95\% Wilson CI & " + " & ".join(ci(val[k]["wilson95"]) for k, _ in cols) + r" \\")
    lines.append(r"Held $<$1 cm for 2 s & " + " & ".join(str(held[k]) for k, _ in cols) + r" \\")
    lines.append(r"Mean time to success (s) & " + " & ".join(f"{val[k]['mean_success_time_s']:.2f}" for k, _ in cols) + r" \\")
    lines.append(r"Compute per 1 kHz step ($\mu$s) & " + " & ".join(f"{val[k]['policy_time_mean_us']:.0f}" for k, _ in cols) + r" \\")
    lines.append(r"Learned parameters & " + f"{gru_params + fk_params:,} & {gru_params:,} & -- " + r"\\")
    lines.append(r"Robot model in the policy & none & FK, $g(q)$ & FK, IK, $g(q)$ \\")
    p1, p2 = f["B_FK"]["paired"], f["B_FK_vs_B_urdf"]
    lines += [r"\midrule",
              r"\multicolumn{4}{@{}l@{}}{Exact McNemar (discordant pairs): B-FK vs expert %d/%d, $p=%s$;} \\"
              % (p1["only_first"], p1["only_second"], pval(p1["mcnemar_p"])),
              r"\multicolumn{4}{@{}l@{}}{B-FK vs B %d/%d, $p=%s$. Held: B-FK vs expert %d/%d, $p=%s$.} \\"
              % (p2["only_first"], p2["only_second"], pval(p2["mcnemar_p"]), h["B_FK_vs_expert_held"]["only_first"],
                 h["B_FK_vs_expert_held"]["only_second"], pval(h["B_FK_vs_expert_held"]["mcnemar_p"])),
              r"\bottomrule", r"\end{tabular}"]
    (out / "main.tex").write_text("\n".join(lines) + "\n")


def table_memory(mem, fk, out):
    """Memory study: L0 (no memory) and three levels of stage memory with the learned FK, same data and scenes."""
    names = {"L0mlp": (r"L0 plain MLP regressor", "none"),
             "L1": (r"L1 four single-frame MLPs", "latch only"),
             "L2": (r"L2 GRU, no stop head", "hidden state only"),
             "L3": (r"\textbf{L3 = B-FK}: GRU + stop head", "hidden state + latch")}
    fm = fk["memory"]
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Policy & Stage memory & Val./208 & Fresh/300 & $p$ vs L3$^*$ & Never done$^\dagger$ \\", r"\midrule"]
    for lvl, (name, memo) in names.items():
        if lvl == "L0mlp":
            v, f = mem["validation"][lvl], mem["fresh"][lvl]
            # L0 succeeds in no scene, so every L3 success is a discordant pair: exact McNemar p = 2 * 0.5^n.
            assert f["success"] == 0
            p, nd = min(1.0, 2 * 0.5 ** fm["fresh_L3"]["success"]), (v["all"]["never_done"], f["all"]["never_done"])
        else:
            v, f = fm[f"validation_{lvl}"], fm[f"fresh_{lvl}"]
            p = f["paired"]["mcnemar_p"] if "paired" in f else None
            nd = (v["mechanisms"]["never_done"], f["mechanisms"]["never_done"])
        lines.append(f"{name} & {memo} & {v['success']} & {f['success']} & " + ("--" if p is None else "$%s$" % pval(p))
                     + f" & {nd[0]} / {nd[1]} " + r"\\")
    mech = lambda split, lvl, k: fm[f"{split}_{lvl}"]["mechanisms"][k]
    lines += [r"\midrule",
              r"\multicolumn{6}{@{}l@{}}{$^*$Exact McNemar, fresh set. $^\dagger$Base never stops (val./fresh).} \\",
              r"\multicolumn{6}{@{}l@{}}{L0: raw 21 inputs $\to$ raw 11 actions. Open-loop val.\ $R^2$ (wheels / torques):} \\",
              r"\multicolumn{6}{@{}l@{}}{\quad MLP %.2f / %.2f, linear %.2f / %.2f.} \\"
              % tuple(mem["L0_open_loop"][k][f"r2_mean_{g}"] for k in ("mlp", "linear") for g in ("wheels", "torques")),
              r"\multicolumn{6}{@{}l@{}}{Wheels restart after ``done'' (val./fresh): L2 %d / %d; with a latch impossible.} \\"
              % (mech("validation", "L2", "restarts"), mech("fresh", "L2", "restarts")),
              r"\multicolumn{6}{@{}l@{}}{Arm drift before ``done'', median (rad): L1 %.2f, L2 %.2f, L3 %.2f (validation).} \\"
              % tuple(mech("validation", l, "arm_drift_rad_median") for l in ("L1", "L2", "L3")),
              r"\bottomrule", r"\end{tabular}"]
    (out / "memory.tex").write_text("\n".join(lines) + "\n")


def table_ablation(an, mem, fk, results, out):
    grid_b = json.loads((results / "grid_B_park" / "grid.json").read_text())
    grid_v3 = json.loads((results / "grid_v3_bearing" / "grid.json").read_text())
    g = an["gravity_ablation"]
    hold = an["hold"]
    seeds = an["seeds"]
    u = fk["gru_trained_on_urdf_ee"]["validation"]["success"]
    rows = [
        (r"\textbf{B-FK}: learned FK, no $g(q)$", r"\textbf{%d}" % fk["validation"]["B_FK"]["success"], "final"),
        ("B-FK, GRU trained on URDF $e$", str(u), "train/run mismatch"),
        (r"\emph{URDF FK and $g(q)$ (rows below):}", None, None),
        ("B, direct torque output", str(mem["torque_validation"]["success"]), "no PD layer"),
        ("v3: hand-made bearing feature", f"{grid_v3['chosen']['success_1cm']}", "gains %g/%g/%g" % (
            grid_v3["chosen"]["kp"], grid_v3["chosen"]["kd"], grid_v3["chosen"]["lead"])),
        (r"\textbf{B}: learned parking pose", r"\textbf{%d}" % grid_b["chosen"]["success_1cm"], "gains %g/%g/%g" % (
            grid_b["chosen"]["kp"], grid_b["chosen"]["kd"], grid_b["chosen"]["lead"])),
        ("B, GRU seeds 1 / 2", " / ".join(str(seeds[s]["success"]) for s in ("1", "2")), "seed spread"),
        ("B, gains: other 11 combinations", "%d--%d" % (
            min(r["success_1cm"] for r in grid_b["table"]), sorted(r["success_1cm"] for r in grid_b["table"])[-2]),
         "gain sensitivity"),
        ("B without $g(q)$", str(g["kp200_kd30_lead0.2"]["without_gravity"]["success"]), "chosen gains"),
        ("B without $g(q)$, gains 50/30/0.1", str(g["kp50_kd30_lead0.1"]["without_gravity"]["success"]),
         "with $g(q)$: %d" % g["kp50_kd30_lead0.1"]["with_gravity"]["success"]),
    ]
    lines = [r"\begin{tabular}{@{}lcl@{}}", r"\toprule", r"Variant & Success / 208 & Note \\", r"\midrule"]
    for a, b, c in rows:
        lines.append(a + r" \\" if b is None else f"{a} & {b} & {c} " + r"\\")
    lines += [r"\midrule", r"\multicolumn{3}{@{}l}{\emph{Hold test, fresh set: held $<$1 cm for 2 s / successes}} \\"]
    for name, key in (("v3, Kd=10 (own gains)", "v3"), ("v3 with Kd=30", "diag_v3_kp100_kd30_lead0.1"),
                      ("B, Kd=30 (own gains)", "B"), ("B with Kd=10", "diag_B_kp100_kd10_lead0.1")):
        h = hold[key]
        lines.append(f"{name} & {h['held_within_1cm']} / {h['success_1cm']} & " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "ablation.tex").write_text("\n".join(lines) + "\n")


def table_learning(an, out):
    lc = sorted(an["learning_curve"].values(), key=lambda r: r["train_trajectories"])
    lines = [r"\begin{tabular}{@{}rcccc@{}}", r"\toprule",
             r"Traj. & Success & Val.\ loss wheel / arm / stop & Train (min) \\", r"\midrule"]
    for r in lc:
        lines.append(f"{r['train_trajectories']} & {r['success']}/208 & {r['validation_wheel_loss']:.3f} / "
                     f"{r['validation_arm_loss']:.3f} / {r['validation_stop_loss']:.3f} & "
                     f"{r['training_wall_s'] / 60:.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "learning.tex").write_text("\n".join(lines).replace("rcccc", "rccc") + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/final_run"))
    parser.add_argument("--artifacts", type=Path, help="Full run directory, only to (re)extract the per-step trace CSVs")
    parser.add_argument("--fk-rollouts", type=Path, default=Path("artifacts/fk_study"),
                        help="run_fk_study.sh roll-outs, only to (re)extract the per-step trace CSVs")
    parser.add_argument("--output", type=Path, default=Path("report"))
    args = parser.parse_args()
    suite = json.loads((args.results / "suite" / "summary.json").read_text())
    an = json.loads((args.results / "analysis" / "summary.json").read_text())
    figs, tabs = args.output / "figures", args.output / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    fk = json.loads((args.results / "fk_run" / "summary.json").read_text())  # summarize_fk.py
    fig_conditions(dict(conditions=fk_condition_rows(fk)), figs)
    fig_learning_curve(an, figs)
    fig_hold(hold_trace(args.results, args.artifacts, args.fk_rollouts), figs)
    table_main(suite, an, fk, args.results, tabs)
    mem = json.loads((args.results / "memory" / "summary.json").read_text())
    fig_memory_trace(memory_trace(args.results, args.artifacts, args.fk_rollouts), figs)
    fig_fit_vs_success(an, mem, fk, args.results, figs)
    table_memory(mem, fk, tabs)
    table_ablation(an, mem, fk, args.results, tabs)
    table_learning(an, tabs)
    print("wrote", sorted(str(p) for p in (*figs.iterdir(), *tabs.iterdir())))


if __name__ == "__main__":
    main()
