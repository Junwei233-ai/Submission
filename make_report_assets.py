"""Build the report's figures (PDF) and LaTeX tables from the versioned result summaries.

Every number in report/tables/*.tex and report/figures/*.pdf is read from results/final_run/
(suite/summary.json, analysis/summary.json, memory/summary.json, grid_*/grid.json, gru_*/gru.json), so the report cannot
drift from the code. The hold-trace figure needs per-step errors; they are extracted once from a full
run (--artifacts) into results/final_run/analysis/hold_trace.csv, which is versioned.

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
STYLE = {"B": dict(color="#2a78d6", marker="o", label="B (ours)"),
         "expert": dict(color="#eb6834", marker="s", label="Expert"),
         "v3": dict(color="#1baf7a", marker="^", label="v3 (baseline)")}
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e6ea"
COL_W = 3.5  # IEEE single column, inches
TRACE_SCENE = "scene_2026_0008"

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
def fig_conditions(suite, out):
    """Robustness and OOD: success rate with Wilson CI, B vs expert, one row per condition."""
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
    fig.savefig(out / "conditions.pdf")
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
    fig.savefig(out / "learning_curve.pdf")
    plt.close(fig)


def hold_trace(results, artifacts):
    path = results / "analysis" / "hold_trace.csv"
    if artifacts and (artifacts / "analysis").is_dir():
        cols = {}
        for c in ("B", "v3", "expert"):
            folder = artifacts / "analysis" / f"hold_{c}" / "rollouts" / TRACE_SCENE
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
    fig.savefig(out / "hold_trace.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- tables
def table_main(suite, an, out):
    fresh = next(r for r in suite["conditions"] if r["condition"] == "fresh")
    hold = an["hold"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r" & \textbf{B (ours)} & v3 & Expert \\", r"\midrule"]
    lines.append("1 cm success / 300 & " + " & ".join(
        (r"\textbf{%d}" if c == "B" else "%d") % fresh[c]["success"] for c in ("B", "v3", "expert")) + r" \\")
    lines.append(r"\quad rate (\%) & " + " & ".join(pct(fresh[c]["rate"]) for c in ("B", "v3", "expert")) + r" \\")
    lines.append(r"\quad 95\% Wilson CI & " + " & ".join(ci(fresh[c]["wilson95"]) for c in ("B", "v3", "expert")) + r" \\")
    lines.append(r"Held $<$1 cm for 2 s & " + " & ".join(str(hold[c]["held_within_1cm"]) for c in ("B", "v3", "expert")) + r" \\")
    lines.append(r"Worst err.\ last 1 s, median (mm) & " + " & ".join(
        f"{1000 * hold[c]['last_second_max_median_m']:.1f}" for c in ("B", "v3", "expert")) + r" \\")
    lines.append(r"Mean time to success (s) & " + " & ".join(
        f"{fresh[c]['mean_success_time_s']:.2f}" for c in ("B", "v3", "expert")) + r" \\")
    lines.append(r"Compute per 1 kHz step ($\mu$s) & " + " & ".join(
        f"{fresh[c]['policy_time_mean_us']:.0f}" for c in ("B", "v3", "expert")) + r" \\")
    m = suite["models"]
    lines.append(r"Parameters & " + f"{m['B']['gru_parameters'] + m['B']['park_mlp_parameters']:,} & "
                 + f"{m['v3']['gru_parameters']:,} & -- " + r"\\")
    lines.append(r"Model file (KiB) & " + f"{m['B']['file_kib']:.0f} & {m['v3']['file_kib']:.0f} & -- " + r"\\")
    lines += [r"\midrule",
              r"\multicolumn{4}{@{}l@{}}{Paired exact McNemar (discordant pairs): B vs expert %d/%d, $p=%s$;} \\"
              % (fresh["B_vs_expert"]["only_first"], fresh["B_vs_expert"]["only_second"], pval(fresh["B_vs_expert"]["mcnemar_p"])),
              r"\multicolumn{4}{@{}l@{}}{B vs v3 %d/%d, $p=%s$; held $<$1 cm: B vs v3 %d/%d, $p=%s$.} \\"
              % (fresh["B_vs_v3"]["only_first"], fresh["B_vs_v3"]["only_second"], pval(fresh["B_vs_v3"]["mcnemar_p"]),
                 hold["B_vs_v3_held_1cm"]["only_first"], hold["B_vs_v3_held_1cm"]["only_second"],
                 pval(hold["B_vs_v3_held_1cm"]["mcnemar_p"])),
              r"\bottomrule", r"\end{tabular}"]
    (out / "main.tex").write_text("\n".join(lines) + "\n")


def table_memory(mem, out):
    """Memory study: three levels of stage memory, same data, split and scenes (memory/summary.json)."""
    names = {"L0mlp": (r"L0 plain MLP regressor", "none"),
             "L1": (r"L1 four single-frame MLPs", "latch only"),
             "L2": (r"L2 GRU, no stop head", "hidden state only"),
             "L3": (r"\textbf{L3 = B}: GRU + stop head", "hidden state + latch")}
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Policy & Stage memory & Val./208 & Fresh/300 & $p$ vs B$^*$ & Never done$^\dagger$ \\", r"\midrule"]
    for lvl, (name, memo) in names.items():
        v, f = mem["validation"][lvl], mem["fresh"][lvl]
        p = "--" if f["vs_B"] is None else "$%s$" % pval(f["vs_B"]["mcnemar_p"])
        lines.append(f"{name} & {memo} & {v['success']} & {f['success']} & {p} & "
                     f"{v['all']['never_done']} / {f['all']['never_done']} " + r"\\")
    lines += [r"\midrule",
              r"\multicolumn{6}{@{}l@{}}{$^*$Exact McNemar, fresh set. $^\dagger$Base never stops (val./fresh).} \\",
              r"\multicolumn{6}{@{}l@{}}{L0: raw 21 inputs $\to$ raw 11 actions. Open-loop val.\ $R^2$ (wheels / torques):} \\",
              r"\multicolumn{6}{@{}l@{}}{\quad MLP %.2f / %.2f, linear %.2f / %.2f.} \\"
              % tuple(mem["L0_open_loop"][k][f"r2_mean_{g}"] for k in ("mlp", "linear") for g in ("wheels", "torques")),
              r"\multicolumn{6}{@{}l@{}}{Wheels restart after ``done'' (val./fresh): L2 %d / %d; with a latch impossible.} \\"
              % (mem["validation"]["L2"]["all"]["restarts"], mem["fresh"]["L2"]["all"]["restarts"]),
              r"\multicolumn{6}{@{}l@{}}{Arm drift before ``done'', median (rad): L1 %.2f, L2 %.2f, L3 %.2f (validation).} \\"
              % tuple(mem["validation"][l]["all"]["arm_drift_rad_median"] for l in ("L1", "L2", "L3")),
              r"\bottomrule", r"\end{tabular}"]
    (out / "memory.tex").write_text("\n".join(lines) + "\n")


def table_ablation(an, mem, results, out):
    grid_b = json.loads((results / "grid_B_park" / "grid.json").read_text())
    grid_v3 = json.loads((results / "grid_v3_bearing" / "grid.json").read_text())
    g = an["gravity_ablation"]
    hold = an["hold"]
    seeds = an["seeds"]
    rows = [
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
    parser.add_argument("--artifacts", type=Path, help="Full run directory, only to (re)extract hold_trace.csv")
    parser.add_argument("--output", type=Path, default=Path("report"))
    args = parser.parse_args()
    suite = json.loads((args.results / "suite" / "summary.json").read_text())
    an = json.loads((args.results / "analysis" / "summary.json").read_text())
    figs, tabs = args.output / "figures", args.output / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    fig_conditions(suite, figs)
    fig_learning_curve(an, figs)
    fig_hold(hold_trace(args.results, args.artifacts), figs)
    table_main(suite, an, tabs)
    mem = json.loads((args.results / "memory" / "summary.json").read_text())
    table_memory(mem, tabs)
    table_ablation(an, mem, args.results, tabs)
    table_learning(an, tabs)
    print("wrote", sorted(str(p) for p in (*figs.iterdir(), *tabs.iterdir())))


if __name__ == "__main__":
    main()
