"""One-command demo and check of the submitted policy (no data collection or training needed).

GUI mode (default): plays the frozen final policy B-FK in the PyBullet window on a few fresh test scenes, one
after another, in real time (red sphere = target; press 'q' in the window to skip a scene), and prints
whether each scene succeeded. Batch mode (--no-gui): runs a whole scene set headless in parallel and
prints the success rate next to the number reported in the README.

Scene sets (all versioned in scenes/, so this runs on any machine with the RoboEnv environment):
  fresh        300 held-out test scenes collected after the models were frozen (main result)
  robustness   the first 60 expert-successful fresh scenes (use with --perturb, e.g. "--push-base 20")
  far_target, high_target, low_target, wide_start   out-of-distribution families (40 scenes each)

Controllers: B-FK (final method: no robot model; GRU trained and run on the learned forward kinematics,
no gravity term), B (the same design with the URDF forward kinematics and gravity term, for reference),
v3 (baseline with the URDF), expert (the template).
The learned controllers never import the expert module; each result records this
("expert_module_loaded": false).

Examples:
  python demo.py                                   # GUI, B-FK, 3 fresh scenes
  python demo.py --controller expert --scenes 2    # GUI, the expert on the same scenes
  python demo.py --no-gui --set fresh              # all 300 fresh scenes headless (about 2 min on 18 cores)
  python demo.py --no-gui --set high_target --controller expert
  python demo.py --no-gui --set robustness --perturb "--push-base 20"
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "final_run"
SETS = {"fresh": "fresh_test", "robustness": "robustness_60", "far_target": "ood/far_target",
        "high_target": "ood/high_target", "low_target": "ood/low_target", "wide_start": "ood/wide_start"}
# Reported 1 cm successes (README Section 7) for the unperturbed sets, to compare against.
REPORTED = {("B-FK", "fresh"): "288/300", ("B-FK", "far_target"): "40/40", ("B-FK", "high_target"): "17/40",
            ("B-FK", "low_target"): "3/40", ("B-FK", "wide_start"): "29/40", ("B", "fresh"): "296/300", ("v3", "fresh"): "289/300", ("expert", "fresh"): "293/300",
            ("B", "far_target"): "40/40", ("expert", "far_target"): "40/40",
            ("B", "high_target"): "40/40", ("expert", "high_target"): "15/40",
            ("B", "low_target"): "29/40", ("expert", "low_target"): "30/40",
            ("B", "wide_start"): "38/40", ("expert", "wide_start"): "39/40"}


def controller_args(name):
    if name == "expert":
        return ["--base", "expert", "--arm", "expert"]
    model = {"B-FK": "fk_run/gru_B_fk", "B": "gru_B_park", "v3": "gru_v3_bearing"}[name]
    grid = {"B-FK": "grid_B_park", "B": "grid_B_park", "v3": "grid_v3_bearing"}[name]
    chosen = json.loads((RESULTS / grid / "grid.json").read_text())["chosen"]
    learned = ["--fk-model", str(RESULTS / "fk_run" / "fk.npz"), "--gravity", "zero"] if name == "B-FK" else []
    return ["--base", "gru", "--arm", "gru", "--gru-model", str(RESULTS / model / "gru.npz"), *learned,
            "--kp", str(chosen["kp"]), "--kd", str(chosen["kd"]), "--lead", str(chosen["lead"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", choices=("B-FK", "B", "v3", "expert"), default="B-FK")
    parser.add_argument("--set", choices=sorted(SETS), default="fresh")
    parser.add_argument("--scenes", type=int, help="Number of scenes (default: 3 with GUI, all without)")
    parser.add_argument("--scene-id", help="Run one scene by id, e.g. scene_2026_0008")
    parser.add_argument("--speed", type=float, default=1.0, help="GUI playback speed (2 = twice real time)")
    parser.add_argument("--no-gui", action="store_true", help="Headless batch run with a success summary")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--perturb", default="", help='Extra rollout flags, e.g. "--obs-noise 0.01" or "--push-base 20"')
    parser.add_argument("--output", type=Path, help="Where to keep the roll-outs (default: a temporary folder)")
    args = parser.parse_args()

    scene_dir = ROOT / "scenes" / SETS[args.set]
    paths = sorted(scene_dir.glob("*.json"))
    if args.scene_id:
        paths = [p for p in paths if p.stem == args.scene_id]
        if not paths:
            parser.error(f"{args.scene_id} is not in scenes/{SETS[args.set]}")
    paths = paths[:args.scenes or (len(paths) if args.no_gui else 3)]
    out = args.output or Path(tempfile.mkdtemp(prefix="demo_"))
    extra = shlex.split(args.perturb)

    if args.no_gui:
        # staged_evaluate.py needs a split file only for its bookkeeping; --scenes-dir supplies the scenes.
        selected = Path(tempfile.mkdtemp(prefix="scenes_"))
        for p in paths:
            (selected / p.name).write_text(p.read_text())
        cmd = [sys.executable, str(ROOT / "staged_evaluate.py"), "--model-dir", str(RESULTS / "fresh_split"),
               "--scenes-dir", str(selected), *controller_args(args.controller), *extra,
               "--workers", str(args.workers), "--output", str(out / "eval")]
        print("running", len(paths), "scenes headless with", args.workers, "workers ...", flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        m = json.loads((out / "eval" / "metrics.json").read_text())
        rows = [json.loads(p.read_text()) for p in (out / "eval" / "rollouts").glob("*/result.json")]
        print(f"\n{args.controller} on {args.set} ({len(paths)} scenes{', ' + args.perturb if extra else ''}):"
              f" {m['success_1cm']}/{m['scenes']} reached the target within 1 cm")
        if not extra and len(paths) == len(list(scene_dir.glob('*.json'))) and (args.controller, args.set) in REPORTED:
            print(f"reported in README: {REPORTED[(args.controller, args.set)]}")
        if args.controller != "expert":
            print("expert module loaded in any learned run:", any(r["expert_module_loaded"] for r in rows))
        print("roll-outs:", out / "eval")
        return

    ok = 0
    for p in paths:
        scene_id = p.stem
        print(f"{scene_id}: playing {args.controller} ...", flush=True)
        cmd = [sys.executable, str(ROOT / "staged_rollout.py"), "--scene", str(p), "--output", str(out / scene_id),
               *controller_args(args.controller), *extra, "--gui", "--gui-speed", str(args.speed),
               "--gui-label", f"{args.controller} | {scene_id}"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r = json.loads((out / scene_id / "result.json").read_text())
        ok += r["success_1cm"]
        print(f"  -> {r['stop_reason']}, t = {r['elapsed_s']:.1f} s, EE error {100 * r['final_error_m']:.2f} cm"
              + ("" if args.controller == "expert" else f", expert module loaded: {r['expert_module_loaded']}"))
    print(f"\n{ok}/{len(paths)} scenes succeeded. Roll-outs: {out}")


if __name__ == "__main__":
    main()
