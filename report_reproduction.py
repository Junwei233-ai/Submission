"""Compare the local retraining with the numbers of the development runs (teammate repo, branch trying @ 61bc8f8).
Writes <run-dir>/reproduction.json and reproduction.md. Reproduction criterion (fixed in advance):
successes within 3 scenes of the reported count on the same split."""
import argparse
import json
from pathlib import Path

REPORTED = {  # README.md section 5 (teammate's runs)
    "v3 validation, own grid gains": ("grid_v3_bearing", 203, 208),
    "B validation, own grid gains": ("grid_B_park", 202, 208),
    "v3 test, own grid gains": ("oldtest/v3_own_gains", 205, 210),
    "v3 test, 100/30/0.1": ("oldtest/v3_kp100_kd30_lead0.1", 204, 210),
    "B test, own grid gains": ("oldtest/B_own_gains", 206, 210),
    "expert test": ("oldtest/expert", None, 210),
}
TOLERANCE = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for label, (folder, reported, n) in REPORTED.items():
        path = args.run_dir / folder
        if folder.startswith("grid_"):
            grid = json.loads((path / "grid.json").read_text())
            chosen = grid["chosen"]
            local, gains = chosen["success_1cm"], f"{chosen['kp']:g}/{chosen['kd']:g}/{chosen['lead']:g}"
            planned = chosen["scenes"]
        else:
            metrics = json.loads((path / "metrics.json").read_text())
            config = json.loads((path / "config.json").read_text())
            local, planned = metrics["success_1cm"], metrics["scenes"]
            gains = (f"{config['kp']:g}/{config['kd']:g}/{config['lead']:g}"
                     if config.get("kp") is not None else "expert")
        rows.append(dict(item=label, gains=gains, local=local, planned=planned, reported=reported,
                         reproduced=None if reported is None else abs(local - reported) <= TOLERANCE))
    (args.run_dir / "reproduction.json").write_text(json.dumps(dict(tolerance=TOLERANCE, rows=rows), indent=2) + "\n")
    lines = ["| item | gains Kp/Kd/lead | local | reported | reproduced (±3) |", "|---|---|---|---|---|"]
    for r in rows:
        mark = "-" if r["reproduced"] is None else ("yes" if r["reproduced"] else "**no**")
        lines.append(f"| {r['item']} | {r['gains']} | {r['local']}/{r['planned']} | "
                     f"{'-' if r['reported'] is None else r['reported']} | {mark} |")
    (args.run_dir / "reproduction.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
