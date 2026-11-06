"""Interpretability check (after freezing; nothing is trained): what did B's arm matrix M learn?

B's arm velocity is v = M e (M: 7x3 from the GRU, e: EE error in the base frame). We compare M with
the arm's position Jacobian J (3x7, base frame, from pinocchio at the logged state), which the policy
never sees:
  alignment   cos(J v, e): does the commanded joint velocity move the EE straight at the target?
  task gain   |J v| / |e| (1/s): the closed-loop rate at which the EE error shrinks
  null share  |(I - J+ J) v| / |v|: fraction of the joint motion that does not move the EE (posture)
  cos(M, J+), cos(M, J^T): matrix cosine with the pseudo-inverse and the transpose (two classic
              inverse-kinematics laws)
For B, M is recomputed by replaying the frozen GRU on the logged states of its fresh-test roll-outs
(--rollouts), at every GRU update after the latch while |e| >= 1 cm. For the expert the same alignment,
gain and null share are computed from the public arm label v (central difference of the recorded joint
angles, as in training) on validation demonstrations (--model-dir split), after the public stop time.
Writes results/final_run/analysis/arm_matrix.json.

Usage: python analyze_arm_matrix.py --rollouts artifacts/final_run/suite/fresh_B/rollouts \
           [--model-dir results/final_run/base] [--workers 18]
"""
import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "robotnik_arm_task"
RESULTS = ROOT / "results" / "final_run"
HALF = 10  # central-difference half window (steps), as in the training labels
_kin = None


def kinematics():
    """Pinocchio model of the robot (same construction as staged_rollout.py), headless."""
    import pinocchio as pin
    from simulation_and_control import pb, PinWrapper
    config = json.loads((TASK_DIR / "configs/robotnik_arm.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "robotnik_arm.json"
        path.write_text(json.dumps(config))
        sim = pb.SimInterface(str(path), conf_file_path_ext=str(TASK_DIR), use_gui=False)
        dyn = PinWrapper(str(path), "pybullet", np.expand_dims(np.array(sim.getNameActiveJoints()), 0),
                         ["pybullet"], False, 0, str(TASK_DIR))
    model, data = dyn.pin_model, dyn.pin_data
    ee_id = model.getFrameId("robot_j2s7s300_end_effector")
    arm_v = [model.joints[model.getJointId(f"robot_j2s7s300_joint_{i}")].idx_v for i in range(1, 8)]

    def at(x):
        """EE position (world) and arm position Jacobian rotated into the base (yaw) frame."""
        yaw, motors = x[3], np.zeros(17)
        motors[4:11], motors[11:] = x[4:11], 0.1
        q = dyn.ReoderJoints2PinVec(np.concatenate((x[:3], [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)], motors)), "pos")
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        jac = pin.getFrameJacobian(model, data, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3][:, arm_v]
        return data.oMf[ee_id].translation.copy(), rot @ jac
    return at


def measures(v, e, jac, matrix=None):
    pinv = np.linalg.pinv(jac)
    dx = jac @ v
    row = [dx @ e / (np.linalg.norm(dx) * np.linalg.norm(e) + 1e-12), np.linalg.norm(dx) / np.linalg.norm(e),
           np.linalg.norm(v - pinv @ dx) / (np.linalg.norm(v) + 1e-12)]
    if matrix is not None:
        cos = lambda a, b: float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
        row += [cos(matrix, pinv), cos(matrix, jac.T)]
    return row


def replay(folder):
    """B: replay the frozen GRU on the logged states of one roll-out; one row per post-latch update."""
    global _kin
    from staged_policy import GRUPolicy, gru_park_features, park_pose
    _kin = _kin or kinematics()
    x = np.load(folder / "rollout.npz")["inputs"]
    gru = GRUPolicy(RESULTS / "gru_B_park" / "gru.npz")
    gru.reset(x[0, 4:11])
    park = park_pose(gru.park_net, x[0, 18:21])
    rows = []
    for k in range(0, len(x), gru.period):
        ee, jac = _kin(x[k])
        f = gru_park_features(x[k, :3], x[k, 3], x[k, 4:11], ee, x[k, 18:21], *park)
        gru._update(f)
        e = f[8:11]
        if gru.latched and np.linalg.norm(e) >= 0.01:
            rows.append(measures(gru.matrix @ e, e, jac, gru.matrix))
    return rows


def demo(path):
    """Expert: public velocity label at demonstration states after the public stop time."""
    global _kin
    from staged_data import load_trajectory
    _kin = _kin or kinematics()
    t = load_trajectory(path)
    x, rows = t["inputs"], []
    for k in range(max(t["stop"], HALF), len(x) - HALF, 10):
        ee, jac = _kin(x[k])
        c, s = np.cos(x[k, 3]), np.sin(x[k, 3])
        w = x[k, 18:21] - ee
        e = np.array([c * w[0] + s * w[1], -s * w[0] + c * w[1], w[2]])
        if np.linalg.norm(e) >= 0.01:
            rows.append(measures((x[k + HALF, 4:11] - x[k - HALF, 4:11]) / (2 * HALF * 1e-3), e, jac))
    return rows


def summary(rows, names):
    r = np.asarray(rows)
    return dict(samples=len(r), **{n: dict(zip(("p5", "p25", "median", "p75", "p95"),
                                              np.percentile(r[:, i], [5, 25, 50, 75, 95]).round(4).tolist()))
                                   for i, n in enumerate(names)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, required=True, help="B's fresh-test roll-outs (rollouts/ folder)")
    parser.add_argument("--model-dir", type=Path, default=RESULTS / "base", help="split.json with the demonstrations")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--output", type=Path, default=RESULTS / "analysis" / "arm_matrix.json")
    args = parser.parse_args()
    split = json.loads((args.model_dir / "split.json").read_text())
    demos = [Path(split["source"]) / n for n in split["splits"]["validation"]]
    with Pool(args.workers) as pool:
        b = [r for rows in pool.map(replay, sorted(p for p in args.rollouts.iterdir() if p.is_dir())) for r in rows]
        expert = [r for rows in pool.map(demo, demos) for r in rows]
    out = dict(B=summary(b, ["alignment", "task_gain_per_s", "null_share", "cos_M_pinv", "cos_M_transpose"]),
               expert=summary(expert, ["alignment", "task_gain_per_s", "null_share"]),
               note="B: frozen GRU replayed on its fresh-test roll-outs, post-latch updates with |e| >= 1 cm; "
                    "expert: public velocity labels on validation demonstrations after the stop time")
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
