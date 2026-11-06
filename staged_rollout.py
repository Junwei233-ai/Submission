"""Run one scene with any base/arm combination: ours (staged MLP) or the expert.

Simulation setup, step order and success test follow robotnik_arm_task/robotnik_arm_collect.py:
success means the active base has finished, the EE error is below 1 cm and more than 8 s have
passed. The arm starts reaching when the active base reports done (our latch or the expert's).
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "robotnik_arm_task"
sys.path.insert(0, str(TASK_DIR))

from staged_policy import DirectPolicy, GRUPolicy, LearnedFK, StagedArm, StagedBase  # noqa: E402

FINGER_ANGLES = np.full(6, 0.1)
BASE_NAV_TIMEOUT, ARM_REACH_LIMIT = 25.0, 0.85
STANDOFF_Z_POINTS, STANDOFF_POINTS = [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]
REACH_TOLERANCE, MIN_TIME, MAX_STEPS = 0.01, 8.0, 40000
WHEEL_LIMIT = (0.25 + 1.0 * 0.66 / 2) / 0.11
TORQUE_LIMIT = json.loads((ROOT / "torque_limit.json").read_text())["torque_limit"]  # compute_torque_limit.py
GRU_WHEELS_STOPPED = 0.05  # rad/s; evaluation-only 'base done' threshold for the GRU policy


class Viewer:
    """--gui only (as in the template): PyBullet window paced to real time, a red marker on the target,
    a status line, and 'q' to quit. It reads nothing the controllers do not already see."""

    def __init__(self, client, target, base_pos, label, speed):
        self.client, self.speed, self.label = client, speed, label
        self.start, self.text, self.next_text = time.perf_counter(), -1, 0.0
        shape = client.createVisualShape(client.GEOM_SPHERE, radius=0.025, rgbaColor=[0.9, 0.1, 0.1, 0.9])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=list(target))
        mid = (np.asarray(base_pos[:2]) + np.asarray(target[:2])) / 2
        client.resetDebugVisualizerCamera(cameraDistance=3.2, cameraYaw=45, cameraPitch=-35,
                                          cameraTargetPosition=[mid[0], mid[1], 0.4])
        self.target = np.asarray(target, dtype=float)

    def _status(self, line, color=(0.1, 0.1, 0.1)):
        kwargs = dict(replaceItemUniqueId=self.text) if self.text >= 0 else {}
        self.text = self.client.addUserDebugText(line, (self.target + [0, 0, 0.35]).tolist(), textColorRGB=list(color),
                                                 textSize=1.4, **kwargs)

    def update(self, t, error, done, success_t):
        """Pace to real time, refresh the status line; True if the user pressed 'q'."""
        if t >= self.next_text:
            stage = "reaching" if done else "driving"
            self._status(f"{self.label}  t={t:4.1f}s  {stage}  EE error {100 * error:5.1f} cm"
                         + ("  SUCCESS" if success_t is not None else ""))
            self.next_text = t + 0.25
        lag = t / self.speed - (time.perf_counter() - self.start)
        if lag > 0:
            time.sleep(lag)
        keys = self.client.getKeyboardEvents()
        return ord("q") in keys and keys[ord("q")] & self.client.KEY_WAS_TRIGGERED

    def finish(self, reason, error, t, hold_s=3.0):
        ok = reason == "success"
        self._status(f"{self.label}  {'SUCCESS' if ok else reason.upper()} at t={t:.1f}s, EE error {100 * error:.1f} cm",
                     (0.0, 0.6, 0.0) if ok else (0.8, 0.0, 0.0))
        time.sleep(hold_s)


def run(args):
    import pinocchio as pin
    from simulation_and_control import pb, MotorCommands, PinWrapper
    from sim_utils import hold_gripper_fingers, quaternion_yaw
    # The expert controller is imported only when the expert drives the base or the arm
    # (baselines). Learned-only runs (--base gru/ours --arm gru/ours) never load it.
    uses_expert = "expert" in (args.base, args.arm)
    if uses_expert:
        from robotnik_arm_control import (ArmControllerParams, BaseControllerParams, BaseNavState,
                                          arm_controller, base_controller)

    scene = json.loads(args.scene.read_text())
    target = np.asarray(scene["target"], dtype=float)
    if "seed" in scene:
        np.random.seed(scene["seed"])
    config = json.loads((TASK_DIR / "configs/robotnik_arm.json").read_text())
    robot = config["robot_pybullet"]
    robot["init_link_base_position"][0] = scene["base_pos"]
    yaw0 = scene["base_bearing"]
    robot["init_link_base_orientation"][0] = [0, 0, np.sin(yaw0 / 2), np.cos(yaw0 / 2)]
    robot["init_motor_angles"][0][4:11] = scene["arm_q"]
    robot["init_motor_vel"][0][4:11] = scene["arm_qd"]
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "robotnik_arm.json"
        config_path.write_text(json.dumps(config))
        sim = pb.SimInterface(str(config_path), conf_file_path_ext=str(TASK_DIR), use_gui=args.gui)
        client, rid = sim.pybullet_client, sim.bot[0].bot_pybullet
        sim.SetFloorFriction(1.0)
        for j in range(client.getNumJoints(rid)):
            if "wheel" in client.getJointInfo(rid, j)[1].decode():
                client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                      anisotropicFriction=[1.0, 0.01, 1.0])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        hold_gripper_fingers(sim, client, rid, FINGER_ANGLES)
        dyn = PinWrapper(str(config_path), "pybullet", np.expand_dims(np.array(sim.getNameActiveJoints()), 0),
                         ["pybullet"], False, 0, str(TASK_DIR))
    try:
        ee_id = dyn.pin_model.getFrameId("robot_j2s7s300_end_effector")
        ee_link = next(j for j in range(client.getNumJoints(rid))
                       if client.getJointInfo(rid, j)[12].decode() == "robot_j2s7s300_end_effector")
        shoulder_id = dyn.pin_model.getFrameId("robot_j2s7s300_link_2")
        lower, upper = (np.asarray(v)[4:11] for v in sim.GetBotJointsLimit())
        dt = sim.GetTimeStep()
        q_init = np.asarray(sim.GetMotorAngles(0))[4:11].copy()

        our_base = StagedBase(args.base_model, args.latch_steps) if args.base == "ours" else None
        our_arm = StagedArm(args.arm_model, args.kp, args.kd, args.lead) if args.arm == "ours" else None
        if our_arm is not None:
            our_arm.reset(q_init)
        learned_fk = LearnedFK(args.fk_model) if args.fk_model else None
        gru = GRUPolicy(args.gru_model, args.kp, args.kd, args.lead) if args.base == "gru" else None
        if gru is not None:
            gru.reset(q_init)
        direct = DirectPolicy(args.direct_model) if args.base == "direct" else None
        if uses_expert:
            nav_state, base_params = BaseNavState(), BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
            arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        # Template parking rule: drives the expert base, and for learned runs it is used only
        # afterwards to report how far the learned stop pose is from it (never fed to a policy).
        bearing = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
        standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        rule_goal = target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
        q_des = q_init.copy()

        limits = np.array([WHEEL_LIMIT] * 4 + [TORQUE_LIMIT] * 7)
        cmd = MotorCommands()
        log = dict(inputs=[], actions=[], ee_error=[], done=[], ee_pos=[])
        done_time = done_pose = first_5cm = None
        clipped, reason, t = 0, "step_limit", 0.0
        success_t = success_error = hold_max = None  # --hold-time: keep simulating after success
        # Robustness perturbations (all off by default; identical to the original rollout then).
        # They act on whichever controller runs (learned or expert), so comparisons are paired.
        rng = np.random.default_rng([args.perturb_seed, zlib.crc32(scene["id"].encode())])
        noisy = args.obs_noise > 0
        push_dir = rng.normal(size=2)
        push_dir = np.append(push_dir / np.linalg.norm(push_dir), 0.0)
        policy_time = []
        gui = Viewer(client, target, base_pos=scene["base_pos"], label=args.gui_label, speed=args.gui_speed) if args.gui else None
        for step in range(MAX_STEPS):
            base_pos = np.asarray(sim.GetBasePosition())
            base_ori = np.asarray(sim.GetBaseOrientation())
            yaw = quaternion_yaw(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes, qd_mes = np.asarray(sim.GetMotorAngles(0)), np.asarray(sim.GetMotorVelocities(0))
            q_pin = dyn.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
            pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
            pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
            ee = dyn.pin_data.oMf[ee_id].translation.copy()
            error = float(np.linalg.norm(target - ee))  # scoring always uses the true state
            if noisy:
                # Sensor noise: std obs_noise (m) on base xy, (rad) on yaw and arm angles,
                # 10x on arm velocities. The controller sees only the noisy values and its own
                # FK of them; the true state is used only for scoring above.
                s = args.obs_noise
                base_pos = base_pos + np.array([*rng.normal(0, s, 2), 0.0])
                yaw = yaw + rng.normal(0, s)
                q_mes, qd_mes = q_mes.copy(), qd_mes.copy()
                q_mes[4:11] += rng.normal(0, s, 7)
                qd_mes[4:11] += rng.normal(0, 10 * s, 7)
                base_ori = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
                q_pin = dyn.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
                pin.forwardKinematics(dyn.pin_model, dyn.pin_data, q_pin)
                pin.updateFramePlacements(dyn.pin_model, dyn.pin_data)
                ee = dyn.pin_data.oMf[ee_id].translation.copy()
            started = time.perf_counter()

            if direct is not None:
                # Plain BC baseline: the regressor's raw outputs are the commands. "Done" for the
                # success test uses the same evaluation-only wheel threshold as the GRU without stop head.
                wheels, torques = direct.act(np.concatenate((base_pos, [yaw], q_mes[4:11], qd_mes[4:11], target)))
                done = bool(np.max(np.abs(wheels)) < GRU_WHEELS_STOPPED)
            elif gru is not None:
                # Ablation: --gravity zero removes the analytic model prior from the torque layer.
                gravity = (np.zeros(7) if args.gravity == "zero"
                           else pin.computeGeneralizedGravity(dyn.pin_model, dyn.pin_data, q_pin)[6:13])
                # With --fk-model the policy's EE position is learned; the URDF value above is for scoring only.
                policy_ee = ee if learned_fk is None else learned_fk(q_mes[4:11], base_pos, yaw)
                wheels, torques = gru.step(base_pos, yaw, q_mes[4:11], qd_mes[4:11], policy_ee, target, gravity, dt)
                # With a stop head "base done" is the GRU's own latch. Without one (evaluation only)
                # it means the GRU is no longer commanding the wheels.
                done = gru.latched if gru.stop_head else bool(np.max(np.abs(wheels)) < GRU_WHEELS_STOPPED)
            elif our_base is not None:
                # Optional hand-written rule borrowed from the task script's BASE_NAV_TIMEOUT.
                if args.nav_timeout is not None and t > args.nav_timeout:
                    our_base.done = True
                wheels, done = our_base.step(base_pos[:2], yaw, target)
            else:
                wheels = base_controller(nav_state, base_pos, yaw, rule_goal, bearing, t, base_params)
                done = nav_state.done
            if done and done_time is None:
                done_time, done_pose = t, [*base_pos[:2], yaw]
            if gru is not None or direct is not None:
                pass
            elif our_arm is not None:
                gravity = (np.zeros(7) if args.gravity == "zero"
                           else pin.computeGeneralizedGravity(dyn.pin_model, dyn.pin_data, q_pin)[6:13])
                policy_ee = ee if learned_fk is None else learned_fk(q_mes[4:11], base_pos, yaw)
                torques = our_arm.step(q_mes[4:11], qd_mes[4:11], policy_ee, target, yaw, gravity, done, dt)
            else:
                torques, q_des, _ = arm_controller(dyn, q_pin, q_mes[4:11], qd_mes[4:11], q_des, q_init,
                                                   target, ee, ee_id, shoulder_id, lower, upper, dt,
                                                   arm_params, reach_active=done)
            policy_time.append(time.perf_counter() - started)
            # Actuator weakness: the arm motors deliver only torque_scale of the command.
            raw = np.concatenate((wheels, args.torque_scale * np.asarray(torques)))
            if not np.isfinite(raw).all():
                reason = "nonfinite"
                break
            action = np.clip(raw, -limits, limits)
            clipped += int(np.any(action != raw))
            log["inputs"].append(np.concatenate((base_pos, [yaw], q_mes[4:11], qd_mes[4:11], target)))
            log["actions"].append(action)
            log["ee_error"].append(error)
            log["done"].append(done)
            log["ee_pos"].append(ee)  # labels for staged_fk_train.py; no policy reads this log
            if first_5cm is None and done and error < 0.05 and t > MIN_TIME:
                first_5cm = t
            # Same test as check_stop: on the pre-step state, after the minimum time.
            if success_t is None and done and error < REACH_TOLERANCE and t > MIN_TIME:
                reason, success_t, success_error, hold_max = "success", t, error, error
                if args.hold_time <= 0:
                    break
            if success_t is not None:
                # Hold check (evaluation only): the controller keeps running unchanged; we record
                # the worst EE error over the next hold_time seconds.
                hold_max = max(hold_max, error)
                if t >= success_t + args.hold_time:
                    break
            cmd.SetControlCmd(np.concatenate((action, np.zeros(6))), ["velocity"] * 4 + ["torque"] * 13)
            # External pushes (0.2 s, random horizontal direction per scene): on the base at
            # push_time, and on the end-effector 1 s after the base has finished.
            if args.push_base > 0 and args.push_time <= t < args.push_time + 0.2:
                client.applyExternalForce(rid, -1, (args.push_base * push_dir).tolist(),
                                          sim.GetBasePosition(), client.WORLD_FRAME)
            if (args.push_ee > 0 and done_time is not None
                    and done_time + 1.0 <= t < done_time + 1.2):
                ee_world = client.getLinkState(rid, ee_link)[0]
                client.applyExternalForce(rid, ee_link, (args.push_ee * push_dir).tolist(),
                                          ee_world, client.WORLD_FRAME)
            sim.Step(cmd, "torque")
            t += dt
            if gui is not None and gui.update(t, error, done, success_t):
                reason = "quit_by_user"
                break
        if gui is not None:
            gui.finish(reason, error, t)
    finally:
        client.disconnect()

    errors = np.asarray(log["ee_error"])
    expert_loaded = "robotnik_arm_control" in sys.modules
    if expert_loaded and not uses_expert:
        raise RuntimeError("Expert controller module was loaded during a learned-only rollout")
    result = dict(id=scene["id"], base=args.base, arm=args.arm, stop_reason=reason,
                  expert_module_loaded=expert_loaded,
                  perturbation=dict(obs_noise=args.obs_noise, torque_scale=args.torque_scale,
                                    push_base_n=args.push_base, push_time_s=args.push_time,
                                    push_ee_n=args.push_ee, perturb_seed=args.perturb_seed),
                  gravity=args.gravity,
                  policy_time_mean_us=float(np.mean(policy_time) * 1e6) if policy_time else None,
                  policy_time_p99_us=float(np.quantile(policy_time, 0.99) * 1e6) if policy_time else None,
                  success_1cm=reason == "success", success_5cm=first_5cm is not None,
                  final_error_m=float(errors[-1] if success_t is None else success_error),
                  min_error_m=float(errors.min()), elapsed_s=t if success_t is None else success_t,
                  hold=(None if success_t is None or args.hold_time <= 0 else dict(
                      requested_s=args.hold_time, completed_s=t - success_t, max_error_m=hold_max,
                      end_error_m=float(errors[-1]), within_1cm=hold_max < REACH_TOLERANCE,
                      within_2cm=hold_max < 0.02)),
                  steps=len(errors), clipped_step_fraction=clipped / max(len(errors), 1),
                  base_done_s=done_time,
                  base_done_offset_from_rule_m=(float(np.linalg.norm(np.asarray(done_pose[:2]) - rule_goal))
                                                if done_pose else None),
                  base_done_yaw_offset_rad=(float(np.angle(np.exp(1j * (done_pose[2] - bearing))))
                                            if done_pose else None))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "rollout.npz", **{k: np.asarray(v) for k, v in log.items()},
                        scenario=json.dumps(scene))
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="JSON scene (collect_data.py format)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", choices=("ours", "expert", "gru", "direct"), required=True)
    parser.add_argument("--arm", choices=("ours", "expert", "gru", "direct"), required=True)
    parser.add_argument("--fk-model", type=Path,
                        help="staged_fk_train.py network: the policy's EE position instead of the URDF (needs --gravity zero)")
    parser.add_argument("--direct-model", type=Path, help="staged_direct_train.py linear.npz or mlp.npz (--base direct --arm direct)")
    parser.add_argument("--gru-model", type=Path, help="staged_gru_train.py policy (needed for --base gru --arm gru)")
    parser.add_argument("--base-model", type=Path, help="staged_train.py policy.npz (needed for --base ours)")
    parser.add_argument("--arm-model", type=Path, help="arm policy .npz (needed for --arm ours)")
    parser.add_argument("--latch-steps", type=int, help="Override the stored stop latch length")
    parser.add_argument("--kp", type=float, help="Override the arm PD position gain")
    parser.add_argument("--kd", type=float, help="Override the arm PD velocity gain")
    parser.add_argument("--lead", type=float, help="Override the reference lead limit (rad)")
    parser.add_argument("--nav-timeout", type=float, help="Force our base latch after this many seconds")
    # Robustness tests (defaults = no perturbation)
    parser.add_argument("--obs-noise", type=float, default=0.0,
                        help="Gaussian sensor noise std: m on base xy, rad on yaw and arm angles, x10 on arm velocities")
    parser.add_argument("--torque-scale", type=float, default=1.0, help="Arm actuator gain (e.g. 0.8 = 20%% weaker)")
    parser.add_argument("--push-base", type=float, default=0.0, help="Horizontal push on the base (N) for 0.2 s")
    parser.add_argument("--push-time", type=float, default=2.0, help="Start time of the base push (s)")
    parser.add_argument("--push-ee", type=float, default=0.0, help="Push on the end-effector (N), 0.2 s, 1 s after base done")
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--hold-time", type=float, default=0.0,
                        help="After success keep simulating this long and record the worst EE error (s)")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet window, paced to real time")
    parser.add_argument("--gui-speed", type=float, default=1.0, help="--gui playback speed (2 = twice real time)")
    parser.add_argument("--gui-label", default="", help="--gui status-line prefix (e.g. the controller name)")
    parser.add_argument("--gravity", choices=("model", "zero"), default="model",
                        help="Arm torque layer: pinocchio gravity compensation (model) or none (zero)")
    args = parser.parse_args()
    if args.fk_model and (args.gravity != "zero" or args.arm not in ("gru", "ours")):
        parser.error("--fk-model needs --gravity zero and a learned arm (--arm gru or ours): the policy uses no robot model")
    if (args.base == "gru") != (args.arm == "gru") or (args.base == "gru" and args.gru_model is None):
        parser.error("The GRU drives both base and arm: use --base gru --arm gru --gru-model PATH")
    if (args.base == "direct") != (args.arm == "direct") or (args.base == "direct" and args.direct_model is None):
        parser.error("The plain regressor drives both base and arm: use --base direct --arm direct --direct-model PATH")
    if args.base == "ours" and args.base_model is None:
        parser.error("--base ours needs --base-model")
    if args.arm == "ours" and args.arm_model is None:
        parser.error("--arm ours needs --arm-model")
    run(args)
