"""NumPy runtime for the staged policy.

StagedBase: learned parking pose, wheel commands and stop latch (base pose and target only).
StagedArm: learned desired joint velocity, integrated into a joint reference tracked by a PD
law with model-based gravity compensation; holds the initial pose until the base latches.
GRUPolicy: one recurrent network for both base and arm. It outputs wheel commands and a 7x3
matrix M (arm velocity = M @ EE error), optionally a stop logit that is latched (the final
method, trained with --stop-head); without the stop head any stage memory lives in the hidden
state. A fixed joint PD layer with model gravity compensation turns the velocity into torques;
the --arm-output torque ablation instead applies the GRU's torques directly with no PD layer.
"""
import json

import numpy as np

from staged_data import base_features

NETS = ("park", "wheels", "stop")


class NumpyMLP:
    """Two hidden ReLU layers, matching staged_train.make_mlp, with stored normalisers."""

    def __init__(self, arrays, name):
        self.layers = [(arrays[f"{name}_w{i}"], arrays[f"{name}_b{i}"]) for i in (0, 2, 4)]
        self.x_mean, self.x_scale = arrays[f"{name}_x_mean"], arrays[f"{name}_x_scale"]
        self.y_mean, self.y_scale = arrays[f"{name}_y_mean"], arrays[f"{name}_y_scale"]

    def __call__(self, x):
        h = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale
        for i, (weight, bias) in enumerate(self.layers):
            h = h @ weight.T + bias
            if i < 2:
                h = np.maximum(h, 0)
        return h * self.y_scale + self.y_mean


def park_pose(net, target):
    """Parking xy and yaw predicted from the target position (rows or a single target)."""
    out = net(target)
    return out[..., :2], np.arctan2(out[..., 2], out[..., 3])


class StagedBase:
    def __init__(self, path, latch_steps=None):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if k != "metadata_json"}
            self.metadata = json.loads(str(data["metadata_json"]))
        self.nets = {name: NumpyMLP(arrays, name) for name in NETS}
        self.latch_steps = int(latch_steps or self.metadata["latch_steps"])
        self.reset()

    def reset(self):
        self.park, self.done, self.streak = None, False, 0

    def step(self, base_xy, yaw, target):
        """Return (4 wheel commands, done). Done latches after latch_steps consecutive stop predictions."""
        if self.park is None:
            self.park = park_pose(self.nets["park"], np.asarray(target, dtype=float))
        if not self.done:
            features = base_features(np.asarray(base_xy, dtype=float), yaw, *self.park)
            self.streak = self.streak + 1 if self.nets["stop"](features)[0] > 0 else 0
            self.done = self.streak >= self.latch_steps
        if self.done:
            return np.zeros(4), True
        return self.nets["wheels"](features), False


class DirectPolicy:
    """Plain BC baseline (L0): raw 21 inputs -> raw 4 wheel commands and 7 arm torques, applied directly.

    kind "linear": ridge regression (weight, bias); kind "mlp": NumpyMLP with prefix "direct".
    Both standardise inputs and outputs with stored training statistics. No latch, no PD layer.
    """

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.w = {k: data[k] for k in data.files}
        self.kind = str(self.w["kind"])
        self.mlp = NumpyMLP(self.w, "direct") if self.kind == "mlp" else None

    def act(self, x):
        """x: base_pos (3), base_bearing (1), q_mes (7), qd_mes (7), target (3). Returns (wheels, torques)."""
        x = np.asarray(x, dtype=np.float64)
        if self.mlp is not None:
            y = self.mlp(x)
        else:
            w = self.w
            y = (w["weight"] @ ((x - w["x_mean"]) / w["x_scale"]) + w["bias"]) * w["y_scale"] + w["y_mean"]
        return y[:4], y[4:11]


def arm_features(ee_error_world, base_yaw, q_arm):
    """End-effector error rotated into the base yaw frame, followed by the 7 arm joint angles."""
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    e = np.asarray(ee_error_world, dtype=float)
    local = np.stack((c * e[..., 0] + s * e[..., 1], -s * e[..., 0] + c * e[..., 1], e[..., 2]), axis=-1)
    return np.concatenate((local, np.asarray(q_arm, dtype=float)), axis=-1)


class StagedArm:
    """The network outputs a 7x3 matrix M(error, q); desired velocity is M @ error, zero at zero error."""

    def __init__(self, path, kp=None, kd=None, lead=None):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if k != "metadata_json"}
            self.metadata = json.loads(str(data["metadata_json"]))
        self.net = NumpyMLP(arrays, "arm")
        m = self.metadata
        self.kp = np.full(7, kp) if kp is not None else np.asarray(m["kp"], dtype=float)
        self.kd = np.full(7, kd) if kd is not None else np.asarray(m["kd"], dtype=float)
        self.lead = float(lead if lead is not None else m["reference_lead"])
        self.v_max = np.asarray(m["velocity_limit"], dtype=float)
        self.hold = self.reference = None

    def reset(self, q_arm):
        self.hold, self.reference = np.array(q_arm, dtype=float), np.array(q_arm, dtype=float)

    def step(self, q_arm, qd_arm, ee_pos, target, base_yaw, gravity, reaching, dt):
        """Arm torques before actuator clipping; gravity is the model term for the 7 arm joints."""
        if reaching:
            features = arm_features(target - ee_pos, base_yaw, q_arm)
            velocity = self.net(features).reshape(7, 3) @ features[:3]
            velocity = np.clip(velocity, -self.v_max, self.v_max)
            self.reference = np.clip(self.reference + velocity * dt, q_arm - self.lead, q_arm + self.lead)
        else:
            velocity, self.reference = np.zeros(7), self.hold.copy()
        return self.kp * (self.reference - q_arm) + self.kd * (velocity - qd_arm) + gravity


GRU_FEATURES = ["target_local_x", "target_local_y", "target_dz", "target_dist_xy",
                "sin_target_dir_local", "cos_target_dir_local",
                "sin_target_world_bearing_minus_yaw", "cos_target_world_bearing_minus_yaw",
                "ee_err_local_x", "ee_err_local_y", "ee_err_local_z"] + [f"q{i}" for i in range(1, 8)]
GRU_PERIOD = 10  # simulation steps per GRU update (1 ms steps -> 100 Hz)


def gru_features(base_pos, yaw, q_arm, ee_pos, target):
    """Relative-coordinate features from the listed measurements and the target (broadcasts over rows)."""
    base_pos, ee_pos, target = (np.asarray(a, dtype=float) for a in (base_pos, ee_pos, target))
    c, s = np.cos(yaw), np.sin(yaw)
    rotate = lambda v: np.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)
    rel = rotate(target - base_pos)
    direction = np.arctan2(rel[..., 1], rel[..., 0])
    world = np.arctan2(target[..., 1], target[..., 0]) - yaw
    return np.concatenate((rel, np.linalg.norm(rel[..., :2], axis=-1, keepdims=True),
                           np.stack((np.sin(direction), np.cos(direction), np.sin(world), np.cos(world)), axis=-1),
                           rotate(target - ee_pos), np.asarray(q_arm, dtype=float)), axis=-1)


# Variant without the hand-made world-bearing feature: the parking pose comes from the learned
# park MLP (staged_train.py) and enters only as continuous base-frame quantities.
PARK_GRU_FEATURES = ["target_local_x", "target_local_y", "target_dz", "target_dist_xy",
                     "park_local_x", "park_local_y", "sin_park_yaw_err", "cos_park_yaw_err",
                     "ee_err_local_x", "ee_err_local_y", "ee_err_local_z"] + [f"q{i}" for i in range(1, 8)]


def gru_park_features(base_pos, yaw, q_arm, ee_pos, target, park_xy, park_yaw):
    """Like gru_features, with the target direction and world bearing replaced by the park pose."""
    base_pos, ee_pos, target = (np.asarray(a, dtype=float) for a in (base_pos, ee_pos, target))
    c, s = np.cos(yaw), np.sin(yaw)
    rotate = lambda v: np.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]), axis=-1)
    rel = target - base_pos
    rel_xy = rotate(rel)
    park = rotate(np.asarray(park_xy, dtype=float) - base_pos[..., :2])
    yaw_err = park_yaw - yaw
    ee = target - ee_pos
    return np.concatenate((rel_xy, rel[..., 2:3], np.linalg.norm(rel_xy, axis=-1, keepdims=True), park,
                           np.stack((np.sin(yaw_err), np.cos(yaw_err)), axis=-1),
                           rotate(ee), ee[..., 2:3], np.asarray(q_arm, dtype=float)), axis=-1)


def local_position(point, base_pos, yaw):
    """A world point relative to the base, in the base yaw frame (broadcasts over rows)."""
    v = np.asarray(point, dtype=float) - np.asarray(base_pos, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)


class LearnedFK:
    """Learned forward kinematics (staged_fk_train.py): arm joints and base pose -> EE world position."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.net = NumpyMLP({k: data[k] for k in data.files}, "fk")

    def __call__(self, q_arm, base_pos, yaw):
        local = self.net(q_arm)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.asarray(base_pos, dtype=float) + np.array((c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GRUPolicy:
    """Outputs 4 wheel commands and a 7x3 matrix M every GRU_PERIOD steps; arm velocity is M @ error.

    With a stop head (metadata stop_head), a learned stop logit is latched after latch_ticks
    consecutive positive updates: before the latch the arm holds its initial pose, after it the
    wheels are zero. Without a stop head there is no latch and the GRU drives everything.
    With metadata arm_output "torque" (ablation) the reaching arm applies the GRU's 7 torques
    directly, held between updates, instead of M @ error tracked by the PD layer.
    With metadata feature_set "park" the inputs are gru_park_features, using the park MLP stored
    in the same file.
    """

    def __init__(self, path, kp=None, kd=None, lead=None, latch_ticks=None):
        with np.load(path, allow_pickle=False) as data:
            self.w = {k: data[k] for k in data.files if k != "metadata_json"}
            self.metadata = json.loads(str(data["metadata_json"]))
        m = self.metadata
        self.kp = np.full(7, kp if kp is not None else m["kp"])
        self.kd = np.full(7, kd if kd is not None else m["kd"])
        self.lead = float(lead if lead is not None else m["reference_lead"])
        self.v_max = np.asarray(m["velocity_limit"], dtype=float)
        self.period = int(m["period"])
        self.stop_head = bool(m.get("stop_head", False))
        self.latch_ticks = int(latch_ticks or m.get("latch_ticks", 1))
        self.direct_torque = m.get("arm_output", "matrix") == "torque"
        self.park_net = NumpyMLP(self.w, "park") if m.get("feature_set") == "park" else None

    def reset(self, q_arm):
        self.h = np.zeros(self.w["w_hh"].shape[1])
        self.hold, self.reference = np.array(q_arm, dtype=float), np.array(q_arm, dtype=float)
        self.wheels, self.matrix, self.step_count = np.zeros(4), np.zeros((7, 3)), 0
        self.torque, self.park = np.zeros(7), None
        self.stop_logit, self.streak, self.latched = -np.inf, 0, False

    def _update(self, features):
        w = self.w
        x = (features - w["x_mean"]) / w["x_scale"]
        gi, gh = w["w_ih"] @ x + w["b_ih"], w["w_hh"] @ self.h + w["b_hh"]
        size = len(self.h)
        r = sigmoid(gi[:size] + gh[:size])
        z = sigmoid(gi[size:2 * size] + gh[size:2 * size])
        n = np.tanh(gi[2 * size:] + r * gh[2 * size:])
        self.h = (1 - z) * n + z * self.h
        out = self.raw = w["w_out"] @ self.h + w["b_out"]
        self.wheels = out[:4] * w["wheel_scale"] + w["wheel_mean"]
        if self.direct_torque:
            self.torque = out[4:11] * w["torque_scale"] + w["torque_mean"]
        else:
            self.matrix = out[4:25].reshape(7, 3)
        if self.stop_head:
            self.stop_logit = out[-1]
            self.streak = self.streak + 1 if self.stop_logit > 0 else 0
            self.latched = self.latched or self.streak >= self.latch_ticks

    def step(self, base_pos, yaw, q_arm, qd_arm, ee_pos, target, gravity, dt):
        """Return (4 wheel commands, 7 arm torques before actuator clipping)."""
        if self.park_net is None:
            features = gru_features(base_pos, yaw, q_arm, ee_pos, target)
        else:
            if self.park is None:
                self.park = park_pose(self.park_net, np.asarray(target, dtype=float))
            features = gru_park_features(base_pos, yaw, q_arm, ee_pos, target, *self.park)
        if self.step_count % self.period == 0:
            self._update(features)
        self.step_count += 1
        if self.stop_head and not self.latched:
            velocity, self.reference = np.zeros(7), self.hold.copy()
        elif self.direct_torque:
            return (np.zeros(4) if self.latched else self.wheels.copy()), self.torque.copy()
        else:
            velocity = np.clip(self.matrix @ features[8:11], -self.v_max, self.v_max)
            self.reference = np.clip(self.reference + velocity * dt, q_arm - self.lead, q_arm + self.lead)
        wheels = np.zeros(4) if self.latched else self.wheels.copy()
        return wheels, self.kp * (self.reference - q_arm) + self.kd * (velocity - qd_arm) + gravity
