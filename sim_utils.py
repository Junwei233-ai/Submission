"""Simulation helpers needed by every rollout, written independently of the expert controller.

The learned policies must not depend on robotnik_arm_control.py (the expert). The two helpers the
rollout needs from the task template -- yaw from the base quaternion and holding the gripper
fingers at a fixed angle -- are re-implemented here so that a learned-only rollout never imports
the expert module. Behaviour is identical to the template versions (checked in test_sim_utils.py).
"""
import numpy as np

FINGER_JOINT_NAMES = [
    "robot_j2s7s300_joint_finger_1", "robot_j2s7s300_joint_finger_tip_1",
    "robot_j2s7s300_joint_finger_2", "robot_j2s7s300_joint_finger_tip_2",
    "robot_j2s7s300_joint_finger_3", "robot_j2s7s300_joint_finger_tip_3",
]


def quaternion_yaw(q_w, q_x, q_y, q_z):
    """Planar bearing (yaw, rad) of a quaternion, ZYX convention, in (-pi, pi]."""
    norm = np.sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z)
    q_w, q_x, q_y, q_z = q_w / norm, q_x / norm, q_y / norm, q_z / norm
    return float(np.arctan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z)))


def hold_gripper_fingers(sim, client, rid, angles):
    """Reset the six finger joints to `angles` and position-hold them (5 N force) on every step.

    Wheels (0..3) and arm (4..10) keep the normal command path; this is the actuator setup used
    when the demonstrations were recorded, not part of any controller.
    """
    finger_ids = [j for j in range(client.getNumJoints(rid))
                  if client.getJointInfo(rid, j)[1].decode() in FINGER_JOINT_NAMES]
    if len(finger_ids) != 6:
        raise ValueError("Unexpected gripper model")
    angles = np.asarray(angles, dtype=float)
    for joint, angle in zip(finger_ids, angles):
        client.resetJointState(rid, joint, targetValue=float(angle), targetVelocity=0.0)
    sim.ReceiveObservation()
    original = sim._SetMotorTorqueByIds

    def set_torque(motor_ids, commands, index=0):
        original(motor_ids[:11], commands[:11], index=index)
        sim.pybullet_client.setJointMotorControlArray(
            bodyIndex=sim.bot[index].bot_pybullet, jointIndices=motor_ids[11:17],
            controlMode=sim.pybullet_client.POSITION_CONTROL,
            targetPositions=angles.tolist(), forces=[5.0] * 6)

    sim._SetMotorTorqueByIds = set_torque
    return angles
