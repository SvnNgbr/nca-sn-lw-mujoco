import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path(__file__).with_name("humanoid_burpee.xml")

JOINT_NAMES = [
    "left_shoulder",
    "left_elbow",
    "right_shoulder",
    "right_elbow",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
]


def deg(values):
    return np.radians(np.array(values, dtype=float))


# Keyframes: torso pose + target joint angles.
# This is a scripted controller, not learned locomotion. It gives you a clear
# starting point for tuning, reinforcement learning, or inverse kinematics later.
BURPEE = [
    {
        "name": "stand",
        "duration": 1.0,
        "root_pos": [0.0, 0.0, 1.12],
        "root_euler": [0, 0, 0],
        "joints": deg([15, -20, 15, -20, -8, 8, 4, -8, 8, 4]),
    },
    {
        "name": "squat",
        "duration": 1.0,
        "root_pos": [0.0, 0.0, 0.72],
        "root_euler": [0, 22, 0],
        "joints": deg([70, -60, 70, -60, -75, 105, -25, -75, 105, -25]),
    },
    {
        "name": "hands_to_floor",
        "duration": 0.8,
        "root_pos": [0.15, 0.0, 0.54],
        "root_euler": [0, 70, 0],
        "joints": deg([115, -20, 115, -20, -95, 120, -35, -95, 120, -35]),
    },
    {
        "name": "plank",
        "duration": 1.1,
        "root_pos": [0.42, 0.0, 0.45],
        "root_euler": [0, 86, 0],
        "joints": deg([95, -8, 95, -8, -12, 10, 0, -12, 10, 0]),
    },
    {
        "name": "push_up_down",
        "duration": 0.7,
        "root_pos": [0.42, 0.0, 0.31],
        "root_euler": [0, 86, 0],
        "joints": deg([95, -95, 95, -95, -10, 8, 0, -10, 8, 0]),
    },
    {
        "name": "push_up_up",
        "duration": 0.7,
        "root_pos": [0.42, 0.0, 0.45],
        "root_euler": [0, 86, 0],
        "joints": deg([95, -8, 95, -8, -12, 10, 0, -12, 10, 0]),
    },
    {
        "name": "feet_forward",
        "duration": 0.9,
        "root_pos": [0.14, 0.0, 0.58],
        "root_euler": [0, 58, 0],
        "joints": deg([110, -35, 110, -35, -95, 120, -30, -95, 120, -30]),
    },
    {
        "name": "jump",
        "duration": 0.8,
        "root_pos": [0.08, 0.0, 1.28],
        "root_euler": [0, -5, 0],
        "joints": deg([-110, -20, -110, -20, 15, 5, 10, 15, 5, 10]),
    },
    {
        "name": "land",
        "duration": 0.8,
        "root_pos": [0.0, 0.0, 1.02],
        "root_euler": [0, 0, 0],
        "joints": deg([20, -25, 20, -25, -20, 35, -10, -20, 35, -10]),
    },
]


def quat_from_euler_xyz(euler_deg):
    roll, pitch, yaw = np.radians(euler_deg)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def interpolate_pose(a, b, alpha):
    t = smoothstep(alpha)
    root_pos = (1.0 - t) * np.array(a["root_pos"]) + t * np.array(b["root_pos"])
    joints = (1.0 - t) * a["joints"] + t * b["joints"]
    euler = (1.0 - t) * np.array(a["root_euler"]) + t * np.array(b["root_euler"])
    return root_pos, quat_from_euler_xyz(euler), joints


def apply_scripted_root(model, data, root_pos, root_quat):
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    qpos_addr = model.jnt_qposadr[root_joint_id]
    qvel_addr = model.jnt_dofadr[root_joint_id]
    data.qpos[qpos_addr : qpos_addr + 3] = root_pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = root_quat
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ]
    qpos_addrs = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = [0.2, 0.0, 0.65]

        frame_start = time.time()
        while viewer.is_running():
            elapsed = (time.time() - frame_start) % sum(k["duration"] for k in BURPEE)

            cursor = 0.0
            for index, keyframe in enumerate(BURPEE):
                next_keyframe = BURPEE[(index + 1) % len(BURPEE)]
                segment_duration = keyframe["duration"]
                if cursor <= elapsed < cursor + segment_duration:
                    alpha = (elapsed - cursor) / segment_duration
                    root_pos, root_quat, joints = interpolate_pose(
                        keyframe, next_keyframe, alpha
                    )
                    data.ctrl[:] = joints
                    for qpos_addr, joint_value in zip(qpos_addrs, joints):
                        data.qpos[qpos_addr] = joint_value
                    apply_scripted_root(model, data, root_pos, root_quat)
                    break
                cursor += segment_duration

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
