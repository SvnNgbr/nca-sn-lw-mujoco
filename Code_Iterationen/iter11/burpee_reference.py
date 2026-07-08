import math

import numpy as np


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
    return np.radians(np.array(values, dtype=np.float32))


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
        "joints": deg([55, -55, 55, -55, -75, 105, -25, -75, 105, -25]),
    },
    {
        "name": "hands_to_floor",
        "duration": 0.8,
        "root_pos": [0.15, 0.0, 0.54],
        "root_euler": [0, 70, 0],
        "joints": deg([-135, -22, -135, -22, -95, 120, -35, -95, 120, -35]),
    },
    {
        "name": "plank",
        "duration": 1.1,
        "root_pos": [0.42, 0.0, 0.45],
        "root_euler": [0, 86, 0],
        "joints": deg([-125, -8, -125, -8, -12, 10, 0, -12, 10, 0]),
    },
    {
        "name": "push_up_down",
        "duration": 0.7,
        "root_pos": [0.42, 0.0, 0.31],
        "root_euler": [0, 86, 0],
        "joints": deg([-125, -92, -125, -92, -10, 8, 0, -10, 8, 0]),
    },
    {
        "name": "push_up_up",
        "duration": 0.7,
        "root_pos": [0.42, 0.0, 0.45],
        "root_euler": [0, 86, 0],
        "joints": deg([-125, -8, -125, -8, -12, 10, 0, -12, 10, 0]),
    },
    {
        "name": "feet_forward",
        "duration": 0.9,
        "root_pos": [0.14, 0.0, 0.58],
        "root_euler": [0, 58, 0],
        "joints": deg([-120, -30, -120, -30, -95, 120, -30, -95, 120, -30]),
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


TOTAL_DURATION = sum(frame["duration"] for frame in BURPEE)


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
        ],
        dtype=np.float32,
    )


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def reference_at(time_seconds):
    t_abs = float(time_seconds) % TOTAL_DURATION
    cursor = 0.0
    for index, frame in enumerate(BURPEE):
        next_frame = BURPEE[(index + 1) % len(BURPEE)]
        duration = frame["duration"]
        if cursor <= t_abs < cursor + duration:
            alpha = smoothstep((t_abs - cursor) / duration)
            root_pos = (1.0 - alpha) * np.array(frame["root_pos"]) + alpha * np.array(
                next_frame["root_pos"]
            )
            root_euler = (1.0 - alpha) * np.array(frame["root_euler"]) + alpha * np.array(
                next_frame["root_euler"]
            )
            joints = (1.0 - alpha) * frame["joints"] + alpha * next_frame["joints"]
            return {
                "phase": np.array([math.sin(2 * math.pi * t_abs / TOTAL_DURATION), math.cos(2 * math.pi * t_abs / TOTAL_DURATION)], dtype=np.float32),
                "root_pos": root_pos.astype(np.float32),
                "root_quat": quat_from_euler_xyz(root_euler),
                "joints": joints.astype(np.float32),
                "name": frame["name"],
            }
        cursor += duration
    return reference_at(0.0)
