import math
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np


# Humanoid v5 aus Gymnasium
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "torso_joint",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow"
]

def deg(values):
    return np.radians(np.array(values, dtype=float))

# Burpee-Keyframes für Humanoid v5 angepasst
BURPEE = [
    {
        "name": "stand",
        "duration": 1.0,
        "root_pos": [0.0, 0.0, 1.5],  # Höhe angepasst für v5
        "root_euler": [0, 0, 0],
        "joints": deg([0, 0, 0, 0, 0, 0,   # linkes Bein
                      0, 0, 0, 0, 0, 0,   # rechtes Bein
                      0,                  # torso
                      0, 0, 0, 0,         # linker Arm
                      0, 0, 0, 0])        # rechter Arm
    },
    {
        "name": "squat",
        "duration": 1.0,
        "root_pos": [0.0, 0.0, 1.0],
        "root_euler": [0, 20, 0],
        "joints": deg([-30, 0, 0, 60, -20, 0,   # linkes Bein
                      -30, 0, 0, 60, -20, 0,    # rechtes Bein
                      0,                         # torso
                      0, 0, 0, 0,                # linker Arm
                      0, 0, 0, 0])               # rechter Arm
    },
    {
        "name": "hands_to_floor",
        "duration": 0.8,
        "root_pos": [0.15, 0.0, 0.7],
        "root_euler": [0, 60, 0],
        "joints": deg([-60, 0, 0, 90, -30, 0,   # linkes Bein
                      -60, 0, 0, 90, -30, 0,    # rechtes Bein
                      0,                         # torso
                      -90, 20, 0, -90,           # linker Arm
                      -90, -20, 0, -90])         # rechter Arm
    },
    {
        "name": "plank",
        "duration": 1.1,
        "root_pos": [0.4, 0.0, 0.5],
        "root_euler": [0, 80, 0],
        "joints": deg([-20, 0, 0, 10, 0, 0,   # linkes Bein
                      -20, 0, 0, 10, 0, 0,    # rechtes Bein
                      0,                      # torso
                      -110, 0, 0, -80,        # linker Arm
                      -110, 0, 0, -80])       # rechter Arm
    },
    {
        "name": "push_up_down",
        "duration": 0.7,
        "root_pos": [0.4, 0.0, 0.35],
        "root_euler": [0, 80, 0],
        "joints": deg([-20, 0, 0, 10, 0, 0,   # linkes Bein
                      -20, 0, 0, 10, 0, 0,    # rechtes Bein
                      0,                      # torso
                      -130, 20, 0, -100,      # linker Arm
                      -130, -20, 0, -100])    # rechter Arm
    },
    {
        "name": "push_up_up",
        "duration": 0.7,
        "root_pos": [0.4, 0.0, 0.5],
        "root_euler": [0, 80, 0],
        "joints": deg([-20, 0, 0, 10, 0, 0,   # linkes Bein
                      -20, 0, 0, 10, 0, 0,    # rechtes Bein
                      0,                      # torso
                      -110, 0, 0, -80,        # linker Arm
                      -110, 0, 0, -80])       # rechter Arm
    },
    {
        "name": "feet_forward",
        "duration": 0.9,
        "root_pos": [0.15, 0.0, 0.7],
        "root_euler": [0, 50, 0],
        "joints": deg([-60, 0, 0, 90, -30, 0,   # linkes Bein
                      -60, 0, 0, 90, -30, 0,    # rechtes Bein
                      0,                         # torso
                      -90, 20, 0, -90,           # linker Arm
                      -90, -20, 0, -90])         # rechter Arm
    },
    {
        "name": "jump",
        "duration": 0.8,
        "root_pos": [0.1, 0.0, 1.6],
        "root_euler": [0, -5, 0],
        "joints": deg([-10, 0, 0, 5, 10, 0,   # linkes Bein
                      -10, 0, 0, 5, 10, 0,    # rechtes Bein
                      0,                      # torso
                      -90, 0, 0, -80,         # linker Arm
                      -90, 0, 0, -80])        # rechter Arm
    },
    {
        "name": "land",
        "duration": 0.8,
        "root_pos": [0.0, 0.0, 1.3],
        "root_euler": [0, 0, 0],
        "joints": deg([-15, 0, 0, 25, -10, 0,   # linkes Bein
                      -15, 0, 0, 25, -10, 0,    # rechtes Bein
                      0,                         # torso
                      0, 0, 0, 0,                # linker Arm
                      0, 0, 0, 0])               # rechter Arm
    },
]

def quat_from_euler_xyz(euler_deg):
    roll, pitch, yaw = np.radians(euler_deg)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)

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
    data.qpos[qpos_addr:qpos_addr+3] = root_pos
    data.qpos[qpos_addr+3:qpos_addr+7] = root_quat
    data.qvel[qvel_addr:qvel_addr+6] = 0.0

def main():
    # Humanoid v5 aus Gymnasium laden
    env = gym.make("Humanoid-v5", render_mode="human")
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Joint IDs für unsere definierten Joint-Namen
    joint_ids = []
    for name in JOINT_NAMES:
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_ids.append(joint_id)
        except:
            print(f"Warning: Joint {name} not found")
            joint_ids.append(-1)
    
    qpos_addrs = []
    for joint_id in joint_ids:
        if joint_id >= 0:
            qpos_addrs.append(model.jnt_qposadr[joint_id])
        else:
            qpos_addrs.append(-1)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]

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
                    
                    # Joint-Werte anwenden
                    for qpos_addr, joint_value in zip(qpos_addrs, joints):
                        if qpos_addr >= 0:
                            data.qpos[qpos_addr] = joint_value
                    
                    apply_scripted_root(model, data, root_pos, root_quat)
                    break
                cursor += segment_duration

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()