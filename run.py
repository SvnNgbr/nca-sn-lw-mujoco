### setup environment ###
import os
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np


os.makedirs("videos", exist_ok=True)

env = gym.make("Humanoid-v5", render_mode="rgb_array")
obs, info = env.reset()

video_name = datetime.now().strftime("videos/00_test_humanoid_%Y%m%d_%H%M%S.mp4")

frame = env.render()
height, width = frame.shape[:2]

video = cv2.VideoWriter(
    video_name,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (width, height),
)

for step in range(300):
    action = env.action_space.sample() * 0.0

    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

    if terminated or truncated:
        obs, info = env.reset()

video.release()
env.close()

print(f"Video gespeichert: {video_name}")



ACT = {
    "abdomen_y": 0,
    "abdomen_z": 1,
    "abdomen_x": 2,
    "right_hip_x": 3,
    "right_hip_z": 4,
    "right_hip_y": 5,
    "right_knee": 6,
    "left_hip_x": 7,
    "left_hip_z": 8,
    "left_hip_y": 9,
    "left_knee": 10,
    "right_shoulder1": 11,
    "right_shoulder2": 12,
    "right_elbow": 13,
    "left_shoulder1": 14,
    "left_shoulder2": 15,
    "left_elbow": 16,
}


def make_action(phase):
    a = np.zeros(17)

    if phase == "stand":
        a[ACT["abdomen_y"]] = 0.05

    elif phase == "squat":
        a[ACT["abdomen_y"]] = 0.15
        a[ACT["right_hip_y"]] = 0.35
        a[ACT["left_hip_y"]] = 0.35
        a[ACT["right_knee"]] = -0.35
        a[ACT["left_knee"]] = -0.35
        a[ACT["right_shoulder2"]] = 0.25
        a[ACT["left_shoulder2"]] = -0.25

    elif phase == "jump":
        a[ACT["abdomen_y"]] = -0.20
        a[ACT["right_hip_y"]] = -0.40
        a[ACT["left_hip_y"]] = -0.40
        a[ACT["right_knee"]] = 0.40
        a[ACT["left_knee"]] = 0.40
        a[ACT["right_shoulder2"]] = -0.40
        a[ACT["left_shoulder2"]] = 0.40

    elif phase == "land":
        a[ACT["abdomen_y"]] = 0.10
        a[ACT["right_hip_y"]] = 0.20
        a[ACT["left_hip_y"]] = 0.20
        a[ACT["right_knee"]] = -0.20
        a[ACT["left_knee"]] = -0.20

    return np.clip(a, -0.4, 0.4)


os.makedirs("videos", exist_ok=True)

env = gym.make("Humanoid-v5", render_mode="rgb_array")
obs, info = env.reset()

video_name = datetime.now().strftime("videos/01_jump_%Y%m%d_%H%M%S.mp4")

frame = env.render()
height, width = frame.shape[:2]

video = cv2.VideoWriter(
    video_name,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (width, height),
)

for step in range(600):
    cycle_step = step % 200

    if cycle_step < 50:
        phase = "stand"
    elif cycle_step < 100:
        phase = "squat"
    elif cycle_step < 125:
        phase = "jump"
    else:
        phase = "land"

    action = make_action(phase)

    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

    if terminated or truncated:
        obs, info = env.reset()

video.release()
env.close()

print(f"Video gespeichert: {video_name}")

import gymnasium as gym

env = gym.make("Humanoid-v5")

print("qpos:", env.unwrapped.model.nq)
print("qvel:", env.unwrapped.model.nv)

print("\nJoint Names:")

for i in range(env.unwrapped.model.njnt):
    print(i, env.unwrapped.model.joint(i).name)

env.close()