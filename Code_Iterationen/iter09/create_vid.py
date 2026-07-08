!!!NOT YET WORKING!!!

### setup environment ###
import os

import cv2
import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import PPO

from run import BurpeeEnv
from stable_baselines3.common.env_util import make_vec_env

#------------------------------------------------------------------------

### create video ###

print("this will create a video only of a already trained model, no training will be done here")

video_name = time.strftime("videos/humanoid_%Y%m%d_%H%M%S.mp4")
os.makedirs("videos", exist_ok=True)


# Setup
FPS = 30
DURATION = 60  # Sekunden
NUM_STEPS = FPS * DURATION

env = BurpeeEnv(render_mode="rgb_array")
model = PPO.load("models/humanoid_burpee")


# Initialisierung
obs, info = env.reset()

frame = env.render()
height, width = frame.shape[:2]

video = cv2.VideoWriter(
    video_name,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (width, height)
)

# Simulation + Recording
for step in range(NUM_STEPS):

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()

    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    # wenn Episode endet -> einfach reset, Video läuft weiter
    if terminated or truncated:
        obs, info = env.reset()

# Cleanup
video.release()
env.close()

print(f"Video gespeichert: {video_name}")