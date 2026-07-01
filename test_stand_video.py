import os
import cv2
import gymnasium as gym
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from standing_env import StandingEnv

video_name = datetime.now().strftime("videos/humanoid_%Y%m%d_%H%M%S.mp4")
os.makedirs("videos", exist_ok=True)


# Setup
FPS = 30
DURATION = 20  # Sekunden
NUM_STEPS = FPS * DURATION

env = StandingEnv(render_mode="rgb_array")
model = PPO.load("models/humanoid_stand")


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