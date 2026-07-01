import gymnasium as gym
from stable_baselines3 import PPO
import cv2
import os
import numpy as np

# 1. Umgebung laden (mit RGB-Rendering für Frames)
env = gym.make("Humanoid-v5", render_mode="rgb_array")

# 2. Modell laden
model = PPO.load("humanoid_stand_ppo")

# 3. Video-Einstellungen
fps = 30
duration = 20
num_frames = fps * duration

# 4. Frames rendern
os.makedirs("frames", exist_ok=True)
frame_files = []

obs, _ = env.reset()
for i in range(num_frames):
    action, _ = model.predict(obs)
    obs, _, terminated, truncated, _ = env.step(action)

    # Rendere Frame (mit OpenCV)
    img = env.render()
    if img is not None:
        img = np.array(img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fname = f"frames/frame_{i:04d}.png"
        cv2.imwrite(fname, img_bgr)
        frame_files.append(fname)

    if terminated or truncated:
        obs, _ = env.reset()

# 5. Video erstellen
frame_shape = cv2.imread(frame_files[0]).shape[:2][::-1] if frame_files else (640, 480)
video = cv2.VideoWriter(
    "humanoid_stand.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    frame_shape
)

for fname in frame_files:
    video.write(cv2.imread(fname))

video.release()
for fname in frame_files:
    os.remove(fname)
os.rmdir("frames")

print(" Video wurde als 'humanoid_stand.mp4' gespeichert!")