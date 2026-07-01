import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import cv2
import os
import mujoco

# 1. Umgebung laden (ohne Rendering)
env = gym.make("StickmanStandup-v0")

# 2. Modell laden
model = PPO.load("stickman_standup_ppo")

# 3. Video-Einstellungen
fps = 30
duration = 10  # Sekunden
num_frames = fps * duration
render_width, render_height = 640, 480

# 4. Ordner für Frames erstellen
os.makedirs("frames", exist_ok=True)
frame_files = []

# 5. Simuliere und rendere Frames
with mujoco.Renderer(env.model, render_height, render_width) as renderer:
    obs, _ = env.reset()
    for i in range(num_frames):
        # Berechne Aktion mit dem Modell
        action, _ = model.predict(obs)

        # Führe die Aktion aus
        obs, _, terminated, truncated, _ = env.step(action)

        # Rendere den aktuellen Frame
        renderer.update_scene(env.data, camera_id=0)
        img = renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Speichere Frame
        fname = f"frames/frame_{i:04d}.png"
        cv2.imwrite(fname, img_bgr)
        frame_files.append(fname)

        # Beende, wenn die Episode endet
        if terminated or truncated:
            obs, _ = env.reset()

# 6. Erstelle Video aus den Frames
video = cv2.VideoWriter(
    "stickman_standup_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (render_width, render_height)
)

for fname in frame_files:
    frame = cv2.imread(fname)
    video.write(frame)

# 7. Aufräumen
video.release()
for fname in frame_files:
    os.remove(fname)
os.rmdir("frames")

print("✅ Video wurde als 'stickman_standup_video.mp4' gespeichert!")