### setup environment ###

import os

import cv2

import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import PPO

import psutil
from stable_baselines3.common.env_util import make_vec_env


#------------------------------------------------------------------------

### setup training ###

class StandingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make(
            "Humanoid-v5",
            render_mode=render_mode
        )
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # hight torso
        torso_height = self.unwrapped.data.qpos[2]

        # orientation of torso (3x3 Rotationsmatrix)
        torso_mat = self.unwrapped.data.xmat[1].reshape(3, 3)

        # amount of uprightness
        upright = torso_mat[2, 2]

        # Geschwindigkeit
        #velocity = np.linalg.norm(self.unwrapped.data.qvel)

        # Energieverbrauch
        energy = np.sum(action ** 2)

        reward = 0.0

        # upright position is good
        reward += 5.0 * torso_height

        # do not diverge to sides
        reward += 3.0 * upright

        # no strong movements
        reward -= 0.001 * energy

        # punish if down
        if torso_height < 0.9:
            reward -= 100
            terminated = True

        return obs, reward, terminated, truncated, info
    

#------------------------------------------------------------------------

#------------------------------------------------------------------------

### train model ###

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)

# Nutze alle physischen Kerne
n_envs = physical_cores

print(f"Physische Kerne : {physical_cores}")
print(f"Logische Kerne  : {logical_cores}")
print(f"Parallele Envs  : {n_envs}")

env = make_vec_env(
    StandingEnv,
    n_envs=n_envs
)

#device = "cuda" if torch.cuda.is_available() else "cpu" #bringt hier nichts

model = PPO(
    "MlpPolicy",
    env,
    device="cpu",     
    #device=device,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    verbose=1,
    tensorboard_log="logs/"
)

start_time = time.perf_counter()

model.learn(total_timesteps=500_000)

end_time = time.perf_counter()

training_time = end_time - start_time

print(f"Training abgeschlossen in {training_time:.2f} Sekunden ")

model.save("models/humanoid_stand")
env.close()

#------------------------------------------------------------------------

### create video ###

video_name = time.strftime("videos/humanoid_%Y%m%d_%H%M%S.mp4")
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