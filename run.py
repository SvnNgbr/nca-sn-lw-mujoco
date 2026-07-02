### setup environment ###
import os

import cv2
import gymnasium as gym
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO

import psutil
from stable_baselines3.common.env_util import make_vec_env


#------------------------------------------------------------------------

### setup training ###

#class BurpeeEnv(gym.Wrapper):
  #  def __init__(self, render_mode=None):
   #     env = gym.make("Humanoid-v5", render_mode=render_mode)
   #     super().__init__(env)
   #     self.step_count = 0
   #     self.phase_length = 150

   # def reset(self, **kwargs):
       # self.step_count = 0
       # return self.env.reset(**kwargs)

    #def step(self, action):
       # obs, _, terminated, truncated, info = self.env.step(action)

       # self.step_count += 1

       # torso_height = self.unwrapped.data.qpos[2]
       # torso_mat = self.unwrapped.data.xmat[1].reshape(3, 3)
        #upright = torso_mat[2, 2]

        #forward_velocity = self.unwrapped.data.qvel[0]
        #vertical_velocity = self.unwrapped.data.qvel[2]
        #energy = np.sum(action ** 2)

        #phase = (self.step_count // self.phase_length) % 6

        #reward = 0.0

        # Phase 0: stehen
        #if phase == 0:
         #   reward += 5.0 * torso_height
          #  reward += 4.0 * upright

        # Phase 1: runter in Squat
        #elif phase == 1:
         #   target_height = 1.0
          #  reward -= 10.0 * abs(torso_height - target_height)
           # reward += 2.0 * upright

        # Phase 2: nach vorne / Plank
        #elif phase == 2:
         #   target_height = 0.75
          #  reward -= 10.0 * abs(torso_height - target_height)
           # reward += 2.0 * forward_velocity

        # Phase 3: Push-up tief
        #elif phase == 3:
         #   target_height = 0.55
          #  reward -= 15.0 * abs(torso_height - target_height)

        # Phase 4: wieder hochkommen
       # elif phase == 4:
        #    reward += 8.0 * torso_height
         #   reward += 3.0 * upright
          #  reward += 2.0 * vertical_velocity

        # Phase 5: Sprung
        #elif phase == 5:
         #   reward += 10.0 * vertical_velocity
          #  reward += 3.0 * upright

        # Energie-Strafe
        #reward -= 0.001 * energy

        # nicht komplett umfallen
        #if torso_height < 0.35:
         #   reward -= 50
          #  terminated = True

        #return obs, reward, terminated, truncated, info
#class BurpeeEnv(gym.Wrapper):
   # def __init__(self, render_mode=None):
    #    env = gym.make(
     #       "Humanoid-v5",
      #      render_mode=render_mode
       # )
        #super().__init__(env)

 #   def step(self, action):
  #      obs, _, terminated, truncated, info = self.env.step(action)
#
 #       # hight torso
  #      torso_height = self.unwrapped.data.qpos[2]
#
        # orientation of torso (3x3 Rotationsmatrix)
 #       torso_mat = self.unwrapped.data.xmat[1].reshape(3, 3)
#
        # amount of uprightness
  #      upright = torso_mat[2, 2]

        # Geschwindigkeit
        #velocity = np.linalg.norm(self.unwrapped.data.qvel)

        # Energieverbrauch
   #     energy = np.sum(action ** 2)

    #    reward = 0.0

        # upright position is good
     #   reward += 5.0 * torso_height

        # do not diverge to sides
      #  reward += 3.0 * upright

        # no strong movements
       # reward -= 0.001 * energy

        # punish if down
        #if torso_height < 0.9:
       #     reward -= 100
        #    terminated = True

    #    return obs, reward, terminated, truncated, info
class BurpeeEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("Humanoid-v5", render_mode=render_mode)
        super().__init__(env)

        self.step_count = 0
        self.phase_length = 120

        self.pose_targets = {
            0: np.array([0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0,   0.2, -0.4, 0.2,   -0.2, 0.4, -0.2]),
            1: np.array([0.2, 0.0, 0.0,   0.8, 0.0, 0.0, -1.4,  0.8, 0.0, 0.0, -1.4,  0.5, -0.8, 0.4,  -0.5, 0.8, -0.4]),
            2: np.array([0.8, 0.0, 0.0,   0.5, 0.0, 0.0, -1.0,  0.5, 0.0, 0.0, -1.0,  1.2, -1.4, 0.8,  -1.2, 1.4, -0.8]),
            3: np.array([1.2, 0.0, 0.0,   0.1, 0.0, 0.0, -0.2,  0.1, 0.0, 0.0, -0.2,  1.5, -1.2, 0.2,  -1.5, 1.2, -0.2]),
            4: np.array([1.2, 0.0, 0.0,   0.1, 0.0, 0.0, -0.2,  0.1, 0.0, 0.0, -0.2,  1.5, -1.2, 1.2,  -1.5, 1.2, -1.2]),
            5: np.array([1.0, 0.0, 0.0,   0.2, 0.0, 0.0, -0.5,  0.2, 0.0, 0.0, -0.5,  1.2, -1.0, 0.3,  -1.2, 1.0, -0.3]),
            6: np.array([0.3, 0.0, 0.0,   0.8, 0.0, 0.0, -1.4,  0.8, 0.0, 0.0, -1.4,  0.5, -0.8, 0.4,  -0.5, 0.8, -0.4]),
            7: np.array([0.0, 0.0, 0.0,   -0.2, 0.0, 0.0, 0.4,  -0.2, 0.0, 0.0, 0.4,  0.0, -0.3, 0.0,   0.0, 0.3, 0.0]),
        }

        self.height_targets = {
            0: 1.40,
            1: 1.05,
            2: 0.85,
            3: 0.65,
            4: 0.45,
            5: 0.70,
            6: 1.00,
            7: 1.45,
        }

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        self.step_count += 1
        phase = (self.step_count // self.phase_length) % 8

        qpos = self.unwrapped.data.qpos
        qvel = self.unwrapped.data.qvel

        torso_height = qpos[2]
        joint_angles = qpos[7:]

        torso_mat = self.unwrapped.data.xmat[1].reshape(3, 3)
        upright = torso_mat[2, 2]

        vertical_velocity = qvel[2]
        energy = np.sum(action ** 2)

        target_pose = self.pose_targets[phase]
        target_height = self.height_targets[phase]

        pose_error = np.linalg.norm(joint_angles - target_pose)
        height_error = abs(torso_height - target_height)

        reward = 0.0

        reward -= 4.0 * pose_error
        reward -= 12.0 * height_error
        reward -= 0.001 * energy

        if phase in [0, 1, 6]:
            reward += 2.0 * upright

        if phase == 7:
            reward += 8.0 * vertical_velocity
            reward += 2.0 * upright

        if torso_height < 0.25:
            reward -= 100
            terminated = True

        return obs, reward, terminated, truncated, info

#------------------------------------------------------------------------

#------------------------------------------------------------------------

### train model ###

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = StandingEnv()

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

model.learn(total_timesteps=50_000)

model.save("models/humanoid_burpee")
env.close()

#------------------------------------------------------------------------

### create video ###

video_name = time.strftime("videos/humanoid_%Y%m%d_%H%M%S.mp4")
os.makedirs("videos", exist_ok=True)


# Setup
FPS = 30
DURATION = 20  # Sekunden
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