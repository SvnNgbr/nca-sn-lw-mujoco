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

class BurpeeEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make(
            "Humanoid-v5",
            render_mode=render_mode,
            terminate_when_unhealthy=False # cuz healthy is defined as hight > 1.0
        )
        super().__init__(env)
        '''
        nichtmehr step gebunden sonder phase gebunden
        erst weiter wenn target erreicht wurde
        self.step_count = 0
        self.phase_length = 120
        '''
        self.step_count = 0

        # NEW:
        # Current phase of the burpee sequence.
        # The agent stays in this phase until it reaches the target.
        self.current_phase = 0


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

    '''
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
    '''
    def reset(self, **kwargs):
        self.step_count = 0

        # NEW:
        # Always start a new episode from phase 0.
        self.current_phase = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        self.step_count += 1
        '''
        phase = (self.step_count // self.phase_length) % 8
        '''
        # NEW:
        # Stay in the current phase until the target is reached.
        phase = self.current_phase

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

        # ----------------------------------------------------------
        # NEW:
        # Move to the next phase only after reaching the current one.
        # ----------------------------------------------------------

        POSE_THRESHOLD = 0.35
        HEIGHT_THRESHOLD = 0.08

        reward = 0.0


        reward -= 1.0 * pose_error  # to not punish doing somthing other than falling in the beginning
        reward -= 12.0 * height_error
        reward -= 0.001 * energy

        if phase in [0, 1, 6]:
            reward += 2.0 * upright

        if phase == 7:
            reward += 8.0 * vertical_velocity
            reward += 2.0 * upright

        # --- SUCCESS BONUS (ONLY ADDITION) ---
        if (
            pose_error < POSE_THRESHOLD
            and height_error < HEIGHT_THRESHOLD
        ):
            reward += 100.0

            self.current_phase = min(self.current_phase + 1, 7)

            print(f"Reached phase {self.current_phase}")

        return obs, reward, terminated, truncated, info

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
    BurpeeEnv,
    n_envs=n_envs
)


model = PPO(
    "MlpPolicy",
    env,
    device="cpu",     
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    verbose=1,
    tensorboard_log="logs/"
)

start_time = time.perf_counter()

model.learn(total_timesteps=10_000_000) #brauchen hier wohl 10mio (5mio ca 45min)

end_time = time.perf_counter()

training_time = end_time - start_time

print(f"Training abgeschlossen in {training_time:.2f} Sekunden ")

model.save("models/humanoid_burpee")
env.close()

#------------------------------------------------------------------------

### create video ###

video_name = time.strftime("videos/humanoid_%Y%m%d_%H%M%S.mp4")
os.makedirs("videos", exist_ok=True)


# Setup
FPS = 30
DURATION = 30  # Sekunden
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