import os
import sys
import math
import argparse
import subprocess
from datetime import datetime


def ensure_package(module, package):
    try:
        __import__(module)
    except ImportError:
        print(f"Installiere fehlendes Paket: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


ensure_package("numpy", "numpy")
ensure_package("cv2", "opencv-python")
ensure_package("gymnasium", "gymnasium[mujoco]")
ensure_package("stable_baselines3", "stable-baselines3")
ensure_package("mujoco", "mujoco")

import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


MODEL_PATH = "models/humanoid_burpee_mocap_v3"


# ------------------------------------------------------------
# Kleine Hilfsfunktionen
# ------------------------------------------------------------

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def set_if_exists(names, values, joint_name, value):
    if joint_name in names:
        values[names[joint_name]] = value


# ------------------------------------------------------------
# Synthetische MoCap-Keyframes
# ------------------------------------------------------------

class BurpeeMocap:
    """Erzeugt eine Burpee-Referenzbewegung fuer Humanoid-v5.

    Die Keyframes bestehen aus:
    - root position/orientation
    - Gelenkzielpositionen fuer qpos[7:]
    - Phasenname
    """

    def __init__(self, env, fps=30):
        self.env = env
        self.model = env.unwrapped.model
        self.fps = fps
        self.dt = self.model.opt.timestep
        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.joint_names = self._get_joint_qpos_map()
        self.keyframes = self._make_keyframes()
        self.total_frames = self.keyframes[-1][0]

    def _get_joint_qpos_map(self):
        names = {}
        # freie Wurzel hat qpos 0..6, danach kommen die hinge/ball joints.
        for j in range(self.model.njnt):
            name = self.model.joint(j).name
            adr = int(self.model.jnt_qposadr[j])
            if adr >= 7:
                names[name] = adr - 7
        return names

    def _pose(self, **kwargs):
        p = np.zeros(self.qpos_dim - 7, dtype=np.float64)
        for k, v in kwargs.items():
            set_if_exists(self.joint_names, p, k, v)
        return p

    def _make_keyframes(self):
        # Hinweis: Die genauen Joint-Namen koennen je nach Gymnasium-Version leicht variieren.
        # Deshalb werden nur Namen gesetzt, die tatsaechlich existieren.

        stand = self._pose(
            abdomen_y=0.00, abdomen_z=0.00, abdomen_x=0.00,
            right_hip_y=0.05, right_knee=0.05, right_ankle_y=0.00,
            left_hip_y=0.05, left_knee=0.05, left_ankle_y=0.00,
            right_shoulder1=0.15, right_shoulder2=-0.35, right_elbow=0.10,
            left_shoulder1=-0.15, left_shoulder2=0.35, left_elbow=-0.10,
        )

        squat = self._pose(
            abdomen_y=0.35,
            right_hip_y=-0.95, right_knee=1.85, right_ankle_y=-0.55,
            left_hip_y=-0.95, left_knee=1.85, left_ankle_y=-0.55,
            right_shoulder1=0.55, right_shoulder2=-0.85, right_elbow=0.45,
            left_shoulder1=-0.55, left_shoulder2=0.85, left_elbow=-0.45,
        )

        hands_floor = self._pose(
            abdomen_y=0.95,
            right_hip_y=-1.25, right_knee=1.55, right_ankle_y=-0.35,
            left_hip_y=-1.25, left_knee=1.55, left_ankle_y=-0.35,
            right_shoulder1=1.25, right_shoulder2=-1.15, right_elbow=0.25,
            left_shoulder1=-1.25, left_shoulder2=1.15, left_elbow=-0.25,
        )

        plank = self._pose(
            abdomen_y=1.35,
            right_hip_y=0.05, right_knee=0.15, right_ankle_y=0.10,
            left_hip_y=0.05, left_knee=0.15, left_ankle_y=0.10,
            right_shoulder1=1.45, right_shoulder2=-1.20, right_elbow=0.15,
            left_shoulder1=-1.45, left_shoulder2=1.20, left_elbow=-0.15,
        )

        push_down = self._pose(
            abdomen_y=1.35,
            right_hip_y=0.05, right_knee=0.10, right_ankle_y=0.10,
            left_hip_y=0.05, left_knee=0.10, left_ankle_y=0.10,
            right_shoulder1=1.35, right_shoulder2=-1.15, right_elbow=1.15,
            left_shoulder1=-1.35, left_shoulder2=1.15, left_elbow=-1.15,
        )

        jump = self._pose(
            abdomen_y=-0.10,
            right_hip_y=0.25, right_knee=-0.15, right_ankle_y=0.25,
            left_hip_y=0.25, left_knee=-0.15, left_ankle_y=0.25,
            right_shoulder1=-0.65, right_shoulder2=-0.25, right_elbow=0.05,
            left_shoulder1=0.65, left_shoulder2=0.25, left_elbow=-0.05,
        )

        # frame, x, z, pitch, pose, phase_name
        # pitch dreht den Koerper optisch nach vorne in Richtung Boden.
        return [
            (0,   0.00, 1.38, 0.00, stand,       "stand"),
            (35,  0.03, 1.02, 0.10, squat,       "squat"),
            (70,  0.10, 0.72, 0.65, hands_floor, "hands_down"),
            (105, 0.45, 0.56, 1.30, plank,       "plank"),
            (135, 0.50, 0.43, 1.35, push_down,   "pushup_down"),
            (165, 0.55, 0.57, 1.28, plank,       "pushup_up"),
            (205, 0.32, 0.95, 0.35, squat,       "feet_forward"),
            (235, 0.36, 1.58, -0.05, jump,       "jump"),
            (270, 0.42, 1.38, 0.00, stand,       "land"),
        ]

    def get(self, frame):
        frame = frame % self.total_frames
        for i in range(len(self.keyframes) - 1):
            a = self.keyframes[i]
            b = self.keyframes[i + 1]
            if a[0] <= frame <= b[0]:
                local = (frame - a[0]) / max(1, b[0] - a[0])
                t = smoothstep(local)
                x = lerp(a[1], b[1], t)
                z = lerp(a[2], b[2], t)
                pitch = lerp(a[3], b[3], t)
                pose = lerp(a[4], b[4], t)
                phase = b[5]
                return self.make_qpos(x, z, pitch, pose), phase
        a = self.keyframes[-1]
        return self.make_qpos(a[1], a[2], a[3], a[4]), a[5]

    def make_qpos(self, x, z, pitch, pose):
        qpos = np.zeros(self.qpos_dim, dtype=np.float64)
        qpos[0] = x
        qpos[1] = 0.0
        qpos[2] = z

        # Quaternion fuer Rotation um y-Achse: [w, x, y, z]
        half = pitch / 2.0
        qpos[3] = math.cos(half)
        qpos[4] = 0.0
        qpos[5] = math.sin(half)
        qpos[6] = 0.0

        n = min(len(pose), len(qpos[7:]))
        qpos[7:7+n] = pose[:n]
        return qpos


# ------------------------------------------------------------
# Demo: direkte Wiedergabe der Referenzbewegung
# ------------------------------------------------------------

def write_video(path, frames, fps=30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def render_mocap_demo():
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    mocap = BurpeeMocap(env)
    obs, info = env.reset()
    frames = []

    cycles = 2
    for frame_id in range(mocap.total_frames * cycles):
        qpos, phase = mocap.get(frame_id)
        env.unwrapped.set_state(qpos, np.zeros(env.unwrapped.model.nv))
        env.unwrapped.data.qvel[:] = 0.0
        env.unwrapped.data.qacc[:] = 0.0
        env.unwrapped.data.ctrl[:] = 0.0
        frame = env.render()
        frames.append(frame)

    filename = datetime.now().strftime("videos/mocap_burpee_reference_v3_%Y%m%d_%H%M%S.mp4")
    write_video(filename, frames, fps=30)
    env.close()
    print(f"MoCap-Referenzvideo gespeichert: {filename}")


# ------------------------------------------------------------
# RL-Umgebung: lernt der Referenztrajektorie zu folgen
# ------------------------------------------------------------

class BurpeeImitationEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("Humanoid-v5", render_mode=render_mode)
        super().__init__(env)
        self.mocap = BurpeeMocap(self.env)
        self.frame_id = 0
        self.max_episode_steps = self.mocap.total_frames

    def reset(self, **kwargs):
        self.frame_id = 0
        obs, info = self.env.reset(**kwargs)
        # Start nahe an der Referenz, damit Lernen nicht komplett chaotisch beginnt.
        qpos, _ = self.mocap.get(0)
        noise = np.random.normal(0.0, 0.015, size=qpos.shape)
        noise[:7] = 0.0
        self.unwrapped.set_state(qpos + noise, np.zeros(self.unwrapped.model.nv))
        obs = self.unwrapped._get_obs()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.frame_id += 1

        target_qpos, phase = self.mocap.get(self.frame_id)
        qpos = self.unwrapped.data.qpos.copy()
        qvel = self.unwrapped.data.qvel.copy()

        root_pos_error = np.linalg.norm(qpos[:3] - target_qpos[:3])
        root_rot_error = np.linalg.norm(qpos[3:7] - target_qpos[3:7])
        joint_error = np.linalg.norm(qpos[7:] - target_qpos[7:])
        velocity_penalty = np.linalg.norm(qvel) * 0.01
        energy_penalty = np.sum(np.square(action)) * 0.001

        reward = 0.0
        reward -= 8.0 * root_pos_error
        reward -= 3.0 * root_rot_error
        reward -= 2.5 * joint_error
        reward -= velocity_penalty
        reward -= energy_penalty

        # Bonus, wenn die Pose nah an der Referenz ist.
        reward += 5.0 * math.exp(-joint_error)
        reward += 2.0 * math.exp(-root_pos_error * 3.0)

        # Nicht komplett in den Boden fallen.
        if qpos[2] < 0.20:
            reward -= 100.0
            terminated = True

        if self.frame_id >= self.max_episode_steps:
            truncated = True

        info["phase"] = phase
        info["joint_error"] = float(joint_error)
        info["root_pos_error"] = float(root_pos_error)
        return obs, reward, terminated, truncated, info


def train(timesteps):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DummyVecEnv([lambda: BurpeeImitationEnv()])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.97,
        gae_lambda=0.92,
        ent_coef=0.01,
        tensorboard_log="logs/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATH)
    env.close()
    print(f"Modell gespeichert: {MODEL_PATH}.zip")


def play_learned():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print("Kein trainiertes Modell gefunden. Erst ausfuehren:")
        print("python burpee_mocap_imitation_v3.py --train --timesteps 200000")
        return

    env = BurpeeImitationEnv(render_mode="rgb_array")
    model = PPO.load(MODEL_PATH)
    obs, info = env.reset()
    frames = []

    for _ in range(env.mocap.total_frames * 2):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, info = env.reset()

    filename = datetime.now().strftime("videos/learned_mocap_burpee_v3_%Y%m%d_%H%M%S.mp4")
    write_video(filename, frames, fps=30)
    env.close()
    print(f"Gelerntes Video gespeichert: {filename}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-only", action="store_true", help="Nur die Referenzbewegung rendern")
    parser.add_argument("--train", action="store_true", help="PPO auf die Referenzbewegung trainieren")
    parser.add_argument("--play", action="store_true", help="Trainiertes Modell als Video rendern")
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()

    if args.demo_only:
        render_mocap_demo()
    elif args.train:
        render_mocap_demo()
        train(args.timesteps)
        play_learned()
    elif args.play:
        play_learned()
    else:
        render_mocap_demo()
        print("\nNaechster Schritt zum Trainieren:")
        print("python burpee_mocap_imitation_v3.py --train --timesteps 200000")
