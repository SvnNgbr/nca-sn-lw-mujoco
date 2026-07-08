import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from burpee_imitation_env import BurpeeImitationEnv


MODEL_PATH = Path("trained_models/burpee_ppo.zip")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run first: python train_burpee_ppo.py"
        )

    env = BurpeeImitationEnv(
        model_path="humanoid_burpee_learning.xml",
        random_start=False,
    )
    policy = PPO.load(MODEL_PATH)
    obs, _ = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = [0.2, 0.0, 0.65]

        while viewer.is_running():
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            time.sleep(env.dt)
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
