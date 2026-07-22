import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from imitationv2.burpee_imitation_env_v2 import BurpeeImitationEnvV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="trained_models/burpee_ppo_curriculum.zip",
        help="Path to the trained .zip model.",
    )
    parser.add_argument(
        "--root-assist",
        type=float,
        default=0.45,
        help="Teacher assistance while watching. Try 0.98, 0.75, 0.45, then 0.0.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}")

    env = BurpeeImitationEnvV2(
        model_path="humanoid_burpee_learning.xml",
        random_start=False,
        root_assist=args.root_assist,
    )
    policy = PPO.load(model_path)
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
