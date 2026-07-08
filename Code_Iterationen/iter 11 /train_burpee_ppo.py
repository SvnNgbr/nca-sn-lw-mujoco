from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from burpee_imitation_env import BurpeeImitationEnv


MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(exist_ok=True)


def make_env():
    env = BurpeeImitationEnv(
        model_path="humanoid_burpee_learning.xml",
        random_start=True,
    )
    return Monitor(env)


def main():
    env = make_env()
    check_env(env.unwrapped, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.97,
        gae_lambda=0.92,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log="runs",
    )

    model.learn(total_timesteps=300_000, progress_bar=True)
    model.save(MODEL_DIR / "burpee_ppo")
    print("Saved trained model to trained_models/burpee_ppo.zip")


if __name__ == "__main__":
    main()
