from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from burpee_imitation_env_v3_physical import BurpeeImitationEnvV3Physical


MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(exist_ok=True)


STAGES = [
    {
        "name": "pose_and_contacts",
        "steps": 500_000,
        "random_start": False,
        "weak_root_assist": 0.12,
        "action_scale": 0.30,
    },
    {
        "name": "less_assist",
        "steps": 800_000,
        "random_start": True,
        "weak_root_assist": 0.04,
        "action_scale": 0.40,
    },
    {
        "name": "physical_final",
        "steps": 1_200_000,
        "random_start": True,
        "weak_root_assist": 0.0,
        "action_scale": 0.45,
    },
]


def make_env(stage):
    return Monitor(
        BurpeeImitationEnvV3Physical(
            model_path="humanoid_burpee_v3.xml",
            random_start=stage["random_start"],
            weak_root_assist=stage["weak_root_assist"],
            action_scale=stage["action_scale"],
        )
    )


def main():
    model = None
    for stage in STAGES:
        print(
            f"Training {stage['name']} "
            f"(steps={stage['steps']}, weak_root_assist={stage['weak_root_assist']})"
        )
        env = make_env(stage)
        if model is None:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                n_steps=2048,
                batch_size=256,
                learning_rate=2e-4,
                gamma=0.98,
                gae_lambda=0.94,
                ent_coef=0.004,
                clip_range=0.2,
                tensorboard_log="runs_v3_physical",
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=stage["steps"],
            reset_num_timesteps=False,
            progress_bar=True,
        )
        path = MODEL_DIR / f"burpee_v3_{stage['name']}"
        model.save(path)
        print(f"Saved {path}.zip")

    final_path = MODEL_DIR / "burpee_v3_physical"
    model.save(final_path)
    print(f"Saved final model to {final_path}.zip")


if __name__ == "__main__":
    main()
