from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from v5envtrain import BurpeeHumanoidV5Env

MODEL_DIR = Path("trained_models_humanoid_v5")
MODEL_DIR.mkdir(exist_ok=True)

STAGES = [
    {"name": "teacher_full", "root_assist": 0.99, "steps": 200_000, "hold": 0.5, "threshold": 1.0},
    {"name": "teacher_strong", "root_assist": 0.95, "steps": 300_000, "hold": 0.8, "threshold": 0.7},
    {"name": "teacher_medium", "root_assist": 0.80, "steps": 500_000, "hold": 1.0, "threshold": 0.5},
    {"name": "teacher_light", "root_assist": 0.50, "steps": 700_000, "hold": 1.5, "threshold": 0.4},
    {"name": "teacher_none", "root_assist": 0.0, "steps": 1_000_000, "hold": 2.0, "threshold": 0.3},
]

def make_env(root_assist):
    return Monitor(
        BurpeeHumanoidV5Env(
            render_mode=None,
            random_start=False,
            root_assist=root_assist,
            action_scale=0.45,
            hold_seconds=0.5,
        )
    )

def main():
    print("Starte Training...")
    print("="*50)
    
    model = None
    for stage in STAGES:
        print(f"\nStage: {stage['name']}")
        print(f"Root Assist: {stage['root_assist']}")
        print(f"Steps: {stage['steps']}")
        print("-"*30)
        
        env = make_env(stage["root_assist"])
        
        if model is None:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=2,
                n_steps=2048,
                batch_size=256,
                learning_rate=2e-4,
                gamma=0.98,
                gae_lambda=0.94,
                ent_coef=0.003,
                clip_range=0.2,
                tensorboard_log="runs_humanoid_v5",
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=stage["steps"],
            reset_num_timesteps=False,
            progress_bar=True,
        )
        stage_path = MODEL_DIR / f"humanoid_v5_{stage['name']}"
        model.save(stage_path)
        print(f"Gespeichert: {stage_path}.zip")

    final_path = MODEL_DIR / "humanoid_v5_curriculum_final"
    model.save(final_path)
    print(f"\nFERTIG! Finales Modell: {final_path}.zip")

if __name__ == "__main__":
    main()