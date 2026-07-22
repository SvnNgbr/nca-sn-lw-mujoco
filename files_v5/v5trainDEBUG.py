from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from v5.v5envtrain import BurpeeHumanoidV5Env

MODEL_DIR = Path("trained_models_humanoid_v5")
MODEL_DIR.mkdir(exist_ok=True)

# Nur EINE Stage für den Test
STAGES = [
    {"name": "test", "root_assist": 0.95, "steps": 10_000},
]

def make_env(root_assist):
    return Monitor(
        BurpeeHumanoidV5Env(
            render_mode=None,
            random_start=False,
            root_assist=root_assist,
            action_scale=0.45,
            hold_seconds=0.5,  # Kurze Haltezeit für Test
        )
    )

def main():
    print("Starte Training...")
    print("="*50)
    
    for stage in STAGES:
        print(f"\nStage: {stage['name']}")
        print(f"Root Assist: {stage['root_assist']}")
        print(f"Steps: {stage['steps']}")
        print("-"*30)
        
        env = make_env(stage["root_assist"])
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=512,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            tensorboard_log="runs_v5_test",
        )
        
        print("Lerne...")
        model.learn(
            total_timesteps=stage["steps"],
            progress_bar=True,
        )
        
        stage_path = MODEL_DIR / f"humanoid_v5_{stage['name']}"
        model.save(stage_path)
        print(f"Gespeichert: {stage_path}.zip")
        print("="*50)

if __name__ == "__main__":
    main()