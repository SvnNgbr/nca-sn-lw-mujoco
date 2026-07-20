import argparse
import time
from pathlib import Path
import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from v5envtrain import BurpeeHumanoidV5Env
import threading

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="trained_models_humanoid_v5/humanoid_v5_ppo_teacher_strong.zip",
        help="Pfad zum trainierten Modell",
    )
    parser.add_argument(
        "--root-assist",
        type=float,
        default=0.0,
        help="Teacher Assistance",
    )
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        return
    
    env = BurpeeHumanoidV5Env(
        render_mode=None,  # Kein eigenes Rendering
        random_start=False,
        root_assist=args.root_assist,
    )
    
    policy = PPO.load(model_path)
    obs, _ = env.reset()
    
    # Viewer im Hauptthread starten
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        
        print("Drücken Sie Strg+C zum Beenden")
        print(f"Root-Assist: {args.root_assist}")
        
        try:
            while viewer.is_running():
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                viewer.sync()
                time.sleep(env.dt)
                
                if terminated or truncated:
                    obs, _ = env.reset()
        except KeyboardInterrupt:
            print("\nBeende...")
        finally:
            env.close()

if __name__ == "__main__":
    main()