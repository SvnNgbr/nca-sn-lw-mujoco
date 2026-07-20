import argparse
import time
from pathlib import Path
import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from v5envtrain import BurpeeHumanoidV5Env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="trained_models_humanoid_v5/humanoid_v5_curriculum_final.zip",
        help="Pfad zum trainierten Modell",
    )
    parser.add_argument(
        "--root-assist",
        type=float,
        default=0.0,
        help="Teacher Assistance (0 = keine Hilfe)",
    )
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        print("Verfuegbare Modelle:")
        for f in Path("trained_models_humanoid_v5").glob("*.zip"):
            print(f"  - {f}")
        return
    
    env = BurpeeHumanoidV5Env(
        render_mode=None,
        random_start=False,
        root_assist=args.root_assist,
    )
    
    policy = PPO.load(model_path)
    obs, _ = env.reset()
    
    print(f"\nModell geladen: {model_path}")
    print(f"Root-Assist: {args.root_assist}")
    print("Viewer wird gestartet...")
    print("Druecken Sie Strg+C zum Beenden\n")
    
    step_count = 0
    total_reward = 0.0
    
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 4.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -18
            viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
            
            while viewer.is_running():
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                step_count += 1
                total_reward += reward
                root_height = info.get('root_height', 0)
                
                # Alle 100 Schritte Info anzeigen
                if step_count % 100 == 0:
                    phase = env._reference_func(env.elapsed)["name"]
                    print(f"Step: {step_count}, Reward: {reward:.2f}, Root: {root_height:.2f}, Phase: {phase}")
                
                viewer.sync()
                time.sleep(env.dt)
                
                if terminated:
                    print(f"Roboter gefallen! Root Height: {root_height:.2f}")
                    obs, _ = env.reset()
                    step_count = 0
                    total_reward = 0.0
                elif truncated:
                    print(f"Episode fertig. Total Reward: {total_reward:.2f}")
                    obs, _ = env.reset()
                    step_count = 0
                    total_reward = 0.0
                    
    except Exception as e:
        print(f"Fehler: {e}")
    
    env.close()

if __name__ == "__main__":
    main()