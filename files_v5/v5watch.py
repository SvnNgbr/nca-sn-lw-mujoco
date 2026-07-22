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
        help="Path to trained model",
    )
    parser.add_argument(
        "--root-assist",
        type=float,
        default=0.0,
        help="Teacher assistance",
    )
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    env = BurpeeHumanoidV5Env(
        render_mode=None,
        random_start=False,
        root_assist=args.root_assist,
        hold_seconds=0.5,
    )
    
    policy = PPO.load(model_path)
    obs, _ = env.reset()
    
    print(f"\nModel loaded: {model_path}")
    print(f"Root assist: {args.root_assist}")
    print("Starting viewer...")
    print("Press Ctrl+C to stop\n")
    
    step_count = 0
    total_reward = 0.0
    episode_count = 0
    
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
                phase = info.get('phase', 'unknown')
                fallen = info.get('fallen', False)
                
                if step_count % 50 == 0:
                    status = "FALLEN" if fallen else "Standing"
                    print(f"Step: {step_count}, Phase: {phase}, Root: {root_height:.2f}, {status}, Reward: {reward:.2f}")
                
                viewer.sync()
                time.sleep(env.dt)
                
                if terminated:
                    episode_count += 1
                    print(f"Episode {episode_count} ended (robot on ground). Total reward: {total_reward:.2f}")
                    obs, _ = env.reset()
                    step_count = 0
                    total_reward = 0.0
                elif truncated:
                    episode_count += 1
                    print(f"Episode {episode_count} complete! Total reward: {total_reward:.2f}")
                    obs, _ = env.reset()
                    step_count = 0
                    total_reward = 0.0
                    
    except Exception as e:
        print(f"Error: {e}")
    
    env.close()

if __name__ == "__main__":
    main()