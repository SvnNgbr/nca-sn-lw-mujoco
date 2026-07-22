import gymnasium as gym
import mujoco
import numpy as np
import time
import cv2
import os
import argparse
from stable_baselines3 import PPO
from v5ref import BURPEE_V5, JOINT_NAMES_V5, deg
from v5envtrain import BurpeeHumanoidV5Env

def quat_from_euler_xyz(euler_deg):
    roll, pitch, yaw = np.radians(euler_deg)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--root-assist", type=float, default=0.0, help="Root assist for video")
    parser.add_argument("--runs", type=int, default=3, help="Number of burpee cycles")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()
    
    # Video settings
    FPS = args.fps
    RUNS = args.runs
    FRAME_INTERVAL = 1.0 / FPS
    
    os.makedirs("videos", exist_ok=True)
    video_name = time.strftime("videos/burpee_v5_policy_%Y%m%d_%H%M%S.mp4")
    
    print(f"Model: {args.model}")
    print(f"Root assist: {args.root_assist}")
    print(f"Runs: {RUNS}")
    print(f"Output: {video_name}")
    print("Generating video...")
    
    # Environment MIT render_mode="rgb_array" für Video
    env = BurpeeHumanoidV5Env(
        render_mode="rgb_array",  # <-- WICHTIG: rgb_array für Video
        random_start=False,
        root_assist=args.root_assist,
        hold_seconds=0.5,
    )
    
    policy = PPO.load(args.model)
    obs, _ = env.reset()
    
    # Erstes Frame für Dimensionen
    frame = env.render()
    if frame is None:
        print("Error: Could not get frame. Trying alternative...")
        # Fallback: Direkt mit gym.make
        env.close()
        env = gym.make("Humanoid-v5", render_mode="rgb_array")
        obs, _ = env.reset()
        frame = env.render()
        if frame is None:
            print("Error: Still could not get frame.")
            env.close()
            return
    
    height, width = frame.shape[:2]
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )
    
    frame_count = 0
    total_reward = 0.0
    episode_count = 0
    
    try:
        while episode_count < RUNS:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            frame = env.render()
            
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)
                frame_count += 1
            else:
                print(f"Warning: No frame at step {frame_count}")
            
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count}/{RUNS} complete. Reward: {total_reward:.2f}")
                obs, _ = env.reset()
                total_reward = 0.0
            
            time.sleep(FRAME_INTERVAL)
            
    except KeyboardInterrupt:
        print("Interrupted...")
    finally:
        video.release()
        env.close()
    
    if frame_count > 0:
        file_size = os.path.getsize(video_name) / (1024*1024)
        print(f"\nVideo saved: {video_name}")
        print(f"Total frames: {frame_count}")
        print(f"File size: {file_size:.1f} MB")
    else:
        print("\nNo frames were recorded. Video generation failed.")

if __name__ == "__main__":
    main()