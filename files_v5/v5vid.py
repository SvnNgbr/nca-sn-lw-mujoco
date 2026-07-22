import gymnasium as gym
import mujoco
import numpy as np
import time
import cv2
import os
from v5.v5ref import BURPEE_V5, JOINT_NAMES_V5, deg

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

def apply_pose_stable(model, data, pose, joint_qpos, root_addr):
    joints_rad = deg(pose["joints"])
    
    for i, qpos_addr in enumerate(joint_qpos):
        if i < len(joints_rad):
            data.qpos[qpos_addr] = joints_rad[i]
    
    if "root_pos" in pose:
        data.qpos[root_addr:root_addr+3] = pose["root_pos"]
    if "root_euler" in pose:
        root_quat = quat_from_euler_xyz(pose["root_euler"])
        data.qpos[root_addr+3:root_addr+7] = root_quat
    
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

def main():
    # Video settings
    FPS = 30
    RUNS = 3  # Number of burpee cycles
    FRAME_INTERVAL = 1.0 / FPS
    
    # Create videos directory
    os.makedirs("videos", exist_ok=True)
    video_name = time.strftime("videos/burpee_v5_%Y%m%d_%H%M%S.mp4")
    
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    obs, info = env.reset()
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Find joints
    joint_qpos = []
    for name in JOINT_NAMES_V5:
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_qpos.append(model.jnt_qposadr[joint_id])
        except:
            joint_qpos.append(-1)
    
    valid_qpos = [q for q in joint_qpos if q >= 0]
    
    # Find root
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_addr = model.jnt_qposadr[root_id]
    
    # Total duration of one cycle
    total_pose_duration = sum(p["duration"] for p in BURPEE_V5)
    total_video_duration = total_pose_duration * RUNS
    
    print(f"Video duration: {total_video_duration:.1f} seconds")
    print(f"Frame rate: {FPS} FPS")
    print(f"Total frames: {int(total_video_duration * FPS)}")
    print(f"Output: {video_name}")
    print("Generating video... (this may take a moment)")
    
    # Get first frame for video dimensions
    frame = env.render()
    if frame is None:
        print("Error: Could not get frame from environment.")
        env.close()
        return
    
    height, width = frame.shape[:2]
    
    # Initialize video writer
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )
    
    frame_count = 0
    t = 0.0
    
    # Run through the burpee sequence RUNS times
    for run in range(RUNS):
        print(f"Run {run+1}/{RUNS}")
        
        # Reset at start of each run
        obs, info = env.reset()
        
        # Run through one full cycle
        while t < total_pose_duration:
            # Find current pose
            cursor = 0.0
            for idx, keyframe in enumerate(BURPEE_V5):
                next_keyframe = BURPEE_V5[(idx + 1) % len(BURPEE_V5)]
                segment_duration = keyframe["duration"]
                
                if cursor <= t < cursor + segment_duration:
                    alpha = (t - cursor) / segment_duration
                    
                    # Interpolate pose
                    pose = {
                        "name": keyframe["name"],
                        "root_pos": (1 - alpha) * np.array(keyframe["root_pos"]) + alpha * np.array(next_keyframe["root_pos"]),
                        "root_euler": (1 - alpha) * np.array(keyframe["root_euler"]) + alpha * np.array(next_keyframe["root_euler"]),
                        "joints": (1 - alpha) * np.array(keyframe["joints"]) + alpha * np.array(next_keyframe["joints"])
                    }
                    
                    # Apply pose
                    apply_pose_stable(model, data, pose, valid_qpos, root_addr)
                    
                    # Render frame
                    frame = env.render()
                    if frame is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video.write(frame_bgr)
                        frame_count += 1
                    
                    # MuJoCo step
                    for _ in range(5):
                        mujoco.mj_step(model, data)
                    
                    break
                cursor += segment_duration
            
            t += FRAME_INTERVAL
            
            # Show progress
            if frame_count % 100 == 0:
                progress = (run * total_pose_duration + t) / (RUNS * total_pose_duration) * 100
                print(f"  Progress: {progress:.1f}% (frames: {frame_count})")
        
        t = 0.0  # Reset for next run
    
    # Cleanup
    video.release()
    env.close()
    
    print(f"\nVideo saved: {video_name}")
    print(f"Total frames: {frame_count}")
    print(f"File size: {os.path.getsize(video_name) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()