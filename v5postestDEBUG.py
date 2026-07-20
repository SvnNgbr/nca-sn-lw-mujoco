import gymnasium as gym
import mujoco
import numpy as np
import time
import sys
import select
from v5ref import BURPEE_V5, JOINT_NAMES_V5, deg

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

def apply_pose_stable(model, data, pose, joint_qpos, root_addr, strength=0.95):
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
    env = gym.make("Humanoid-v5", render_mode="human")
    obs, info = env.reset()
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Joints finden
    joint_qpos = []
    for name in JOINT_NAMES_V5:
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_qpos.append(model.jnt_qposadr[joint_id])
        except:
            joint_qpos.append(-1)
    
    valid_qpos = [q for q in joint_qpos if q >= 0]
    
    # Root finden
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_addr = model.jnt_qposadr[root_id]
    
    current_idx = 0
    
    print("\n" + "="*60)
    print("HUMANOID V5 - POSE TESTER")
    print("="*60)
    print("\nVERFUEGBARE POSEN:")
    for i, pose in enumerate(BURPEE_V5):
        print(f"  {i+1}. {pose['name']}")
    print("\nBEDIENUNG:")
    print("  • [Zahl] → Pose auswaehlen")
    print("  • [n] → Naechste Pose")
    print("  • [p] → Vorherige Pose")
    print("  • [q] → Beenden")
    print("="*60)
    
    apply_pose_stable(model, data, BURPEE_V5[0], valid_qpos, root_addr)
    
    try:
        while True:
            # Pose jeden Frame neu anwenden
            apply_pose_stable(model, data, BURPEE_V5[current_idx], valid_qpos, root_addr)
            
            env.render()
            mujoco.mj_step(model, data)
            
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'n':
                    current_idx = (current_idx + 1) % len(BURPEE_V5)
                    print(f"\nPose {current_idx+1}: {BURPEE_V5[current_idx]['name']}")
                elif cmd == 'p':
                    current_idx = (current_idx - 1) % len(BURPEE_V5)
                    print(f"\nPose {current_idx+1}: {BURPEE_V5[current_idx]['name']}")
                elif cmd.isdigit():
                    num = int(cmd) - 1
                    if 0 <= num < len(BURPEE_V5):
                        current_idx = num
                        print(f"\nPose {current_idx+1}: {BURPEE_V5[current_idx]['name']}")
                    else:
                        print(f"Pose {num+1} existiert nicht")
            
            time.sleep(model.opt.timestep * 5)
            
    except KeyboardInterrupt:
        print("\nBeende...")
    finally:
        env.close()

if __name__ == "__main__":
    main()