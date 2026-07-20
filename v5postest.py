import gymnasium as gym
import mujoco
import numpy as np
import time
import sys
import select
from v5ref import BURPEE_V5, JOINT_NAMES_V5, deg

#allows to preview poses 

def quat_from_euler_xyz(euler_deg):
    """Konvertiert Euler-Winkel in Quaternion"""
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

def apply_pose_stable(model, data, pose, joint_qpos, root_qpos_addr, strength=0.8):
    """Wendet eine Pose an mit Stabilisierung"""
    joints_rad = deg(pose["joints"])
    
    # Ziel-Position
    target_root_pos = np.array(pose["root_pos"])
    target_root_quat = quat_from_euler_xyz(pose["root_euler"])
    target_joints = joints_rad
    
    # Aktuelle Position
    current_root_pos = data.qpos[root_qpos_addr:root_qpos_addr+3].copy()
    current_root_quat = data.qpos[root_qpos_addr+3:root_qpos_addr+7].copy()
    current_joints = np.array([data.qpos[addr] for addr in joint_qpos if addr >= 0])
    
    # Sanft zur Zielposition interpolieren
    if strength > 0:
        new_root_pos = (1 - strength) * current_root_pos + strength * target_root_pos
        data.qpos[root_qpos_addr:root_qpos_addr+3] = new_root_pos
        
        new_root_quat = (1 - strength) * current_root_quat + strength * target_root_quat
        new_root_quat = new_root_quat / np.linalg.norm(new_root_quat)
        data.qpos[root_qpos_addr+3:root_qpos_addr+7] = new_root_quat
        
        for i, qpos_addr in enumerate(joint_qpos):
            if i < len(target_joints) and qpos_addr >= 0:
                current_val = data.qpos[qpos_addr]
                target_val = target_joints[i]
                data.qpos[qpos_addr] = (1 - strength) * current_val + strength * target_val
        
        data.qvel[:] = data.qvel[:] * (1 - strength * 0.5)
    
    mujoco.mj_forward(model, data)

def show_pose(pose_idx, env, model, data, joint_qpos, joint_names, root_qpos_addr, strength=0.8):
    """Zeigt eine einzelne Pose an"""
    pose = BURPEE_V5[pose_idx]
    apply_pose_stable(model, data, pose, joint_qpos, root_qpos_addr, strength)
    
    joints_rad = deg(pose["joints"])
    print("\n" + "="*60)
    print(f"POSITION {pose_idx+1}/{len(BURPEE_V5)}: {pose['name']}")
    print(f"Root Position: {pose['root_pos']}")
    print(f"Root Euler: {pose['root_euler']}")
    print("-"*60)
    print("Gelenkwinkel (in Grad):")
    for i, (name, val) in enumerate(zip(joint_names, np.degrees(joints_rad))):
        print(f"  {name:20s} : {val:8.1f}°")
    print("="*60)

def main():
    STRENGTH = 0.95  # Sehr starke Stabilisierung
    
    env = gym.make("Humanoid-v5", render_mode="human")
    obs, info = env.reset()
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Root-Joint finden
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qpos_addr = model.jnt_qposadr[root_joint_id]
    
    # Joint-QPOS-Adressen finden
    joint_qpos = []
    for name in JOINT_NAMES_V5:
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_qpos.append(model.jnt_qposadr[joint_id])
        except:
            print(f"WARNING: Joint {name} nicht gefunden!")
            joint_qpos.append(-1)
    
    valid_qpos = [q for q in joint_qpos if q >= 0]
    valid_names = [name for name, q in zip(JOINT_NAMES_V5, joint_qpos) if q >= 0]
    
    print("\n" + "="*60)
    print("HUMANOID V5 - BURPEE KEYFRAME TESTER")
    print("="*60)
    print("\nZeigt NUR die Keyframes (keine Interpolation zwischen den Posen)")
    print(f"\n{len(BURPEE_V5)} Positionen:")
    for i, pose in enumerate(BURPEE_V5):
        print(f"  {i+1}. {pose['name']}")
    
    print("\n" + "="*60)
    print("BEDIENUNG:")
    print("  • [Enter] → Naechste Position (springt zur Pose)")
    print("  • [p]     → Vorherige Position")
    print("  • [z+Zahl] → Zu Position (z.B. 'z5')")
    print("  • [q]     → Beenden")
    print("="*60)
    
    current_idx = 0
    show_pose(current_idx, env, model, data, valid_qpos, valid_names, root_qpos_addr, STRENGTH)
    
    try:
        while True:
            # Jeden Frame die gleiche Pose anwenden (keine Interpolation!)
            pose = BURPEE_V5[current_idx]
            apply_pose_stable(model, data, pose, valid_qpos, root_qpos_addr, STRENGTH)
            
            env.render()
            mujoco.mj_step(model, data)
            
            # Tastatureingabe
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == '' or cmd == 'n':
                    current_idx = (current_idx + 1) % len(BURPEE_V5)
                    show_pose(current_idx, env, model, data, valid_qpos, valid_names, root_qpos_addr, STRENGTH)
                elif cmd == 'p':
                    current_idx = (current_idx - 1) % len(BURPEE_V5)
                    show_pose(current_idx, env, model, data, valid_qpos, valid_names, root_qpos_addr, STRENGTH)
                elif cmd.startswith('z'):
                    try:
                        num = int(cmd[1:]) - 1
                        if 0 <= num < len(BURPEE_V5):
                            current_idx = num
                            show_pose(current_idx, env, model, data, valid_qpos, valid_names, root_qpos_addr, STRENGTH)
                        else:
                            print(f"Position {num+1} existiert nicht")
                    except:
                        print("Ungueltig! Verwende: z1, z2, ...")
                elif cmd:
                    print("Unbekannter Befehl.")
            
            time.sleep(model.opt.timestep * 5)
            
    except KeyboardInterrupt:
        print("\nBeende...")
    finally:
        env.close()

if __name__ == "__main__":
    main()