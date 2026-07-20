import gymnasium as gym
import mujoco
import numpy as np
import time
import sys
import select
from v5ref import BURPEE_V5, JOINT_NAMES_V5, deg

# give ref pos and allows manipulation of joints in edit mode

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
    env = gym.make("Humanoid-v5", render_mode="human")
    obs, info = env.reset()
    
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Alle Joints finden
    joint_names = []
    joint_qpos = []
    
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            joint_names.append(name)
            joint_qpos.append(model.jnt_qposadr[i])
    
    # Root-Joint finden
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_addr = model.jnt_qposadr[root_id]
    
    # Aktuelle Pose und Joint-Winkel
    current_pose_idx = 0
    joint_angles = [0.0] * len(joint_names)
    current_joint = 0
    
    # Ob wir in der Pose oder im Edit-Modus sind
    edit_mode = False
    
    def apply_pose(pose_idx):
        """Wendet eine Pose aus BURPEE_V5 an"""
        pose = BURPEE_V5[pose_idx]
        joints_rad = deg(pose["joints"])
        
        # Joint-Winkel setzen
        for i, qpos_addr in enumerate(joint_qpos):
            if i < len(joints_rad):
                data.qpos[qpos_addr] = joints_rad[i]
                joint_angles[i] = np.degrees(joints_rad[i])
        
        # Root-Position setzen
        if "root_pos" in pose:
            data.qpos[root_addr:root_addr+3] = pose["root_pos"]
        if "root_euler" in pose:
            root_quat = quat_from_euler_xyz(pose["root_euler"])
            data.qpos[root_addr+3:root_addr+7] = root_quat
        
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        
        show_info(pose_idx)
    
    def apply_current_joints():
        """Wendet die aktuellen Joint-Winkel an"""
        for i, qpos_addr in enumerate(joint_qpos):
            data.qpos[qpos_addr] = np.radians(joint_angles[i])
        
        # Roboter in der Luft halten
        data.qpos[root_addr:root_addr+3] = [0.0, 0.0, 1.5]
        data.qpos[root_addr+3:root_addr+7] = [1, 0, 0, 0]
        data.qvel[:] = 0.0
        
        mujoco.mj_forward(model, data)
    
    def show_info(pose_idx=None):
        print("\n" + "="*70)
        if pose_idx is not None and not edit_mode:
            pose = BURPEE_V5[pose_idx]
            print(f"POSE {pose_idx+1}/{len(BURPEE_V5)}: {pose['name']}")
            print(f"Root Pos: {pose['root_pos']}")
            print(f"Root Euler: {pose['root_euler']}")
        else:
            print("EDIT MODUS - Eigene Joint-Winkel")
        
        print("-"*70)
        print("GELENKE (in Grad):")
        for i, name in enumerate(joint_names):
            marker = " <--" if i == current_joint else ""
            print(f"  {i:2d}. {name:20s} : {joint_angles[i]:8.1f}°{marker}")
        
        if edit_mode:
            print("-"*70)
            print(f"AKTUELLES GELENK: {joint_names[current_joint]}")
            print(f"WINKEL: {joint_angles[current_joint]:.1f}°")
        print("="*70)
    
    def show_menu():
        print("\n" + "="*70)
        print("HUMANOID V5 - POSE + JOINT EDITOR")
        print("="*70)
        print("\nVERFUEGBARE POSEN:")
        for i, pose in enumerate(BURPEE_V5):
            print(f"  {i+1}. {pose['name']}")
        print("\n" + "="*70)
        print("BEDIENUNG:")
        print("  • [Zahl 1-9] → Pose auswaehlen")
        print("  • [e] → Edit-Modus (Joint-Winkel aendern)")
        print("  • [p] → Pose-Modus (zurueck zur Pose)")
        print("  • [z+Zahl] → Joint auswaehlen (z.B. 'z7')")
        print("  • [w] → Winkel +5°")
        print("  • [s] → Winkel -5°")
        print("  • [W] → Winkel +1° (fein)")
        print("  • [S] → Winkel -1° (fein)")
        print("  • [r] → Alle Gelenke auf 0")
        print("  • [q] → Beenden")
        print("="*70)
    
    show_menu()
    
    # Erste Pose laden
    apply_pose(0)
    
    try:
        while True:
            # Jeden Frame die aktuellen Joints anwenden
            if edit_mode:
                apply_current_joints()
            else:
                # Pose neu anwenden (hält die Pose stabil)
                pose = BURPEE_V5[current_pose_idx]
                joints_rad = deg(pose["joints"])
                for i, qpos_addr in enumerate(joint_qpos):
                    if i < len(joints_rad):
                        data.qpos[qpos_addr] = joints_rad[i]
                        joint_angles[i] = np.degrees(joints_rad[i])
                if "root_pos" in pose:
                    data.qpos[root_addr:root_addr+3] = pose["root_pos"]
                if "root_euler" in pose:
                    root_quat = quat_from_euler_xyz(pose["root_euler"])
                    data.qpos[root_addr+3:root_addr+7] = root_quat
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            
            env.render()
            mujoco.mj_step(model, data)
            
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'e':
                    edit_mode = True
                    print("\n>>> EDIT-MODUS AKTIVIERT <<<")
                    show_info()
                elif cmd == 'p':
                    edit_mode = False
                    print("\n>>> POSE-MODUS AKTIVIERT <<<")
                    apply_pose(current_pose_idx)
                elif cmd == 'r':
                    joint_angles = [0.0] * len(joint_names)
                    if edit_mode:
                        apply_current_joints()
                        show_info()
                    else:
                        # In Pose-Modus: Pose zurücksetzen
                        apply_pose(current_pose_idx)
                elif cmd == 'w':
                    if edit_mode:
                        joint_angles[current_joint] += 5.0
                        apply_current_joints()
                        show_info()
                elif cmd == 's':
                    if edit_mode:
                        joint_angles[current_joint] -= 5.0
                        apply_current_joints()
                        show_info()
                elif cmd == 'W':  # Shift+w
                    if edit_mode:
                        joint_angles[current_joint] += 1.0
                        apply_current_joints()
                        show_info()
                elif cmd == 'S':  # Shift+s
                    if edit_mode:
                        joint_angles[current_joint] -= 1.0
                        apply_current_joints()
                        show_info()
                elif cmd.startswith('z'):
                    if edit_mode:
                        try:
                            num = int(cmd[1:])
                            if 0 <= num < len(joint_names):
                                current_joint = num
                                show_info()
                            else:
                                print(f"Joint {num} existiert nicht (0-{len(joint_names)-1})")
                        except:
                            print("Ungueltig! Verwende: z0, z1, z2, ...")
                elif cmd.isdigit():
                    num = int(cmd) - 1
                    if 0 <= num < len(BURPEE_V5):
                        current_pose_idx = num
                        edit_mode = False
                        apply_pose(current_pose_idx)
                    else:
                        print(f"Pose {num+1} existiert nicht (1-{len(BURPEE_V5)})")
                elif cmd:
                    print("Unbekannter Befehl. 'h' fuer Hilfe.")
            
            time.sleep(model.opt.timestep * 5)
            
    except KeyboardInterrupt:
        print("\nBeende...")
    finally:
        env.close()

if __name__ == "__main__":
    main()