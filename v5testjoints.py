import gymnasium as gym
import mujoco
import numpy as np
import time
import sys
import select

# allows to test joints

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
    
    # Roboter in die Luft setzen (schweben)
    data.qpos[root_addr:root_addr+3] = [0.0, 0.0, 1.5]  # Höhe
    data.qpos[root_addr+3:root_addr+7] = [1, 0, 0, 0]   # Keine Rotation
    data.qvel[:] = 0.0
    
    # Alle Gelenke auf 0 setzen
    for qpos_addr in joint_qpos:
        data.qpos[qpos_addr] = 0.0
    
    mujoco.mj_forward(model, data)
    
    print("\n" + "="*70)
    print("HUMANOID V5 - JOINT TESTER (GEFROHREN IN DER LUFT)")
    print("="*70)
    print("\nDer Roboter wird JEDEN FRAME in der Position gehalten!")
    print("\nGELENKE:")
    print("-"*70)
    for i, name in enumerate(joint_names):
        print(f"  {i:2d}. {name}")
    print("-"*70)
    print("\nBEDIENUNG:")
    print("  • [Zahl] → Gelenk auswaehlen (0-{})".format(len(joint_names)-1))
    print("  • [w]    → Winkel +5°")
    print("  • [s]    → Winkel -5°")
    print("  • [W]    → Winkel +1° (fein)")
    print("  • [S]    → Winkel -1° (fein)")
    print("  • [r]    → Alle Gelenke auf 0 zuruecksetzen")
    print("  • [q]    → Beenden")
    print("="*70)
    
    current_joint = 0
    joint_angles = [0.0] * len(joint_names)
    
    def apply_all_joints():
        # ALLE Gelenke JEDEN FRAME setzen
        for i, qpos_addr in enumerate(joint_qpos):
            data.qpos[qpos_addr] = np.radians(joint_angles[i])
        
        # Roboter in der Luft halten (jeden Frame!)
        data.qpos[root_addr:root_addr+3] = [0.0, 0.0, 1.5]
        data.qpos[root_addr+3:root_addr+7] = [1, 0, 0, 0]
        
        # Geschwindigkeiten auf 0 setzen (keine Bewegung)
        data.qvel[:] = 0.0
        
        mujoco.mj_forward(model, data)
    
    def show_info():
        print("\n" + "="*70)
        print(f"GELENK {current_joint}: {joint_names[current_joint]}")
        print(f"WINKEL: {joint_angles[current_joint]:.1f}°")
        print("-"*70)
        print("ALLE GELENKE (in Grad):")
        for i, name in enumerate(joint_names):
            marker = " <--" if i == current_joint else ""
            print(f"  {i:2d}. {name:20s} : {joint_angles[i]:8.1f}°{marker}")
        print("="*70)
    
    # Initial anwenden
    apply_all_joints()
    show_info()
    
    try:
        while True:
            # JEDEN FRAME die Pose neu setzen!
            apply_all_joints()
            
            env.render()
            mujoco.mj_step(model, data)
            
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'r':
                    joint_angles = [0.0] * len(joint_names)
                    apply_all_joints()
                    show_info()
                elif cmd == 'w':
                    joint_angles[current_joint] += 5.0
                    apply_all_joints()
                    show_info()
                elif cmd == 's':
                    joint_angles[current_joint] -= 5.0
                    apply_all_joints()
                    show_info()
                elif cmd == 'W':  # Großbuchstabe W (Shift+w)
                    joint_angles[current_joint] += 1.0
                    apply_all_joints()
                    show_info()
                elif cmd == 'S':  # Großbuchstabe S (Shift+s)
                    joint_angles[current_joint] -= 1.0
                    apply_all_joints()
                    show_info()
                elif cmd.isdigit():
                    num = int(cmd)
                    if 0 <= num < len(joint_names):
                        current_joint = num
                        apply_all_joints()
                        show_info()
                    else:
                        print(f"Gelenk {num} existiert nicht (0-{len(joint_names)-1})")
                elif cmd:
                    print("Unbekannter Befehl")
            
            time.sleep(model.opt.timestep * 5)
            
    except KeyboardInterrupt:
        print("\nBeende...")
    finally:
        env.close()

if __name__ == "__main__":
    main()