import gymnasium as gym
import mujoco
import numpy as np
import time
from PIL import Image
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
    # Video Einstellungen
    WIDTH = 800
    HEIGHT = 600
    FPS = 30
    TOTAL_DURATION = 8.0  # 3 Durchläufe a ca. 2.6s
    FRAME_INTERVAL = 1.0 / FPS
    
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
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
    
    # Gesamtdauer einer Pose
    total_pose_duration = sum(p["duration"] for p in BURPEE_V5)
    total_video_duration = total_pose_duration * 3  # 3 Durchläufe
    
    print(f"Video Laenge: {total_video_duration:.1f} Sekunden")
    print(f"Frame Rate: {FPS} FPS")
    print(f"Gesamt Frames: {int(total_video_duration * FPS)}")
    print("Erstelle Video... (das kann etwas dauern)")
    
    frames = []
    t = 0.0
    
    # 3 Durchläufe
    for run in range(3):
        print(f"Durchlauf {run+1}/3")
        
        # Reset am Anfang jedes Durchlaufs
        obs, info = env.reset()
        
        # Pose durchlaufen
        while t < total_pose_duration:
            # Aktuelle Pose berechnen
            cursor = 0.0
            for idx, keyframe in enumerate(BURPEE_V5):
                next_keyframe = BURPEE_V5[(idx + 1) % len(BURPEE_V5)]
                segment_duration = keyframe["duration"]
                
                if cursor <= t < cursor + segment_duration:
                    alpha = (t - cursor) / segment_duration
                    
                    # Pose interpolieren
                    pose = {
                        "name": keyframe["name"],
                        "root_pos": (1 - alpha) * np.array(keyframe["root_pos"]) + alpha * np.array(next_keyframe["root_pos"]),
                        "root_euler": (1 - alpha) * np.array(keyframe["root_euler"]) + alpha * np.array(next_keyframe["root_euler"]),
                        "joints": (1 - alpha) * np.array(keyframe["joints"]) + alpha * np.array(next_keyframe["joints"])
                    }
                    
                    # Pose anwenden
                    apply_pose_stable(model, data, pose, valid_qpos, root_addr)
                    
                    # Render Frame
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    
                    # MuJoCo Step
                    for _ in range(5):
                        mujoco.mj_step(model, data)
                    
                    break
                cursor += segment_duration
            
            t += FRAME_INTERVAL
            
            # Fortschritt anzeigen
            if len(frames) % 100 == 0:
                progress = (run * total_pose_duration + t) / (3 * total_pose_duration) * 100
                print(f"  Progress: {progress:.1f}%")
        
        t = 0.0  # Reset für nächsten Durchlauf
    
    # Video speichern
    print(f"\nSpeichere Video mit {len(frames)} Frames...")
    
    if frames:
        from PIL import Image
        frames_pil = [Image.fromarray(frame) for frame in frames]
        frames_pil[0].save(
            "burpee_v5.gif",
            save_all=True,
            append_images=frames_pil[1:],
            duration=int(1000 / FPS),
            loop=0
        )
        print("Video gespeichert als: burpee_v5.gif")
    else:
        print("Keine Frames erstellt!")
    
    env.close()

if __name__ == "__main__":
    main()