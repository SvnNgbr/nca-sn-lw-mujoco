import math
import numpy as np

# Humanoid v5 hat 17 Gelenke (nicht 21!)
JOINT_NAMES_V5 = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "torso_joint",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "right_shoulder_pitch",
    "right_shoulder_roll"
]

# Burpee mit 17 Gelenkwerten
BURPEE_V5 = [
    {
        "name": "stand",
        "duration": 1.0,
        "root_pos": [0.0, 0.0, 1.4],
        "root_euler": [0, 0, 0],
        "joints": [
            0, 0, 0,   # left_hip_pitch, roll, yaw
            0, 0, 0,   # left_knee, ankle_pitch, ankle_roll
            0, 0, 0,   # right_hip_pitch, roll, yaw
            0, 0, 0,   # right_knee, ankle_pitch, ankle_roll
            0,         # torso_joint
            0, 0,      # left_shoulder_pitch, roll
            0, 0       # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "squat",
        "duration": 1.0,
        "root_pos": [0.0, 0.0, 0.9],
        "root_euler": [0, 20, 0],
        "joints": [
            -30, 0, 0,  # left_hip_pitch, roll, yaw
            60, -20, 0, # left_knee, ankle_pitch, ankle_roll
            -30, 0, 0,  # right_hip_pitch, roll, yaw
            60, -20, 0, # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            10, 5,      # left_shoulder_pitch, roll
            10, -5      # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "hands_to_floor",
        "duration": 0.8,
        "root_pos": [0.15, 0.0, 0.7],
        "root_euler": [0, 60, 0],
        "joints": [
            -60, 0, 0,  # left_hip_pitch, roll, yaw
            90, -30, 0, # left_knee, ankle_pitch, ankle_roll
            -60, 0, 0,  # right_hip_pitch, roll, yaw
            90, -30, 0, # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            -90, 20,    # left_shoulder_pitch, roll
            -90, -20    # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "plank",
        "duration": 1.1,
        "root_pos": [0.4, 0.0, 0.5],
        "root_euler": [0, 80, 0],
        "joints": [
            -20, 0, 0,  # left_hip_pitch, roll, yaw
            10, 0, 0,   # left_knee, ankle_pitch, ankle_roll
            -20, 0, 0,  # right_hip_pitch, roll, yaw
            10, 0, 0,   # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            -110, 0,    # left_shoulder_pitch, roll
            -110, 0     # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "push_up_down",
        "duration": 0.7,
        "root_pos": [0.4, 0.0, 0.35],
        "root_euler": [0, 80, 0],
        "joints": [
            -20, 0, 0,  # left_hip_pitch, roll, yaw
            10, 0, 0,   # left_knee, ankle_pitch, ankle_roll
            -20, 0, 0,  # right_hip_pitch, roll, yaw
            10, 0, 0,   # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            -130, 20,   # left_shoulder_pitch, roll
            -130, -20   # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "push_up_up",
        "duration": 0.7,
        "root_pos": [0.4, 0.0, 0.5],
        "root_euler": [0, 80, 0],
        "joints": [
            -20, 0, 0,  # left_hip_pitch, roll, yaw
            10, 0, 0,   # left_knee, ankle_pitch, ankle_roll
            -20, 0, 0,  # right_hip_pitch, roll, yaw
            10, 0, 0,   # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            -110, 0,    # left_shoulder_pitch, roll
            -110, 0     # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "feet_forward",
        "duration": 0.9,
        "root_pos": [0.15, 0.0, 0.7],
        "root_euler": [0, 50, 0],
        "joints": [
            -60, 0, 0,  # left_hip_pitch, roll, yaw
            90, -30, 0, # left_knee, ankle_pitch, ankle_roll
            -60, 0, 0,  # right_hip_pitch, roll, yaw
            90, -30, 0, # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            -90, 20,    # left_shoulder_pitch, roll
            -90, -20    # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "jump",
        "duration": 0.8,
        "root_pos": [0.1, 0.0, 1.6],
        "root_euler": [0, -5, 0],
        "joints": [
            -10, 0, 0,  # left_hip_pitch, roll, yaw
            5, 10, 0,   # left_knee, ankle_pitch, ankle_roll
            -10, 0, 0,  # right_hip_pitch, roll, yaw
            5, 10, 0,   # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            -90, 0,     # left_shoulder_pitch, roll
            -90, 0      # right_shoulder_pitch, roll
        ]
    },
    {
        "name": "land",
        "duration": 0.8,
        "root_pos": [0.0, 0.0, 1.3],
        "root_euler": [0, 0, 0],
        "joints": [
            -15, 0, 0,  # left_hip_pitch, roll, yaw
            25, -10, 0, # left_knee, ankle_pitch, ankle_roll
            -15, 0, 0,  # right_hip_pitch, roll, yaw
            25, -10, 0, # right_knee, ankle_pitch, ankle_roll
            0,          # torso_joint
            0, 0,       # left_shoulder_pitch, roll
            0, 0        # right_shoulder_pitch, roll
        ]
    },
]

TOTAL_DURATION_V5 = sum(k["duration"] for k in BURPEE_V5)

def deg(values):
    return np.radians(np.array(values, dtype=float))

def quat_from_euler_xyz(euler_deg):
    roll, pitch, yaw = np.radians(euler_deg)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)

def reference_at_v5(t):
    """Gibt die Referenzpose für Humanoid v5 zurück"""
    t = t % TOTAL_DURATION_V5
    
    cursor = 0.0
    for idx, keyframe in enumerate(BURPEE_V5):
        next_keyframe = BURPEE_V5[(idx + 1) % len(BURPEE_V5)]
        segment_duration = keyframe["duration"]
        
        if cursor <= t < cursor + segment_duration:
            alpha = (t - cursor) / segment_duration
            t_smooth = smoothstep(alpha)
            
            # Interpolation
            root_pos = (1 - t_smooth) * np.array(keyframe["root_pos"]) + \
                       t_smooth * np.array(next_keyframe["root_pos"])
            
            euler = (1 - t_smooth) * np.array(keyframe["root_euler"]) + \
                    t_smooth * np.array(next_keyframe["root_euler"])
            root_quat = quat_from_euler_xyz(euler)
            
            joints_start = deg(keyframe["joints"])
            joints_end = deg(next_keyframe["joints"])
            joints = (1 - t_smooth) * joints_start + t_smooth * joints_end
            
            phase = np.array([math.sin(2*math.pi*t/TOTAL_DURATION_V5), 
                             math.cos(2*math.pi*t/TOTAL_DURATION_V5)])
            
            return {
                "name": keyframe["name"],
                "root_pos": root_pos,
                "root_quat": root_quat,
                "joints": joints,
                "phase": phase
            }
        cursor += segment_duration
    
    # Fallback
    last = BURPEE_V5[-1]
    return {
        "name": last["name"],
        "root_pos": np.array(last["root_pos"]),
        "root_quat": quat_from_euler_xyz(last["root_euler"]),
        "joints": deg(last["joints"]),
        "phase": np.array([0.0, 1.0])
    }