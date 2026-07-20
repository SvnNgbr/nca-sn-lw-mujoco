import math
import numpy as np

# Humanoid v5 Joints (in der richtigen Reihenfolge)
JOINT_NAMES_V5 = [
    "abdomen_z",      # 0: Oberkörper drehen
    "abdomen_y",      # 1: Oberkörper seitlich
    "abdomen_x",      # 2: Oberkörper vor/zurück
    "right_hip_x",    # 3: Rechte Hüfte (positiv = innen, negativ = aussen)
    "right_hip_z",    # 4: Rechte Hüfte drehen
    "right_hip_y",    # 5: Rechte Hüfte seitlich
    "right_knee",     # 6: Rechtes Knie (positiv = strecken, negativ = beugen)
    "left_hip_x",     # 7: Linke Hüfte (positiv = innen, negativ = aussen)
    "left_hip_z",     # 8: Linke Hüfte drehen
    "left_hip_y",     # 9: Linke Hüfte seitlich
    "left_knee",      # 10: Linkes Knie (positiv = strecken, negativ = beugen)
    "right_shoulder1", # 11: Rechte Schulter vor/zurück
    "right_shoulder2", # 12: Rechte Schulter seitlich
    "right_elbow",    # 13: Rechter Ellbogen
    "left_shoulder1",  # 14: Linke Schulter vor/zurück
    "left_shoulder2",  # 15: Linke Schulter seitlich
    "left_elbow"      # 16: Linker Ellbogen
]

# Korrigierte Burpee-Posen basierend auf Ihren Tests
BURPEE_V5 = [
    {
        "name": "stand",
        "duration": 2.0,
        "root_pos": [0.0, 0.0, 1.4],
        "root_euler": [0, 0, 0],
        "joints": [
            0,  # abdomen_z
            0,  # abdomen_y
            0,  # abdomen_x
            0,  # right_hip_x
            0,  # right_hip_z
            0,  # right_hip_y
            0,  # right_knee
            0,  # left_hip_x
            0,  # left_hip_z
            0,  # left_hip_y
            0,  # left_knee
            0,  # right_shoulder1
            0,  # right_shoulder2
            0,  # right_elbow
            0,  # left_shoulder1
            0,  # left_shoulder2
            0   # left_elbow
        ]
    },
    {
        "name": "squat",
        "duration": 2.0,
        "root_pos": [0.0, 0.0, 0.85],
        "root_euler": [0, 5, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            10,   # abdomen_x: Oberkörper leicht nach vorne
            0,    # right_hip_x: neutral
            0,    # right_hip_z
            0,    # right_hip_y
            -60,  # right_knee: NEGATIV = Knie beugen
            0,    # left_hip_x: neutral
            0,    # left_hip_z
            0,    # left_hip_y
            -60,  # left_knee: NEGATIV = Knie beugen
            10,   # right_shoulder1
            0,    # right_shoulder2
            -20,  # right_elbow
            10,   # left_shoulder1
            0,    # left_shoulder2
            -20   # left_elbow
        ]
    },
    {
        "name": "bend_forward",
        "duration": 1.5,
        "root_pos": [0.15, 0.0, 0.65],
        "root_euler": [0, 45, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            30,   # abdomen_x: Oberkörper nach vorne
            0,    # right_hip_x
            0,    # right_hip_z
            0,    # right_hip_y
            -90,  # right_knee: stark beugen
            0,    # left_hip_x
            0,    # left_hip_z
            0,    # left_hip_y
            -90,  # left_knee: stark beugen
            -90,  # right_shoulder1
            20,   # right_shoulder2
            -90,  # right_elbow
            -90,  # left_shoulder1
            -20,  # left_shoulder2
            -90   # left_elbow
        ]
    },
    {
        "name": "plank",
        "duration": 2.0,
        "root_pos": [0.4, 0.0, 0.45],
        "root_euler": [0, 85, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            0,    # abdomen_x
            0,    # right_hip_x
            0,    # right_hip_z
            0,    # right_hip_y
            -5,   # right_knee: fast gestreckt
            0,    # left_hip_x
            0,    # left_hip_z
            0,    # left_hip_y
            -5,   # left_knee: fast gestreckt
            -110, # right_shoulder1
            0,    # right_shoulder2
            -5,   # right_elbow: fast gestreckt
            -110, # left_shoulder1
            0,    # left_shoulder2
            -5    # left_elbow: fast gestreckt
        ]
    },
    {
        "name": "push_up_down",
        "duration": 1.5,
        "root_pos": [0.4, 0.0, 0.35],
        "root_euler": [0, 85, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            5,    # abdomen_x
            0,    # right_hip_x
            0,    # right_hip_z
            0,    # right_hip_y
            -5,   # right_knee
            0,    # left_hip_x
            0,    # left_hip_z
            0,    # left_hip_y
            -5,   # left_knee
            -130, # right_shoulder1
            20,   # right_shoulder2
            -100, # right_elbow: stark gebeugt
            -130, # left_shoulder1
            -20,  # left_shoulder2
            -100  # left_elbow: stark gebeugt
        ]
    },
    {
        "name": "push_up_up",
        "duration": 1.5,
        "root_pos": [0.4, 0.0, 0.45],
        "root_euler": [0, 85, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            0,    # abdomen_x
            0,    # right_hip_x
            0,    # right_hip_z
            0,    # right_hip_y
            -5,   # right_knee
            0,    # left_hip_x
            0,    # left_hip_z
            0,    # left_hip_y
            -5,   # left_knee
            -110, # right_shoulder1
            0,    # right_shoulder2
            -5,   # right_elbow
            -110, # left_shoulder1
            0,    # left_shoulder2
            -5    # left_elbow
        ]
    },
    {
        "name": "feet_forward",
        "duration": 1.5,
        "root_pos": [0.15, 0.0, 0.65],
        "root_euler": [0, 45, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            30,   # abdomen_x
            0,    # right_hip_x
            0,    # right_hip_z
            0,    # right_hip_y
            -90,  # right_knee
            0,    # left_hip_x
            0,    # left_hip_z
            0,    # left_hip_y
            -90,  # left_knee
            -90,  # right_shoulder1
            20,   # right_shoulder2
            -90,  # right_elbow
            -90,  # left_shoulder1
            -20,  # left_shoulder2
            -90   # left_elbow
        ]
    },
    {
        "name": "jump",
        "duration": 1.5,
        "root_pos": [0.1, 0.0, 1.6],
        "root_euler": [0, -5, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            -10,  # abdomen_x: leicht zurück
            0,    # right_hip_x
            0,    # right_hip_z
            0,    # right_hip_y
            -10,  # right_knee: leicht gebeugt
            0,    # left_hip_x
            0,    # left_hip_z
            0,    # left_hip_y
            -10,  # left_knee: leicht gebeugt
            -90,  # right_shoulder1
            0,    # right_shoulder2
            -5,   # right_elbow
            -90,  # left_shoulder1
            0,    # left_shoulder2
            -5    # left_elbow
        ]
    },
    {
        "name": "land",
        "duration": 1.5,
        "root_pos": [0.0, 0.0, 1.3],
        "root_euler": [0, 0, 0],
        "joints": [
            0,    # abdomen_z
            0,    # abdomen_y
            0,    # abdomen_x
            0,    # right_hip_x
            0,    # right_hip_z
            0,    # right_hip_y
            -25,  # right_knee: leicht gebeugt zum Landen
            0,    # left_hip_x
            0,    # left_hip_z
            0,    # left_hip_y
            -25,  # left_knee: leicht gebeugt zum Landen
            0,    # right_shoulder1
            0,    # right_shoulder2
            0,    # right_elbow
            0,    # left_shoulder1
            0,    # left_shoulder2
            0     # left_elbow
        ]
    },
]

TOTAL_DURATION_V5 = sum(k["duration"] for k in BURPEE_V5)

def deg(values):
    return np.radians(np.array(values, dtype=float))

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)

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

def reference_at_v5(t):
    t = t % TOTAL_DURATION_V5
    
    cursor = 0.0
    for idx, keyframe in enumerate(BURPEE_V5):
        next_keyframe = BURPEE_V5[(idx + 1) % len(BURPEE_V5)]
        segment_duration = keyframe["duration"]
        
        if cursor <= t < cursor + segment_duration:
            alpha = (t - cursor) / segment_duration
            t_smooth = smoothstep(alpha)
            
            root_pos = (1 - t_smooth) * np.array(keyframe["root_pos"]) + \
                       t_smooth * np.array(next_keyframe["root_pos"])
            
            root_euler = (1 - t_smooth) * np.array(keyframe["root_euler"]) + \
                         t_smooth * np.array(next_keyframe["root_euler"])
            root_quat = quat_from_euler_xyz(root_euler)
            
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
    
    last = BURPEE_V5[-1]
    return {
        "name": last["name"],
        "root_pos": np.array(last["root_pos"]),
        "root_quat": quat_from_euler_xyz(last["root_euler"]),
        "joints": deg(last["joints"]),
        "phase": np.array([0.0, 1.0])
    }