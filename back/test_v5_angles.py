import mujoco
import mujoco.viewer
import numpy as np
import time

# Lade das Modell
model = mujoco.MjModel.from_xml_path("humanoid_v5.xml")
data = mujoco.MjData(model)

# Definiere die Burpee-Phasen (angepasst für nq=24)
burpee_phases = {
    "stand": np.array([
        0, 0, 1.2,       # root_pos (x, y, z)
        1, 0, 0, 0,      # root_quat (keine Rotation)
        0, 0, 0,         # right_hip_x, right_hip_y, right_hip_z
        0,               # right_knee
        0, 0,            # right_ankle_x, right_ankle_y
        0, 0, 0,         # left_hip_x, left_hip_y, left_hip_z
        0,               # left_knee
        0, 0,            # left_ankle_x, left_ankle_y
        0, 0,            # right_shoulder_x, right_shoulder_y
        0,               # right_elbow
        0, 0             # left_shoulder_x, left_shoulder_y
    ]),
    "squat": np.array([
        0, 0, 0.8,       # root_pos (tiefer)
        0.9, 0, 0.4, 0,  # root_quat (leicht nach vorne geneigt)
        -0.5, 0, 0,      # right_hip_x, right_hip_y, right_hip_z
        1.5,             # right_knee (stark gebeugt)
        -0.3, 0,         # right_ankle_x, right_ankle_y
        -0.5, 0, 0,      # left_hip_x, left_hip_y, left_hip_z
        1.5,             # left_knee
        -0.3, 0,         # left_ankle_x, left_ankle_y
        0, 0,            # right_shoulder_x, right_shoulder_y
        0.5,             # right_elbow
        0, 0             # left_shoulder_x, left_shoulder_y
    ]),
    "plank": np.array([
        0, 0, 0.3,       # root_pos (tief)
        0.7, 0, 0.7, 0,  # root_quat (parallel zum Boden)
        0, 0, 0,         # right_hip_x, right_hip_y, right_hip_z
        0,               # right_knee (gestreckt)
        0, 0,            # right_ankle_x, right_ankle_y
        0, 0, 0,         # left_hip_x, left_hip_y, left_hip_z
        0,               # left_knee
        0, 0,            # left_ankle_x, left_ankle_y
        1.0, 0,          # right_shoulder_x, right_shoulder_y (nach vorne)
        0,               # right_elbow
        -1.0, 0          # left_shoulder_x, left_shoulder_y
    ]),
    "pushup": np.array([
        0, 0, 0.2,       # root_pos (noch tiefer)
        0.6, 0, 0.8, 0,  # root_quat
        0, 0, 0,         # right_hip_x, right_hip_y, right_hip_z
        0,               # right_knee
        0, 0,            # right_ankle_x, right_ankle_y
        0, 0, 0,         # left_hip_x, left_hip_y, left_hip_z
        0,               # left_knee
        0, 0,            # left_ankle_x, left_ankle_y
        1.2, 0,          # right_shoulder_x, right_shoulder_y (weiter nach vorne)
        1.5,             # right_elbow (stark gebeugt)
        1.2, 0           # left_shoulder_x, left_shoulder_y
    ])
}

# Phase order
phase_order = ["stand", "squat", "plank", "pushup", "plank", "squat", "stand"]

# PD-Controller Parameter
kp = 10.0  # Proportionaler Gain
kd = 1.0   # Derivativer Gain

# Starte den Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    current_phase_idx = 0
    current_target = burpee_phases[phase_order[current_phase_idx]]
    steps_per_phase = 1000  # Anzahl der Schritte pro Phase
    step_counter = 0

    while viewer.is_running():
        # PD-Controller anwenden (nur für Aktuatoren)
        error = current_target - data.qpos
        d_error = -data.qvel

        # Nur die ersten model.nu Werte von error und d_error verwenden
        data.ctrl = kp * error[:model.nu] + kd * d_error[:model.nu]

        # Simulation aktualisieren
        mujoco.mj_step(model, data)
        viewer.sync()

        # Phase wechseln
        step_counter += 1
        if step_counter >= steps_per_phase:
            step_counter = 0
            current_phase_idx = (current_phase_idx + 1) % len(phase_order)
            current_target = burpee_phases[phase_order[current_phase_idx]]
            print(f"Wechsel zu Phase: {phase_order[current_phase_idx]}")

        time.sleep(0.01)