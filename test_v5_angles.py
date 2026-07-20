import mujoco
import mujoco.viewer
import numpy as np
import time

# Lade das Modell
model = mujoco.MjModel.from_xml_path("humanoid_v5.xml")
data = mujoco.MjData(model)

# Definiere die Burpee-Phasen
burpee_phases = {
    "stand": np.array([
        0, 0, 1.2, 1, 0, 0, 0,  # Torso
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Beine (rechts + links)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Arme (rechts + links)
        0, 0, 0  # Hals
    ]),
    "squat": np.array([
        0, 0, 0.8, 0.9, 0, 0.4, 0,  # Torso (leicht nach vorne)
        -0.5, 0, 0, 1.5, -0.3, 0, 0,  # Rechtes Bein
        -0.5, 0, 0, 1.5, -0.3, 0, 0,  # Linkes Bein
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Arme
        0, 0, 0  # Hals
    ]),
    "plank": np.array([
        0, 0, 0.3, 0.7, 0, 0.7, 0,  # Torso (parallel zum Boden)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Beine (gestreckt)
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0,  # Arme (nach vorne)
        0, 0, 0  # Hals
    ]),
    "pushup": np.array([
        0, 0, 0.2, 0.6, 0, 0.8, 0,  # Torso (tief)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Beine
        1.2, 0, 0, 1.5, 1.2, 0, 0, 1.5, 0, 0,  # Arme (Ellenbogen gebeugt)
        0, 0, 0  # Hals
    ])
}

# Starte den Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Durchlaufe die Burpee-Phasen
    for phase_name, qpos in burpee_phases.items():
        print(f"Phase: {phase_name}")
        data.qpos = qpos
        mujoco.mj_forward(model, data)

        # Halte die Pose für 3 Sekunden
        for _ in range(300):  # 300 Schritte ≈ 3 Sekunden (bei 100 Hz)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)  # Verlangsamt die Simulation