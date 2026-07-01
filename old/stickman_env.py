import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class StickmanEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Lade das Modell
        model_path = "guntherthestickman.xml"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XML-Datei nicht gefunden: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Aktionsraum: Normalisierte Aktionen für jeden Aktuator (-1 bis 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )

        # Beobachtungsraum: qpos + qvel
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.nq + self.model.nv,),
            dtype=np.float32
        )

        # Maximale Schritte pro Episode
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Setze die Simulation zurück
        mujoco.mj_resetData(self.model, self.data)

        # Setze den Stickman in eine liegende Position (fürs Aufstehen)
        self.data.qpos[2] = 0.5  # Torso auf dem Boden (z=0.5)
        self.data.qpos[3:7] = [0.5, 0.0, 0.0, 0.0]  # Hüften und Knie anpassen
        self.data.qpos[7:11] = [0.5, 0.0, 0.0, 0.0]  # Schultern und Ellbogen

        # Aktualisiere die Vorwärtskinematik
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # Skaliere die Aktion auf die Aktuator-Kräfte
        self.data.ctrl = action * 50  # Skalierungsfaktor (50 ist ein guter Startwert)

        # Simuliere einen Schritt
        mujoco.mj_step(self.model, self.data)

        # Berechne Belohnung
        torso_height = self.data.qpos[2]  # z-Position des Torsos
        reward = torso_height - 1.0  # Belohnung für Höhe > 1.0

        # Strafe für zu starke Bewegungen (vermindert Schwingungen)
        velocity_penalty = -0.01 * np.sum(np.abs(self.data.qvel))

        # Strafe für zu niedrige Höhe (Sturz)
        height_penalty = -10.0 if torso_height < 0.5 else 0.0

        # Gesamtbelohnung
        reward += velocity_penalty + height_penalty

        # Beende, wenn der Stickman zu tief fällt oder die maximale Schrittzahl erreicht ist
        terminated = torso_height < 0.3 or self.current_step >= self.max_steps
        truncated = False

        self.current_step += 1
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # Gib die aktuellen Gelenkwinkel und -geschwindigkeiten zurück
        return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Registriere die Umgebung
gym.register(
    id="StickmanStandup-v0",
    entry_point="stickman_env:StickmanEnv",
)