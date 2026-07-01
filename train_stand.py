import gymnasium as gym
from stable_baselines3 import PPO #quasi pytorch im hintergrund
from stable_baselines3.common.callbacks import BaseCallback
import os

# 1. Umgebung laden
env = gym.make("HumanoidStandup-v5")

# 2. Callback für TensorBoard
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._log_dir = "logs/standup/"
        os.makedirs(self._log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

# 3. Modell erstellen
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    tensorboard_log="logs/standup/"
)

# 4. Modell trainieren
model.learn(total_timesteps=50_000, callback=TensorboardCallback()) #50k weil zeit und so
model.save("humanoid_standup_ppo")

# 5. Umgebung schließen
env.close()