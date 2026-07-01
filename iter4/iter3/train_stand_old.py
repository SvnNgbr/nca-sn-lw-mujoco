import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import numpy as np

# 1. Umgebung registrieren und laden
env = make_vec_env(lambda: gym.make("StickmanStandup-v0"), n_envs=1)

# 2. Callback für TensorBoard (optional, aber nützlich)
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._log_dir = "logs/standup/"
        os.makedirs(self._log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Logge die Belohnung
        for info in self.model.get_env().get_attr("episode_rewards"):
            if info:
                self.logger.record("train/episode_reward", np.mean(info))
        return True

# 3. Modell erstellen
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    clip_range=0.2,
    max_grad_norm=0.5,
    tensorboard_log="logs/standup/"
)

# 4. Modell trainieren
callback = TensorboardCallback()
model.learn(total_timesteps=10_000, callback=callback)  # 500k Schritte (ca. 10-15 Minuten)

# 5. Modell speichern
model.save("stickman_standup_ppo")

# 6. Umgebung schließen
env.close()