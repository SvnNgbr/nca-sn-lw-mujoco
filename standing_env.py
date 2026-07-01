import numpy as np
import gymnasium as gym


class StandingEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make(
            "Humanoid-v5",
            render_mode=render_mode
        )
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Höhe des Körpers
        torso_height = self.unwrapped.data.qpos[2]

        # Geschwindigkeit
        velocity = np.linalg.norm(self.unwrapped.data.qvel)

        # Energieverbrauch
        energy = np.sum(np.square(action))

        #punish bewgung weil starten stehend
        reward = (
            10.0 * torso_height
            - 0.05 * velocity
            - 0.001 * energy
        )

        # Umgefallen
        if torso_height < 0.9:
            reward -= 100
            terminated = True

        return obs, reward, terminated, truncated, info