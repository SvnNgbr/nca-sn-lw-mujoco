import os

from stable_baselines3 import PPO
from Iterationen.iter5.standing_env import StandingEnv

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = StandingEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    tensorboard_log="logs/"
)

model.learn(total_timesteps=50_000)

model.save("models/humanoid_stand")
env.close()