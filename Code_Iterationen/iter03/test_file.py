import gymnasium as gym
print([env.id for env in gym.envs.registry.env_specs.values() if "Stickman" in env.id])