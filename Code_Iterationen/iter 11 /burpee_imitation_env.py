from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from burpee_reference import JOINT_NAMES, TOTAL_DURATION, reference_at


class BurpeeImitationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path="humanoid_burpee_learning.xml",
        episode_seconds=TOTAL_DURATION,
        frame_skip=8,
        random_start=True,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.episode_seconds = float(episode_seconds)
        self.random_start = bool(random_start)
        self.elapsed = 0.0
        self.steps = 0

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.joint_qpos = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids]
        )
        self.joint_qvel = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids]
        )
        self.root_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        self.root_qpos = self.model.jnt_qposadr[self.root_joint_id]
        self.root_qvel = self.model.jnt_dofadr[self.root_joint_id]

        act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        obs_size = (
            self.model.nq
            + self.model.nv
            + len(JOINT_NAMES)
            + 3
            + 4
            + 2
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _set_reference_pose(self, ref):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.root_qpos : self.root_qpos + 3] = ref["root_pos"]
        self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] = ref["root_quat"]
        self.data.qpos[self.joint_qpos] = ref["joints"]
        self.data.ctrl[:] = ref["joints"]
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        ref = reference_at(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_pos_error = self.data.qpos[self.root_qpos : self.root_qpos + 3] - ref[
            "root_pos"
        ]
        root_quat_error = self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] - ref[
            "root_quat"
        ]
        obs = np.concatenate(
            [
                self.data.qpos,
                self.data.qvel,
                joint_error,
                root_pos_error,
                root_quat_error,
                ref["phase"],
            ]
        )
        return obs.astype(np.float32)

    def _reward(self, action):
        ref = reference_at(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_pos_error = self.data.qpos[self.root_qpos : self.root_qpos + 3] - ref[
            "root_pos"
        ]
        root_quat_error = self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] - ref[
            "root_quat"
        ]
        ctrl_error = action - ref["joints"]

        pose_reward = np.exp(-3.0 * np.mean(joint_error * joint_error))
        root_reward = np.exp(-6.0 * np.mean(root_pos_error * root_pos_error))
        orient_reward = np.exp(-2.0 * np.mean(root_quat_error * root_quat_error))
        control_reward = np.exp(-0.05 * np.mean(ctrl_error * ctrl_error))
        alive_bonus = 0.1 if self.data.qpos[self.root_qpos + 2] > 0.18 else -1.0

        return (
            1.8 * pose_reward
            + 1.0 * root_reward
            + 0.6 * orient_reward
            + 0.2 * control_reward
            + alive_bonus
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed = (
            float(self.np_random.uniform(0.0, TOTAL_DURATION))
            if self.random_start
            else 0.0
        )
        self.steps = 0
        ref = reference_at(self.elapsed)
        self._set_reference_pose(ref)

        noise = self.np_random.normal(0.0, 0.025, size=len(JOINT_NAMES))
        self.data.qpos[self.joint_qpos] += noise
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"phase_name": ref["name"]}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.data.ctrl[:] = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.elapsed += self.dt
        self.steps += 1
        reward = self._reward(action)
        root_height = self.data.qpos[self.root_qpos + 2]
        terminated = bool(root_height < 0.12 or root_height > 1.8)
        truncated = bool(self.elapsed >= self.episode_seconds)
        return self._get_obs(), float(reward), terminated, truncated, {
            "phase_name": reference_at(self.elapsed)["name"],
            "root_height": float(root_height),
        }
