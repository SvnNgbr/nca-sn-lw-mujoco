from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from burpee_reference import JOINT_NAMES, TOTAL_DURATION, reference_at


class BurpeeImitationEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path="humanoid_burpee_learning.xml",
        episode_seconds=TOTAL_DURATION,
        frame_skip=8,
        random_start=False,
        root_assist=0.9,
        action_scale=0.45,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.episode_seconds = float(episode_seconds)
        self.random_start = bool(random_start)
        self.root_assist = float(root_assist)
        self.action_scale = float(action_scale)
        self.elapsed = 0.0
        self.previous_action = np.zeros(self.model.nu, dtype=np.float32)

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

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        obs_size = (
            self.model.nq
            + self.model.nv
            + len(JOINT_NAMES)
            + 3
            + 4
            + len(JOINT_NAMES)
            + 2
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _reference_control(self):
        return reference_at(self.elapsed)["joints"].astype(np.float32)

    def _policy_action_to_control(self, action):
        ref_ctrl = self._reference_control()
        ctrl = ref_ctrl + np.asarray(action, dtype=np.float32) * self.action_scale
        return np.clip(ctrl, self.ctrl_low, self.ctrl_high)

    def _set_reference_pose(self, ref):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.root_qpos : self.root_qpos + 3] = ref["root_pos"]
        self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] = ref["root_quat"]
        self.data.qpos[self.joint_qpos] = ref["joints"]
        self.data.ctrl[:] = ref["joints"]
        mujoco.mj_forward(self.model, self.data)

    def _apply_root_assist(self):
        if self.root_assist <= 0.0:
            return
        ref = reference_at(self.elapsed)
        amount = np.clip(self.root_assist, 0.0, 1.0)
        root_slice = slice(self.root_qpos, self.root_qpos + 7)
        target_root = np.concatenate([ref["root_pos"], ref["root_quat"]])
        self.data.qpos[root_slice] = (
            (1.0 - amount) * self.data.qpos[root_slice] + amount * target_root
        )
        self.data.qvel[self.root_qvel : self.root_qvel + 6] *= 1.0 - amount
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
                ref["joints"],
                ref["phase"],
            ]
        )
        return obs.astype(np.float32)

    def _reward(self, action, ctrl):
        ref = reference_at(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_pos_error = self.data.qpos[self.root_qpos : self.root_qpos + 3] - ref[
            "root_pos"
        ]
        root_quat_error = self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] - ref[
            "root_quat"
        ]
        ctrl_error = ctrl - ref["joints"]
        action_delta = action - self.previous_action

        pose_reward = np.exp(-8.0 * np.mean(joint_error * joint_error))
        root_reward = np.exp(-10.0 * np.mean(root_pos_error * root_pos_error))
        orient_reward = np.exp(-4.0 * np.mean(root_quat_error * root_quat_error))
        control_reward = np.exp(-0.5 * np.mean(ctrl_error * ctrl_error))
        smooth_reward = np.exp(-0.1 * np.mean(action_delta * action_delta))
        alive_bonus = 0.2 if self.data.qpos[self.root_qpos + 2] > 0.12 else -1.0

        return (
            2.5 * pose_reward
            + 1.5 * root_reward
            + 0.8 * orient_reward
            + 0.6 * control_reward
            + 0.3 * smooth_reward
            + alive_bonus
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed = (
            float(self.np_random.uniform(0.0, TOTAL_DURATION))
            if self.random_start
            else 0.0
        )
        self.previous_action[:] = 0.0
        ref = reference_at(self.elapsed)
        self._set_reference_pose(ref)

        noise = self.np_random.normal(0.0, 0.01, size=len(JOINT_NAMES))
        self.data.qpos[self.joint_qpos] += noise
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"phase_name": ref["name"]}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = self._policy_action_to_control(action)
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.elapsed += self.dt
        self._apply_root_assist()

        reward = self._reward(action, ctrl)
        self.previous_action = action.copy()
        root_height = self.data.qpos[self.root_qpos + 2]
        terminated = bool(root_height < 0.08 or root_height > 1.9)
        truncated = bool(self.elapsed >= self.episode_seconds)
        return self._get_obs(), float(reward), terminated, truncated, {
            "phase_name": reference_at(self.elapsed)["name"],
            "root_height": float(root_height),
            "root_assist": self.root_assist,
        }
