from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from v5ref import reference_at_v5, JOINT_NAMES_V5, TOTAL_DURATION_V5


class BurpeeHumanoidV5Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode=None,
        episode_seconds=TOTAL_DURATION_V5,
        frame_skip=5,
        random_start=False,
        root_assist=0.9,
        action_scale=0.45,
    ):
        super().__init__()
        
        # Humanoid v5 Umgebung laden
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        
        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.episode_seconds = float(episode_seconds)
        self.random_start = bool(random_start)
        self.root_assist = float(root_assist)
        self.action_scale = float(action_scale)
        self.elapsed = 0.0
        self.previous_action = np.zeros(self.model.nu, dtype=np.float32)
        
        # Referenz-Funktion speichern
        self._reference_func = reference_at_v5
        self.total_duration = TOTAL_DURATION_V5

        # Joint-IDs für spezifische Gelenke
        self.joint_ids = []
        self.joint_qpos = []
        self.joint_qvel = []
        
        for name in JOINT_NAMES_V5:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_ids.append(joint_id)
                self.joint_qpos.append(self.model.jnt_qposadr[joint_id])
                self.joint_qvel.append(self.model.jnt_dofadr[joint_id])
            except:
                print(f"Warning: Joint {name} not found")
                self.joint_ids.append(-1)
                self.joint_qpos.append(-1)
                self.joint_qvel.append(-1)
        
        self.joint_qpos = np.array([x for x in self.joint_qpos if x >= 0])
        self.joint_qvel = np.array([x for x in self.joint_qvel if x >= 0])
        
        # Root-Joint finden
        self.root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qpos = self.model.jnt_qposadr[self.root_joint_id]
        self.root_qvel = self.model.jnt_dofadr[self.root_joint_id]

        # Aktionsraum
        self.action_space = spaces.Box(
            -1.0, 1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
        # Beobachtungsraum
        obs_size = (
            self.model.nq + 
            self.model.nv + 
            len(self.joint_qpos) + 
            3 + 4 + 
            len(self.joint_qpos) + 
            2
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )

    def _reference_control(self):
        """Holt die Referenzpose zum aktuellen Zeitpunkt"""
        ref = self._reference_func(self.elapsed)
        return ref["joints"].astype(np.float32)

    def _policy_action_to_control(self, action):
        ref_ctrl = self._reference_control()
        ctrl = ref_ctrl + np.asarray(action, dtype=np.float32) * self.action_scale
        return np.clip(ctrl, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

    def _apply_root_assist(self):
        if self.root_assist <= 0.0:
            return
        ref = self._reference_func(self.elapsed)
        amount = np.clip(self.root_assist, 0.0, 1.0)
        root_slice = slice(self.root_qpos, self.root_qpos + 7)
        target_root = np.concatenate([ref["root_pos"], ref["root_quat"]])
        self.data.qpos[root_slice] = (
            (1.0 - amount) * self.data.qpos[root_slice] + amount * target_root
        )
        self.data.qvel[self.root_qvel:self.root_qvel+6] *= 1.0 - amount
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        self.elapsed = 0.0
        self.previous_action[:] = 0.0
        
        # Initiale Pose setzen
        self._apply_root_assist()
        return self._get_obs(), info

    def _get_obs(self):
        """Erstellt die Beobachtung mit Referenz-Daten"""
        ref = self._reference_func(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_pos_error = self.data.qpos[self.root_qpos:self.root_qpos+3] - ref["root_pos"]
        root_quat_error = self.data.qpos[self.root_qpos+3:self.root_qpos+7] - ref["root_quat"]
        
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel,
            joint_error,
            root_pos_error,
            root_quat_error,
            ref["joints"],
            ref["phase"],
        ])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = self._policy_action_to_control(action)
        self.data.ctrl[:] = ctrl
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.elapsed += self.dt
        self._apply_root_assist()

        reward = self._compute_reward(action, ctrl)
        self.previous_action = action.copy()
        
        terminated = self.data.qpos[self.root_qpos+2] < 0.08
        truncated = self.elapsed >= self.episode_seconds
        
        return self._get_obs(), float(reward), terminated, truncated, {
            "root_height": float(self.data.qpos[self.root_qpos+2]),
            "root_assist": self.root_assist
        }

    def _compute_reward(self, action, ctrl):
        ref = self._reference_func(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_pos_error = self.data.qpos[self.root_qpos:self.root_qpos+3] - ref["root_pos"]
        root_quat_error = self.data.qpos[self.root_qpos+3:self.root_qpos+7] - ref["root_quat"]
        ctrl_error = ctrl - ref["joints"]
        action_delta = action - self.previous_action

        pose_reward = np.exp(-8.0 * np.mean(joint_error * joint_error))
        root_reward = np.exp(-10.0 * np.mean(root_pos_error * root_pos_error))
        orient_reward = np.exp(-4.0 * np.mean(root_quat_error * root_quat_error))
        control_reward = np.exp(-0.5 * np.mean(ctrl_error * ctrl_error))
        smooth_reward = np.exp(-0.1 * np.mean(action_delta * action_delta))
        alive_bonus = 0.2 if self.data.qpos[self.root_qpos+2] > 0.12 else -1.0

        return (
            2.5 * pose_reward +
            1.5 * root_reward +
            0.8 * orient_reward +
            0.6 * control_reward +
            0.3 * smooth_reward +
            alive_bonus
        )