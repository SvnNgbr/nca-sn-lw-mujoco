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
        
        self._reference_func = reference_at_v5
        self.total_duration = TOTAL_DURATION_V5

        # Joint-IDs
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

        self.action_space = spaces.Box(
            -1.0, 1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
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
        ref = self._reference_func(self.elapsed)
        return ref["joints"].astype(np.float32)

    def _policy_action_to_control(self, action):
        ref_ctrl = self._reference_control()
        ctrl = ref_ctrl + np.asarray(action, dtype=np.float32) * self.action_scale
        return np.clip(ctrl, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

    def _apply_root_assist(self):
        # Root-Assist deaktiviert - der Roboter muss selbst die Balance halten!
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        self.elapsed = 0.0
        self.previous_action[:] = 0.0
        
        # Initiale Pose setzen (nur Gelenke)
        ref = self._reference_func(0)
        for i, qpos_addr in enumerate(self.joint_qpos):
            if i < len(ref["joints"]):
                self.data.qpos[qpos_addr] = ref["joints"][i]
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info

    def _get_obs(self):
        ref = self._reference_func(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_pos = self.data.qpos[self.root_qpos:self.root_qpos+3]
        root_quat = self.data.qpos[self.root_qpos+3:self.root_qpos+7]
        
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel,
            joint_error,
            root_pos - np.array([0, 0, 1.4]),  # Referenzhöhe
            root_quat - np.array([1, 0, 0, 0]),
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

        reward = self._compute_reward(action, ctrl)
        self.previous_action = action.copy()
        
        root_height = self.data.qpos[self.root_qpos+2]
        terminated = root_height < 0.3  # Gefallen
        truncated = self.elapsed >= self.episode_seconds
        
        return self._get_obs(), float(reward), terminated, truncated, {
            "root_height": float(root_height)
        }

    def _compute_reward(self, action, ctrl):
        ref = self._reference_func(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_height = self.data.qpos[self.root_qpos+2]
        
        pose_reward = np.exp(-5.0 * np.mean(joint_error * joint_error))
        height_reward = np.exp(-2.0 * (root_height - 1.3)**2)  # Stehen belohnen
        alive_bonus = 1.0 if root_height > 0.5 else -2.0
        control_cost = -0.01 * np.mean(ctrl * ctrl)
        
        return 2.0 * pose_reward + 1.0 * height_reward + alive_bonus + control_cost