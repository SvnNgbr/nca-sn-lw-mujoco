from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from v5ref import reference_at_v5, JOINT_NAMES_V5, TOTAL_DURATION_V5

class BurpeeHumanoidV5Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        render_mode=None,
        episode_seconds=TOTAL_DURATION_V5,
        frame_skip=5,
        random_start=False,
        root_assist=0.9,
        action_scale=0.45,
        hold_seconds=2.0,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        
        self.frame_skip = int(frame_skip)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.episode_seconds = float(episode_seconds)
        self.random_start = bool(random_start)
        self.root_assist = float(root_assist)
        self.action_scale = float(action_scale)
        self.hold_seconds = float(hold_seconds)
        self.elapsed = 0.0
        self.previous_action = np.zeros(self.model.nu, dtype=np.float32)
        
        self._reference_func = reference_at_v5
        self.total_duration = TOTAL_DURATION_V5

        # Für Halte-Bewertung
        self.pose_start_time = 0.0
        self.current_phase = ""
        self.phase_hold_time = 0.0
        self.pose_held = False

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
        
        # ORIGINAL OBSERVATION SIZE - OHNE hold_time!
        obs_size = (
            self.model.nq + 
            self.model.nv + 
            len(self.joint_qpos) + 
            3 + 4 + 
            len(self.joint_qpos) + 
            2  # KEIN +1 mehr!
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )

    def render(self):
        if self.render_mode == "human":
            return self.env.render()
        elif self.render_mode == "rgb_array":
            return self.env.render()
        return None

    def _reference_control(self):
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
        self.data.qvel[self.root_qvel:self.root_qvel+6] *= 1.0 - amount * 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        self.elapsed = 0.0
        self.previous_action[:] = 0.0
        
        # Halte-Timer zurücksetzen
        self.pose_start_time = 0.0
        self.current_phase = ""
        self.phase_hold_time = 0.0
        self.pose_held = False
        
        ref = self._reference_func(0)
        for i, qpos_addr in enumerate(self.joint_qpos):
            if i < len(ref["joints"]):
                self.data.qpos[qpos_addr] = ref["joints"][i]
        
        if "root_pos" in ref:
            self.data.qpos[self.root_qpos:self.root_qpos+3] = ref["root_pos"]
        if "root_quat" in ref:
            self.data.qpos[self.root_qpos+3:self.root_qpos+7] = ref["root_quat"]
        
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info

    def _get_obs(self):
        ref = self._reference_func(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_pos = self.data.qpos[self.root_qpos:self.root_qpos+3]
        root_quat = self.data.qpos[self.root_qpos+3:self.root_qpos+7]
        
        # OHNE hold_time in der Observation!
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel,
            joint_error,
            root_pos - ref["root_pos"],
            root_quat - ref["root_quat"],
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

        ref = self._reference_func(self.elapsed)
        current_phase = ref["name"]
        
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        pose_error = np.mean(joint_error * joint_error)
        is_pose_good = pose_error < 0.5
        
        if current_phase != self.current_phase:
            self.current_phase = current_phase
            self.phase_hold_time = 0.0
            self.pose_held = False
        elif is_pose_good:
            self.phase_hold_time += self.dt
            if self.phase_hold_time >= self.hold_seconds:
                self.pose_held = True

        reward = self._compute_reward(action, ctrl)
        self.previous_action = action.copy()
        
        root_height = self.data.qpos[self.root_qpos+2]
        
        if root_height < 0.12:
            self.fallen = True
            if root_height < 0.05:
                terminated = True
            else:
                terminated = False
        else:
            self.fallen = False
            terminated = False
        
        truncated = self.elapsed >= self.episode_seconds
        
        # Erweiterter info-Dict
        return self._get_obs(), float(reward), terminated, truncated, {
            "root_height": float(root_height),
            "phase": current_phase,
            "hold_time": self.phase_hold_time,
            "pose_held": self.pose_held,
            "pose_error": float(pose_error),                      # NEU
            "joint_error_mean": float(np.mean(joint_error * joint_error)),  # NEU
            "control_cost": float(np.mean(ctrl * ctrl)),          # NEU
            "fallen": self.fallen
        }
    def _compute_reward(self, action, ctrl):
        ref = self._reference_func(self.elapsed)
        joint_error = self.data.qpos[self.joint_qpos] - ref["joints"]
        root_height = self.data.qpos[self.root_qpos+2]
        root_pos_error = self.data.qpos[self.root_qpos:self.root_qpos+3] - ref["root_pos"]
        
        pose_reward = np.exp(-5.0 * np.mean(joint_error * joint_error))
        
        target_height = ref["root_pos"][2]
        height_reward = np.exp(-3.0 * (root_height - target_height)**2)
        root_reward = np.exp(-2.0 * np.mean(root_pos_error[:2]**2))
        
        # Bonus für Halten der Pose (NUR im Reward, nicht in Observation)
        hold_bonus = 0.0
        if self.pose_held:
            hold_bonus = 2.0
        elif self.phase_hold_time > 0:
            hold_bonus = 0.5 * min(self.phase_hold_time / self.hold_seconds, 1.0)
        
        alive_bonus = 0.5 if root_height > 0.15 else -1.0
        control_cost = -0.01 * np.mean(ctrl * ctrl)
        
        return (2.0 * pose_reward + 
                1.0 * height_reward + 
                0.5 * root_reward + 
                hold_bonus +
                alive_bonus + 
                control_cost)

    def close(self):
        self.env.close()