from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from v5ref import BURPEE_V5, JOINT_NAMES_V5, deg

class BurpeeHumanoidV5Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        render_mode=None,
        random_start=False,
        root_assist=0.9,
        action_scale=0.45,
        hold_seconds=2.0,
        pose_threshold=0.3,
        max_steps_per_phase=500,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        
        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip
        self.random_start = bool(random_start)
        self.root_assist = float(root_assist)
        self.action_scale = float(action_scale)
        self.hold_seconds = float(hold_seconds)
        self.pose_threshold = float(pose_threshold)
        self.max_steps_per_phase = int(max_steps_per_phase)
        self.elapsed = 0.0
        self.previous_action = np.zeros(self.model.nu, dtype=np.float32)
        
        # Phasen-basiert: Tracke die aktuelle Phase
        self.phase_idx = 0
        self.phase_start_time = 0.0
        self.phase_hold_time = 0.0
        self.phase_steps = 0
        self.pose_held = False
        self.phase_complete = False

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
                self.joint_ids.append(-1)
                self.joint_qpos.append(-1)
                self.joint_qvel.append(-1)
        
        self.joint_qpos = np.array([x for x in self.joint_qpos if x >= 0])
        self.joint_qvel = np.array([x for x in self.joint_qvel if x >= 0])
        
        self.root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qpos = self.model.jnt_qposadr[self.root_joint_id]
        self.root_qvel = self.model.jnt_dofadr[self.root_joint_id]

        self.action_space = spaces.Box(
            -1.0, 1.0, 
            shape=(self.model.nu,), 
            dtype=np.float32
        )
        
        # Observation: Phase-Index hinzufügen
        obs_size = (
            self.model.nq + 
            self.model.nv + 
            len(self.joint_qpos) + 
            3 + 4 + 
            len(self.joint_qpos) + 
            1  # Phase-Index (normalisiert)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            return self.env.render()
        return None

    def _get_target_pose(self, idx):
        """Holt die Zielpose für eine bestimmte Phase"""
        return BURPEE_V5[idx % len(BURPEE_V5)]

    def _reference_control(self):
        """Holt die Referenz für die aktuelle Phase"""
        pose = self._get_target_pose(self.phase_idx)
        return deg(pose["joints"]).astype(np.float32)

    def _policy_action_to_control(self, action):
        ref_ctrl = self._reference_control()
        ctrl = ref_ctrl + np.asarray(action, dtype=np.float32) * self.action_scale
        return np.clip(ctrl, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

    def _apply_root_assist(self):
        if self.root_assist <= 0.0:
            return
        pose = self._get_target_pose(self.phase_idx)
        amount = np.clip(self.root_assist, 0.0, 1.0)
        
        target_root_pos = np.array(pose["root_pos"])
        target_root_quat = self._quat_from_euler(pose["root_euler"])
        
        root_slice = slice(self.root_qpos, self.root_qpos + 7)
        target_root = np.concatenate([target_root_pos, target_root_quat])
        self.data.qpos[root_slice] = (
            (1.0 - amount) * self.data.qpos[root_slice] + amount * target_root
        )
        self.data.qvel[self.root_qvel:self.root_qvel+6] *= 1.0 - amount * 0.5

    def _quat_from_euler(self, euler_deg):
        roll, pitch, yaw = np.radians(euler_deg)
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ])

    def _apply_target_pose(self, idx):
        """Setzt die Zielpose (für Reset)"""
        pose = self._get_target_pose(idx)
        joints_rad = deg(pose["joints"])
        
        for i, qpos_addr in enumerate(self.joint_qpos):
            if i < len(joints_rad):
                self.data.qpos[qpos_addr] = joints_rad[i]
        
        self.data.qpos[self.root_qpos:self.root_qpos+3] = pose["root_pos"]
        root_quat = self._quat_from_euler(pose["root_euler"])
        self.data.qpos[self.root_qpos+3:self.root_qpos+7] = root_quat
        
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        self.elapsed = 0.0
        self.previous_action[:] = 0.0
        
        # Phasen zurücksetzen
        self.phase_idx = 0
        self.phase_start_time = 0.0
        self.phase_hold_time = 0.0
        self.phase_steps = 0
        self.pose_held = False
        self.phase_complete = False
        
        self._apply_target_pose(0)
        return self._get_obs(), info

    def _get_obs(self):
        pose = self._get_target_pose(self.phase_idx)
        target_joints = deg(pose["joints"])
        
        joint_error = self.data.qpos[self.joint_qpos] - target_joints
        root_pos = self.data.qpos[self.root_qpos:self.root_qpos+3]
        root_quat = self.data.qpos[self.root_qpos+3:self.root_qpos+7]
        target_root_pos = np.array(pose["root_pos"])
        target_root_quat = self._quat_from_euler(pose["root_euler"])
        
        # Phase als normalisierte Zahl
        phase_norm = self.phase_idx / len(BURPEE_V5)
        
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel,
            joint_error,
            root_pos - target_root_pos,
            root_quat - target_root_quat,
            target_joints,
            [phase_norm],
        ])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        ctrl = self._policy_action_to_control(action)
        self.data.ctrl[:] = ctrl
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.elapsed += self.dt
        self.phase_steps += 1
        self._apply_root_assist()

        # Prüfen ob die Pose erreicht wurde
        pose = self._get_target_pose(self.phase_idx)
        target_joints = deg(pose["joints"])
        joint_error = self.data.qpos[self.joint_qpos] - target_joints
        pose_error = np.mean(joint_error * joint_error)
        is_pose_good = pose_error < self.pose_threshold
        
        root_height = self.data.qpos[self.root_qpos+2]
        target_height = pose["root_pos"][2]
        height_error = abs(root_height - target_height)
        is_height_good = height_error < 0.15
        
        # Phase ist erreicht wenn Pose gut UND Höhe gut
        if is_pose_good and is_height_good:
            self.phase_hold_time += self.dt
            if self.phase_hold_time >= self.hold_seconds:
                self.pose_held = True
                self.phase_complete = True
        else:
            self.phase_hold_time = 0.0
            self.pose_held = False

        # Belohnung
        reward = self._compute_reward(action, ctrl, pose_error, height_error)
        self.previous_action = action.copy()
        
        # Phase wechseln wenn erfolgreich oder zu lange
        if self.phase_complete or self.phase_steps > self.max_steps_per_phase:
            self.phase_idx = (self.phase_idx + 1) % len(BURPEE_V5)
            self.phase_start_time = self.elapsed
            self.phase_hold_time = 0.0
            self.phase_steps = 0
            self.pose_held = False
            self.phase_complete = False
            
            # Bei neuem Phase-Reset die Pose setzen
            self._apply_target_pose(self.phase_idx)
        
        # Termination: Nur wenn Roboter komplett gefallen ist
        terminated = root_height < 0.05
        
        # Episode ist zu Ende wenn alle Phasen einmal durchlaufen wurden
        truncated = self.phase_idx == 0 and self.phase_steps > self.max_steps_per_phase and self.elapsed > 5.0
        
        return self._get_obs(), float(reward), terminated, truncated, {
            "root_height": float(root_height),
            "phase": BURPEE_V5[self.phase_idx]["name"],
            "phase_idx": self.phase_idx,
            "pose_error": float(pose_error),
            "pose_held": self.pose_held,
            "phase_complete": self.phase_complete
        }

    def _compute_reward(self, action, ctrl, pose_error, height_error):
        ref_ctrl = self._reference_control()
        ctrl_error = ctrl - ref_ctrl
        action_delta = action - self.previous_action
        
        # Pose Reward
        pose_reward = np.exp(-5.0 * pose_error)
        height_reward = np.exp(-3.0 * height_error)
        
        # Kontroll-Kosten
        control_reward = np.exp(-0.35 * np.mean(ctrl_error * ctrl_error))
        smooth_reward = np.exp(-0.1 * np.mean(action_delta * action_delta))
        
        # HOLD BONUS - wenn die Pose gehalten wurde
        hold_bonus = 0.0
        if self.pose_held:
            hold_bonus = 3.0
        
        # Phase-Completed Bonus
        phase_bonus = 2.0 if self.phase_complete else 0.0
        
        # Alive Bonus
        root_height = self.data.qpos[self.root_qpos+2]
        alive_bonus = 0.5 if root_height > 0.15 else -1.0
        
        return (2.0 * pose_reward + 
                1.0 * height_reward + 
                0.5 * control_reward + 
                0.25 * smooth_reward +
                hold_bonus +
                phase_bonus +
                alive_bonus)

    def close(self):
        self.env.close()