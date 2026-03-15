import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.base_env import BaseEnv
from utils.dataset import MotionDataset
from utils.motion_lib.rotations import quat_rotate_inverse

ANKLE_IDX = [4, 5, 10, 11]
ACTION_SCALE = 0.5
DEFAULT_TAR_OBS_STEPS = [
    1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
]

def euler_from_quat_gmt(quat_xyzw: torch.Tensor):
    """quat in xyzw to rpy"""
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


def quat_to_euler_wxyz_gmt(quat_wxyz: Sequence[float]) -> np.ndarray:
    """quat in wxyz to rpy"""
    qw, qx, qy, qz = quat_wxyz
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(project_root, expanded))


def _default_dof_pos(
    robot_cfg: Dict,
    asset_cfg: DictConfig,
) -> np.ndarray:
    asset_dict = (
        OmegaConf.to_container(asset_cfg, resolve=True)
        if isinstance(asset_cfg, DictConfig)
        else dict(asset_cfg)
    )
    default_angles = list(asset_dict.get("default_angles", []))
    num_dof = int(asset_dict.get("num_dof", len(default_angles)))
    default_dof_pos = robot_cfg.get("default_dof_pos") if robot_cfg else None
    if default_dof_pos is None:
        if not default_angles:
            raise ValueError("default_angles missing in asset config.")
        default_dof_pos = default_angles
    default_dof_pos = list(default_dof_pos)
    if len(default_dof_pos) != num_dof:
        raise ValueError(
            f"default_dof_pos length {len(default_dof_pos)} does not match expected {num_dof}."
        )
    return np.asarray(default_dof_pos, dtype=np.float32)


class GMTMotionDataset(MotionDataset):
    def __init__(
        self,
        motion_cfg: Dict,
        simulator,
        ref_dof: int = 23,
        total_dof: int = 23,
        gym_idx: bool = True,
        zero_padding_list: Optional[Sequence[int]] = None,
    ):
        zero_padding_list = list(zero_padding_list or [])
        super().__init__(
            motion_cfg,
            simulator=simulator,
            ref_dof=ref_dof,
            total_dof=total_dof,
            gym_idx=gym_idx,
            zero_padding_list=zero_padding_list,
        )
        tar_obs_steps = motion_cfg.get("tar_obs_steps", DEFAULT_TAR_OBS_STEPS)
        self.tar_obs_steps = np.asarray(tar_obs_steps, dtype=np.int32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.motion_root_index = 0

    def get_mimic_obs(self, control_dt: float) -> np.ndarray:
        future_indices = np.clip(
            self.timestep + self.tar_obs_steps,
            0,
            self.motion.time_step_total - 1,
        )

        ref_pos_w = self.motion.body_pos_w[future_indices, self.motion_root_index]
        ref_quat_w = self.motion.body_quat_w[future_indices, self.motion_root_index]
        ref_vel_w = self.motion.body_lin_vel_w[future_indices, self.motion_root_index]
        ref_ang_vel_w = self.motion.body_ang_vel_w[future_indices, self.motion_root_index]
        ref_dof_pos = self.motion.joint_pos[future_indices]

        aligned_pos = self.motion_init_align.align_pos_batch(ref_pos_w)
        aligned_quat_xyzw = self.motion_init_align.align_quat_batch(ref_quat_w[:, [1, 2, 3, 0]])

        aligned_quat_torch = torch.from_numpy(aligned_quat_xyzw).to(self.device)
        roll, pitch, _ = euler_from_quat_gmt(aligned_quat_torch)

        local_vel = quat_rotate_inverse(
            aligned_quat_torch,
            torch.from_numpy(ref_vel_w).to(self.device),
            w_last=True,
        )
        local_ang_vel = quat_rotate_inverse(
            aligned_quat_torch,
            torch.from_numpy(ref_ang_vel_w).to(self.device),
            w_last=True,
        )

        mimic_obs = torch.cat(
            (
                torch.from_numpy(aligned_pos[:, 2:3]).to(self.device),
                roll.unsqueeze(-1),
                pitch.unsqueeze(-1),
                local_vel,
                local_ang_vel[:, 2:3],
                torch.from_numpy(ref_dof_pos).to(self.device),
            ),
            dim=-1,
        )

        return mimic_obs.view(-1).cpu().numpy()


class GMTEnv(BaseEnv):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.config = config

        cfg_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else (config or {})
        self.device = cfg_dict.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_cfg = cfg_dict.get("policy", {}) or {}
        self.motion_cfg = cfg_dict.get("motion", {}) or {}
        self.robot_cfg = cfg_dict.get("robot", {}) or {}

        metrics_path = cfg_dict.get("metrics_path") or self.policy_cfg.get("metrics_path")
        if not metrics_path:
            checkpoint = self.policy_cfg.get("checkpoint")
            if checkpoint:
                ckpt_name = Path(checkpoint).stem
                metrics_path = os.path.join(project_root, "logs", f"metrics_{ckpt_name}.csv")
            else:
                metrics_path = os.path.join(project_root, "logs", "metrics_gmt.csv")
        self.metrics_path = _resolve_path(metrics_path) if metrics_path else None

        self.sim_action_scale = float(getattr(self.simulator.cfg.control, "action_scale", 1.0))
        self.default_angles = np.asarray(self.simulator.default_angles, dtype=np.float32)
        self.active_dof_idx = np.asarray(self.simulator.active_dof_idx, dtype=np.int32)
        self.default_dof_pos = _default_dof_pos(self.robot_cfg, self.simulator.cfg.asset)
        self.default_dof_pos_active = self.default_dof_pos[self.active_dof_idx]
        self.num_action = int(self.simulator.num_action)

        motion_cfg = dict(self.motion_cfg)
        motion_path = _resolve_path(motion_cfg.get("motion_path"))
        if motion_path:
            motion_cfg["motion_path"] = motion_path
        self.motion_cfg = motion_cfg

        ref_dof = int(motion_cfg.get("ref_dof", self.num_action))
        total_dof = int(motion_cfg.get("total_dof", self.num_action))
        gym_idx = bool(motion_cfg.get("gym_idx", True))
        zero_padding_list = list(motion_cfg.get("zero_padding_list", []))

        self.motion_loader = GMTMotionDataset(
            motion_cfg,
            simulator=self.simulator,
            ref_dof=ref_dof,
            total_dof=total_dof,
            gym_idx=gym_idx,
            zero_padding_list=zero_padding_list,
        )
        if self.metrics_path:
            self.motion_loader.set_metrics_file(self.metrics_path)

        policy_path = self.policy_cfg.get("checkpoint") or cfg_dict.get("policy_path")
        self.policy_path = _resolve_path(policy_path) if policy_path else None
        self.policy = None

        import onnxruntime as ort

        # === 魔改开始：支持 ONNX 格式的 GMT 追踪器 ===
        if str(self.policy_path).endswith('.onnx'):
            print(f"[Hack] Loading ONNX policy from {self.policy_path}...")
            class ONNXPolicyWrapper:
                def __init__(self, path, device):
                    self.sess = ort.InferenceSession(str(path))
                    self.input_name = self.sess.get_inputs()[0].name
                    self.device = device
                    
                def __call__(self, obs):
                    # 将 PyTorch tensor 转为 numpy，推理后再转回 tensor
                    obs_np = obs.detach().cpu().numpy().astype(np.float32)
                    act_np = self.sess.run(None, {self.input_name: obs_np})[0]
                    return torch.tensor(act_np, device=self.device)
                    
                def eval(self):
                    pass # 防止后续调用 .eval() 报错
                    
            self.policy = ONNXPolicyWrapper(self.policy_path, self.device)
        else:
            # 兼容原作者的 .pt 写法
            self.policy = torch.jit.load(self.policy_path, map_location=self.device)
        # === 魔改结束 ===
        # if self.policy_path:
        #     self.policy = torch.jit.load(self.policy_path, map_location=self.device)

        self.action_scale = float(self.policy_cfg.get("action_scale", ACTION_SCALE))
        self.action_clip = self.policy_cfg.get("action_clip", None)
        self.history_len = int(self.policy_cfg.get("history_length", 20))
        self.n_proprio = 3 + 2 + 3 * self.num_action

        if self.history_len > 0:
            self.proprio_history = deque(
                [np.zeros(self.n_proprio, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
        else:
            self.proprio_history = deque([], maxlen=0)

        self.last_action = np.zeros(self.num_action, dtype=np.float32)
        self.obs_buf_dict = {}

    def reset(self):
        mujoco.mj_resetData(self.simulator.mujoco_model, self.simulator.mujoco_data)
        self.motion_loader.reset()
        self.motion_loader.cur_motion_end = False
        self.last_action.fill(0)
        if self.history_len > 0:
            self.proprio_history = deque(
                [np.zeros(self.n_proprio, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
        else:
            self.proprio_history = deque([], maxlen=0)
        
        # ==================== 魔改开始：强制同步初始位姿 ====================
        try:
            first_jpos = self.motion_loader.joint_pos[0]
            self.simulator.mujoco_data.qpos[7 : 7 + self.num_action] = first_jpos
            
            # 自动适应矩阵维度：如果是 2D 取 [0]，如果是 3D 取 [0, 0]
            pos_w = np.array(self.motion_loader.body_pos_w)
            quat_w = np.array(self.motion_loader.body_quat_w)
            root_pos = pos_w[0, 0].copy() if pos_w.ndim == 3 else pos_w[0].copy()
            root_quat = quat_w[0, 0] if quat_w.ndim == 3 else quat_w[0]
            
            root_pos[2] += 0.05  # 防穿模抬高 5cm
            
            self.simulator.mujoco_data.qpos[0:3] = root_pos
            self.simulator.mujoco_data.qpos[3:7] = root_quat
            self.simulator.mujoco_data.qvel[:] = 0.0
            
            mujoco.mj_forward(self.simulator.mujoco_model, self.simulator.mujoco_data)
        except Exception as e:
            print(f"[HACK] 初始状态同步失败: {e}")

        # 2. 清空原本填满 0 的错误历史记忆
        if self.history_len > 0:
            self.proprio_history = deque([], maxlen=self.history_len)

        # 3. 【观测记忆预热】：连续采样 N 次！
        # 强行让机器人认为：“我在过去 5 帧里，一直静静地保持着这个舞蹈初始姿势”
        for _ in range(self.history_len):
            obs = self.compute_observation()

        # 新增：我们自己维护一个极其可靠的帧计数器！
        self._my_frame_idx = 0

        self.obs_buf_dict = {"obs": obs}
        return self.obs_buf_dict
        # ==================== 魔改结束 ====================================

        # obs = self.compute_observation()
        # self.obs_buf_dict = {"obs": obs}

        # return self.obs_buf_dict

    def _reset_envs(self, refresh):
        mujoco.mj_resetData(self.simulator.mujoco_model, self.simulator.mujoco_data)
        self.motion_loader.cur_motion_end = False
        self.last_action.fill(0)
        if self.history_len > 0:
            self.proprio_history = deque(
                [np.zeros(self.n_proprio, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
        else:
            self.proprio_history = deque([], maxlen=0)

    def _get_reference_markers_world(self) -> Optional[np.ndarray]:
        body_pos_aligned = self.motion_loader.body_pos_w_aligned
        if body_pos_aligned is None:
            return None
        body_pos_aligned = np.asarray(body_pos_aligned, dtype=np.float32).reshape(-1, 3)
        if body_pos_aligned.size == 0:
            return None
        return body_pos_aligned

    def compute_observation(self) -> np.ndarray:
        self.simulator.get_state()
        sim = self.simulator

        root_quat_wxyz = np.asarray([sim.root_quat[3], sim.root_quat[0], sim.root_quat[1], sim.root_quat[2]])
        rpy = quat_to_euler_wxyz_gmt(root_quat_wxyz)

        obs_dof_vel = sim.dof_vel.copy()
        for idx in ANKLE_IDX:
            if 0 <= idx < obs_dof_vel.shape[0]:
                obs_dof_vel[idx] = 0.0

        obs_prop = np.concatenate(
            [
                sim.base_ang_vel * 0.25,
                rpy[:2],
                (sim.dof_pos - self.default_dof_pos_active),
                obs_dof_vel * 0.05,
                self.last_action,
            ]
        ).astype(np.float32)

        mimic_obs = self.motion_loader.get_mimic_obs(control_dt=self.simulator.high_dt)
        obs_hist = np.array(self.proprio_history, dtype=np.float32).flatten()
        full_obs = np.concatenate([mimic_obs, obs_prop, obs_hist])

        self.proprio_history.append(obs_prop)
        # self.motion_loader._update_metrics()

        if getattr(self.simulator, "marker", False):
            markers_world = self._get_reference_markers_world()
            if markers_world is not None:
                self.simulator.update_marker_pos(markers_world[None, ...])

        return full_obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        # print(f"[Debug] Raw action max: {action.max():.2f}, min: {action.min():.2f}")

        if action.size != self.num_action:
            raise ValueError(f"Expected action size {self.num_action}, got {action.size}.")

        self.last_action = action.copy()
        if self.action_clip is not None:
            action = np.clip(action, -float(self.action_clip), float(self.action_clip))

        action = np.clip(action, -10.0, 10.0)
        
        # 计算目标姿态
        target_dof_pos = action * self.action_scale + self.default_dof_pos_active

        # print(f"[Debug] target_dof_pos: {target_dof_pos}")
        
        # 换算成 MuJoCo 底层指令
        action_cmd = (target_dof_pos - self.default_angles[self.active_dof_idx]) / self.sim_action_scale
        self.simulator.apply_action(action_cmd)

        # === 环境步进与状态更新 ===
        obs = self.compute_observation()
        termination_obs = self._check_termination()
        if termination_obs is not None:
            obs = termination_obs

        self.motion_loader.post_step_callback()
        if self.motion_loader.cur_motion_end:
            self.motion_loader.next_motion(fail=False)
            return self.reset()

        self.obs_buf_dict = {"obs": obs}
        return self.obs_buf_dict
    
        # === 测试一：切除大脑，纯物理 PD 追踪 ===
        
        # # 1. 安全获取当前帧，防止数组越界
        # max_frame = len(self.motion_loader.joint_pos) - 1
        # current_frame = min(self._my_frame_idx, max_frame)
        
        # # 2. 获取标准答案，并让计数器 +1 准备下一帧
        # target_dof_pos = self.motion_loader.joint_pos[current_frame]
        # self._my_frame_idx += 1
        
        # # 3. 换算并发送给 MuJoCo
        # action_cmd = (target_dof_pos - self.default_angles[self.active_dof_idx]) / self.sim_action_scale
        # self.simulator.apply_action(action_cmd)
        
        # # === 以下保持正常流程 ===
        # obs = self.compute_observation()
        # termination_obs = self._check_termination()
        # if termination_obs is not None:
        #     obs = termination_obs
        # self.motion_loader.post_step_callback()
        # if self.motion_loader.cur_motion_end:
        #     self.motion_loader.next_motion(fail=False)
        #     return self.reset()
        # self.obs_buf_dict = {"obs": obs}
        # return self.obs_buf_dict

    def _check_termination(self):
        # hard_reset = self.simulator.check_termination()
        # if hard_reset:
        #     self.next_motion(fail=True)
        #     self._reset_envs(True)
        #     return self.compute_observation()
        return None

    def next_motion(self, fail: bool = False):
        self.motion_loader.next_motion(fail)
        return self.reset()
