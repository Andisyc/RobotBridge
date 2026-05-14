import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _safe_stem(name: Optional[str], fallback: str = "motion") -> str:
    if not name:
        return fallback
    stem = Path(str(name)).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return stem or fallback


class MotionSequenceRecorder:
    """Record a RobotBridge rollout as an IsaacLab/MOSAIC-style motion npz."""

    REQUIRED_KEYS = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    def __init__(self, cfg: Any, simulator: Any, project_root: Optional[str] = None):
        self.cfg = cfg or {}
        self.simulator = simulator
        self.project_root = Path(project_root or os.getcwd())
        self.enabled = bool(_cfg_get(self.cfg, "enabled", False))
        self.output_dir = self._resolve_output_dir(_cfg_get(self.cfg, "output_dir", "./recorded_motions"))
        self.prefix = str(_cfg_get(self.cfg, "prefix", "robotbridge"))
        self.min_frames = int(_cfg_get(self.cfg, "min_frames", 2))
        self.record_initial_frame = bool(_cfg_get(self.cfg, "record_initial_frame", True))
        self.include_incomplete = bool(_cfg_get(self.cfg, "include_incomplete", True))
        self._warned_no_fk = False
        self.sequence_index = 0
        self.frames = {key: [] for key in self.REQUIRED_KEYS}
        self.source_name = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[MotionRecorder] Recording RobotBridge rollouts to {self.output_dir}")

    def _resolve_output_dir(self, output_dir: str) -> Path:
        path = Path(os.path.expanduser(str(output_dir)))
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def start_sequence(self, source_name: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self.frames = {key: [] for key in self.REQUIRED_KEYS}
        self.source_name = source_name

    def record(self, motion_loader: Any = None) -> None:
        if not self.enabled:
            return
        sim = self.simulator
        fk_info = getattr(sim, "robot_fk_info", None)
        if fk_info is None:
            try:
                sim.get_state()
                fk_info = getattr(sim, "robot_fk_info", None)
            except Exception as exc:
                if not self._warned_no_fk:
                    logger.warning(f"[MotionRecorder] Cannot record FK state yet: {exc}")
                    self._warned_no_fk = True
                return
        if fk_info is None:
            if not self._warned_no_fk:
                logger.warning("[MotionRecorder] simulator.robot_fk_info is missing. Is control.update_with_fk enabled?")
                self._warned_no_fk = True
            return

        fk_info = np.asarray(fk_info, dtype=np.float32)
        if fk_info.ndim != 2 or fk_info.shape[1] < 13:
            raise ValueError(f"Expected robot_fk_info shape (num_bodies, >=13), got {fk_info.shape}.")

        joint_pos = np.asarray(sim.dof_pos, dtype=np.float32).reshape(-1)
        joint_vel = np.asarray(sim.dof_vel, dtype=np.float32).reshape(-1)
        self.frames["joint_pos"].append(joint_pos.copy())
        self.frames["joint_vel"].append(joint_vel.copy())
        self.frames["body_pos_w"].append(fk_info[:, 0:3].copy())
        self.frames["body_quat_w"].append(fk_info[:, 3:7].copy())
        self.frames["body_lin_vel_w"].append(fk_info[:, 7:10].copy())
        self.frames["body_ang_vel_w"].append(fk_info[:, 10:13].copy())

    def _infer_fps(self, motion_loader: Any = None) -> float:
        cfg_fps = _cfg_get(self.cfg, "fps", None)
        if cfg_fps is not None:
            return float(cfg_fps)
        motion = getattr(motion_loader, "motion", None)
        motion_fps = getattr(motion, "fps", None)
        if motion_fps is not None:
            return float(motion_fps)
        high_dt = getattr(self.simulator, "high_dt", None)
        if high_dt:
            return float(1.0 / high_dt)
        return 50.0

    def _motion_source_name(self, motion_loader: Any = None) -> str:
        if self.source_name:
            return str(self.source_name)
        motion = getattr(motion_loader, "motion", None)
        motion_file = getattr(motion, "motion_file", None) or getattr(motion, "current_file", None)
        if motion_file:
            return str(motion_file)
        motion_path = getattr(motion_loader, "motion_path", None)
        if motion_path:
            return str(motion_path)
        return f"sequence_{self.sequence_index:04d}"

    def save(self, motion_loader: Any = None, complete: bool = True, reason: str = "motion_end") -> Optional[Path]:
        if not self.enabled:
            return None
        num_frames = len(self.frames["joint_pos"])
        if num_frames < self.min_frames:
            self.start_sequence(self.source_name)
            return None
        if (not complete) and (not self.include_incomplete):
            self.start_sequence(self.source_name)
            return None

        source = _safe_stem(self._motion_source_name(motion_loader), fallback=f"sequence_{self.sequence_index:04d}")
        status = "complete" if complete else "terminated"
        out_path = self.output_dir / f"{self.prefix}_{self.sequence_index:04d}_{source}_{status}.npz"
        while out_path.exists():
            self.sequence_index += 1
            out_path = self.output_dir / f"{self.prefix}_{self.sequence_index:04d}_{source}_{status}.npz"

        arrays = {key: np.stack(values, axis=0).astype(np.float32) for key, values in self.frames.items()}
        body_names = getattr(getattr(self.simulator, "kinematic", None), "body_names", [])
        joint_names = getattr(self.simulator, "dof_names", [])
        np.savez_compressed(
            out_path,
            fps=np.asarray(self._infer_fps(motion_loader), dtype=np.float32),
            complete=np.asarray(bool(complete)),
            reason=np.asarray(str(reason)),
            source_motion=np.asarray(str(self._motion_source_name(motion_loader))),
            body_names_all=np.asarray(body_names, dtype=str),
            joint_names=np.asarray(joint_names, dtype=str),
            **arrays,
        )
        logger.info(f"[MotionRecorder] Saved {num_frames} frames to {out_path}")
        self.sequence_index += 1
        self.start_sequence(None)
        return out_path
