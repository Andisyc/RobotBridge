import os
import re
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import mujoco
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


class MuJoCoVideoRecorder:
    """Save RobotBridge MuJoCo rollouts as mp4 videos."""

    def __init__(self, cfg: Any, simulator: Any, project_root: Optional[str] = None):
        self.cfg = cfg or {}
        self.simulator = simulator
        self.project_root = Path(project_root or os.getcwd())
        self.enabled = bool(_cfg_get(self.cfg, "enabled", False))
        self.output_dir = self._resolve_output_dir(_cfg_get(self.cfg, "output_dir", "./videos"))
        self.prefix = str(_cfg_get(self.cfg, "prefix", "robotbridge"))
        self.fps = float(_cfg_get(self.cfg, "fps", 30.0))
        self.width = int(_cfg_get(self.cfg, "width", 1280))
        self.height = int(_cfg_get(self.cfg, "height", 720))
        self.render_every = int(_cfg_get(self.cfg, "render_every", 0) or 0)
        self.record_initial_frame = bool(_cfg_get(self.cfg, "record_initial_frame", True))
        self.include_incomplete = bool(_cfg_get(self.cfg, "include_incomplete", True))
        self.track_body_name = _cfg_get(self.cfg, "track_body_name", "torso_link")
        self.distance = float(_cfg_get(self.cfg, "distance", 3.0))
        self.azimuth = float(_cfg_get(self.cfg, "azimuth", 180.0))
        self.elevation = float(_cfg_get(self.cfg, "elevation", -12.0))
        self.lookat_height = float(_cfg_get(self.cfg, "lookat_height", 0.75))
        self.min_frames = int(_cfg_get(self.cfg, "min_frames", 2))

        self.renderer = None
        self.camera = None
        self.frames = []
        self.step_counter = 0
        self.sequence_index = 0
        self.source_name = None
        self._track_body_id = -1

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._init_renderer()
            logger.info(f"[VideoRecorder] Recording RobotBridge mp4 videos to {self.output_dir}")

    def _resolve_output_dir(self, output_dir: str) -> Path:
        path = Path(os.path.expanduser(str(output_dir)))
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _init_renderer(self) -> None:
        self.renderer = mujoco.Renderer(self.simulator.mujoco_model, height=self.height, width=self.width)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = self.distance
        self.camera.azimuth = self.azimuth
        self.camera.elevation = self.elevation
        if self.track_body_name:
            self._track_body_id = mujoco.mj_name2id(
                self.simulator.mujoco_model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(self.track_body_name),
            )
            if self._track_body_id < 0:
                logger.warning(f"[VideoRecorder] track_body_name not found: {self.track_body_name}")

    def _infer_render_every(self) -> int:
        if self.render_every > 0:
            return self.render_every
        high_dt = getattr(self.simulator, "high_dt", None)
        if not high_dt:
            return 1
        return max(1, int(round(1.0 / (self.fps * float(high_dt)))))

    def _motion_source_name(self, motion_loader: Any = None) -> str:
        if self.source_name:
            return str(self.source_name)
        motion = getattr(motion_loader, "motion", None)
        motion_file = getattr(motion, "motion_file", None) or getattr(motion, "current_file", None)
        if motion_file:
            return str(motion_file)
        return f"sequence_{self.sequence_index:04d}"

    def start_sequence(self, source_name: Optional[str] = None) -> None:
        if not self.enabled:
            return
        self.frames = []
        self.step_counter = 0
        self.source_name = source_name

    def _update_camera(self) -> None:
        if self.camera is None:
            return
        data = self.simulator.mujoco_data
        if self._track_body_id >= 0:
            lookat = np.asarray(data.xpos[self._track_body_id], dtype=np.float64).copy()
        else:
            lookat = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        lookat[2] = self.lookat_height
        self.camera.lookat[:] = lookat
        self.camera.distance = self.distance
        self.camera.azimuth = self.azimuth
        self.camera.elevation = self.elevation

    def capture(self) -> None:
        if not self.enabled:
            return
        render_every = self._infer_render_every()
        if self.step_counter % render_every != 0:
            self.step_counter += 1
            return
        if self.renderer is None:
            self._init_renderer()
        self._update_camera()
        self.renderer.update_scene(self.simulator.mujoco_data, camera=self.camera)
        self.frames.append(self.renderer.render().copy())
        self.step_counter += 1

    def save(self, motion_loader: Any = None, complete: bool = True, reason: str = "motion_end") -> Optional[Path]:
        if not self.enabled:
            return None
        if len(self.frames) < self.min_frames:
            self.start_sequence(self.source_name)
            return None
        if (not complete) and (not self.include_incomplete):
            self.start_sequence(self.source_name)
            return None

        source = _safe_stem(self._motion_source_name(motion_loader), fallback=f"sequence_{self.sequence_index:04d}")
        status = "complete" if complete else "terminated"
        out_path = self.output_dir / f"{self.prefix}_{self.sequence_index:04d}_{source}_{status}.mp4"
        while out_path.exists():
            self.sequence_index += 1
            out_path = self.output_dir / f"{self.prefix}_{self.sequence_index:04d}_{source}_{status}.mp4"

        imageio.mimsave(out_path, self.frames, fps=self.fps, macro_block_size=1)
        logger.info(f"[VideoRecorder] Saved {len(self.frames)} frames to {out_path} ({reason})")
        self.sequence_index += 1
        self.start_sequence(None)
        return out_path

    def close(self) -> None:
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
