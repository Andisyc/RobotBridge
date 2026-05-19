"""Runtime adapter for applying MOSAIC FrontRES checkpoints in RobotBridge."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn


def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    needle = prefix + "."
    return {key[len(needle):]: value for key, value in state_dict.items() if key.startswith(needle)}


def _linear_layers_from_state(state_dict: Mapping[str, torch.Tensor]) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    def _sort_key(key: str):
        return [int(part) if part.isdigit() else part for part in key.split(".")]

    layers: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for key in sorted(state_dict.keys(), key=_sort_key):
        if not key.endswith(".weight"):
            continue
        stem = key[:-len(".weight")]
        bias_key = stem + ".bias"
        if bias_key in state_dict and getattr(state_dict[key], "ndim", 0) == 2:
            layers.append((stem, state_dict[key], state_dict[bias_key]))
    return layers


def _build_mlp(state_dict: Mapping[str, torch.Tensor], activation: str = "elu") -> nn.Sequential:
    layers = _linear_layers_from_state(state_dict)
    if not layers:
        raise ValueError("Could not infer residual_actor MLP layers from checkpoint.")

    modules: list[nn.Module] = []
    for idx, (_name, weight, _bias) in enumerate(layers):
        modules.append(nn.Linear(int(weight.shape[1]), int(weight.shape[0])))
        if idx != len(layers) - 1:
            if activation.lower() == "relu":
                modules.append(nn.ReLU())
            elif activation.lower() == "tanh":
                modules.append(nn.Tanh())
            else:
                modules.append(nn.ELU())

    model = nn.Sequential(*modules)
    model_state = model.state_dict()
    translated: dict[str, torch.Tensor] = {}
    module_idx = 0
    for _name, weight, bias in layers:
        while f"{module_idx}.weight" not in model_state:
            module_idx += 1
        translated[f"{module_idx}.weight"] = weight
        translated[f"{module_idx}.bias"] = bias
        module_idx += 1
    model.load_state_dict(translated, strict=True)
    return model


def _normalizer_stats(checkpoint: Mapping) -> tuple[np.ndarray | None, np.ndarray | None]:
    for key in ("obs_norm_state_dict", "obs_normalizer_state_dict", "obs_normalizer", "normalizer"):
        state = checkpoint.get(key)
        if not isinstance(state, Mapping):
            continue
        mean = state.get("mean", state.get("_mean", state.get("running_mean")))
        var = state.get("var", state.get("_var", state.get("running_var")))
        if mean is None or var is None:
            continue
        mean_np = np.asarray(mean.detach().cpu() if torch.is_tensor(mean) else mean, dtype=np.float32).reshape(-1)
        var_np = np.asarray(var.detach().cpu() if torch.is_tensor(var) else var, dtype=np.float32).reshape(-1)
        return mean_np, var_np
    return None, None


class FrontRESRuntime:
    """Small inference wrapper for MOSAIC FrontRES residual_actor checkpoints."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        history_length: int = 5,
        max_delta_pos: float = 0.3,
        max_delta_rpy: float = 0.1,
        allow_upward_dz: bool = False,
        ignore_conf: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = torch.device(device)
        self.history_length = int(history_length)
        self.max_delta_pos = float(max_delta_pos)
        self.max_delta_rpy = float(max_delta_rpy)
        self.allow_upward_dz = bool(allow_upward_dz)
        self.ignore_conf = bool(ignore_conf)

        checkpoint_obj = torch.load(self.checkpoint, map_location=self.device)
        model_state = checkpoint_obj.get("model_state_dict", checkpoint_obj)
        residual_state = _strip_prefix(model_state, "residual_actor")
        if not residual_state and isinstance(model_state.get("residual_actor"), Mapping):
            residual_state = dict(model_state["residual_actor"])
        if not residual_state:
            residual_state = {
                key: value
                for key, value in model_state.items()
                if key.endswith(".weight") or key.endswith(".bias")
            }

        self.actor = _build_mlp(residual_state).to(self.device)
        self.actor.eval()
        self.mean, self.var = _normalizer_stats(checkpoint_obj)
        self.history = deque(maxlen=self.history_length)
        self.last_delta = np.zeros(6, dtype=np.float32)

    def reset(self) -> None:
        self.history.clear()
        self.last_delta[:] = 0.0

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.mean is None or self.var is None:
            return obs.astype(np.float32)
        out = obs.astype(np.float32).copy()
        if self.mean.shape[0] == out.shape[0]:
            return (out - self.mean) / np.sqrt(self.var + 1e-8)
        if self.mean.shape[0] < out.shape[0]:
            tail = self.mean.shape[0]
            out[-tail:] = (out[-tail:] - self.mean) / np.sqrt(self.var + 1e-8)
        return out

    def compute(self, env, obs_buf_dict: Mapping[str, np.ndarray]) -> np.ndarray:
        gmt_obs = np.asarray(obs_buf_dict["obs"], dtype=np.float32).reshape(-1)
        current_error = np.asarray(
            getattr(env, "frontres_anchor_error", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(6)
        self.history.append(current_error)

        pad_count = max(0, self.history_length - len(self.history))
        history_flat = np.concatenate([np.zeros(6, dtype=np.float32)] * pad_count + list(self.history), axis=0)
        frontres_obs = np.concatenate([history_flat, gmt_obs], axis=0)
        norm_obs = self._normalize(frontres_obs)

        with torch.no_grad():
            raw = self.actor(torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
            raw_np = raw.detach().cpu().numpy().astype(np.float32)

        delta_pos = np.tanh(raw_np[:3]) * self.max_delta_pos
        delta_rpy = np.tanh(raw_np[3:6]) * self.max_delta_rpy
        if raw_np.shape[0] >= 8 and not self.ignore_conf:
            delta_pos *= 1.0 / (1.0 + np.exp(-raw_np[6]))
            delta_rpy *= 1.0 / (1.0 + np.exp(-raw_np[7]))
        if not self.allow_upward_dz:
            delta_pos[2] = min(float(delta_pos[2]), 0.0)

        self.last_delta = np.concatenate([delta_pos, delta_rpy]).astype(np.float32)
        return self.last_delta.copy()
