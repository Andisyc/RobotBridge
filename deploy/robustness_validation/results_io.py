"""Save / load / summarise experiment results."""
from __future__ import annotations
import json
import os
import csv
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Per-trial result ──────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    success: bool              # True = robot survived T_obs steps after push
    fallen_before_push: bool   # True = robot fell during settle phase
    T_push_step: int           # step within observe phase when push was applied
    zmp_margins_settle: list   # ZMP margin (m) recorded every step during settle phase
    zmp_margins_post: list     # ZMP margin (m) recorded every step after push
    push_dir: list             # (3,) push direction unit vector [dx, dy, 0]


# ── Result container ──────────────────────────────────────────────────────────

class ResultsStore:
    """
    Accumulates TrialResult objects keyed by either:
      legacy:  (epsilon_idx, push_velocity_idx, trial_idx)
      current: (perturbation_mode_idx, epsilon_idx, push_velocity_idx, trial_idx)
    Serialises to results_raw.npz + meta.json.
    """

    def __init__(self, meta: dict):
        self.meta = meta
        # Key: see class docstring → TrialResult
        self._data: dict[tuple, TrialResult] = {}

    def add(self, *args) -> None:
        if len(args) == 4:
            epsilon_idx, push_vel_idx, trial_idx, result = args
            key = (epsilon_idx, push_vel_idx, trial_idx)
        elif len(args) == 5:
            perturbation_mode_idx, epsilon_idx, push_vel_idx, trial_idx, result = args
            key = (perturbation_mode_idx, epsilon_idx, push_vel_idx, trial_idx)
        else:
            raise TypeError(
                "add() expects (epsilon_idx, push_idx, trial_idx, result) or "
                "(mode_idx, epsilon_idx, push_idx, trial_idx, result)"
            )
        self._data[key] = result

    # ── I/O ──────────────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        # meta.json
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

        # results_raw.npz — flatten everything into numpy arrays
        keys_list, success, fallen_before, T_push = [], [], [], []
        settle_lengths, post_lengths = [], []
        all_settle_zmp, all_post_zmp, all_push_dirs = [], [], []

        for key, r in self._data.items():
            keys_list.append(list(key))
            success.append(int(r.success))
            fallen_before.append(int(r.fallen_before_push))
            T_push.append(r.T_push_step)
            settle_lengths.append(len(r.zmp_margins_settle))
            post_lengths.append(len(r.zmp_margins_post))
            all_settle_zmp.extend(r.zmp_margins_settle)
            all_post_zmp.extend(r.zmp_margins_post)
            all_push_dirs.append(r.push_dir)

        np.savez_compressed(
            os.path.join(output_dir, "results_raw.npz"),
            keys=np.array(keys_list, dtype=np.int32),
            success=np.array(success, dtype=np.int8),
            fallen_before=np.array(fallen_before, dtype=np.int8),
            T_push=np.array(T_push, dtype=np.int32),
            settle_lengths=np.array(settle_lengths, dtype=np.int32),
            post_lengths=np.array(post_lengths, dtype=np.int32),
            settle_zmp=np.array(all_settle_zmp, dtype=np.float32),
            post_zmp=np.array(all_post_zmp, dtype=np.float32),
            push_dirs=np.array(all_push_dirs, dtype=np.float32),  # (N, 3)
        )
        print(f"[ResultsStore] Saved {len(self._data)} trials to {output_dir}")
        self.save_summary_csv(output_dir)

    @classmethod
    def load(cls, output_dir: str) -> "ResultsStore":
        with open(os.path.join(output_dir, "meta.json")) as f:
            meta = json.load(f)

        store = cls(meta)
        data = np.load(os.path.join(output_dir, "results_raw.npz"), allow_pickle=False)

        keys   = data["keys"]
        suc    = data["success"]
        fallen = data["fallen_before"]
        T_push = data["T_push"]
        sl     = data["settle_lengths"]
        pl     = data["post_lengths"]
        settle = data["settle_zmp"]
        post   = data["post_zmp"]
        dirs   = data["push_dirs"]

        settle_ptr, post_ptr = 0, 0
        for i, key in enumerate(keys):
            s_len = int(sl[i]);  p_len = int(pl[i])
            store._data[tuple(key.tolist())] = TrialResult(
                success=bool(suc[i]),
                fallen_before_push=bool(fallen[i]),
                T_push_step=int(T_push[i]),
                zmp_margins_settle=settle[settle_ptr:settle_ptr + s_len].tolist(),
                zmp_margins_post=post[post_ptr:post_ptr + p_len].tolist(),
                push_dir=dirs[i].tolist(),
            )
            settle_ptr += s_len
            post_ptr   += p_len

        return store

    # ── Multi-run aggregation ─────────────────────────────────────────────────

    @staticmethod
    def merge_summaries(
        named_summaries: list[tuple[str, dict]],
    ) -> dict:
        """
        Aggregate a list of (motion_name, summary) pairs into a combined dict.

        Returns merged[epsilon_idx][push_vel_idx] = {
            'epsilon':           float,
            'push_velocity':     float,
            # Per-motion values (for individual curve plotting)
            'motion_names':      list[str],
            'rates_per_motion':  list[float],   # recovery rate 0–1
            'zmp_per_motion':    list[float],   # mean ZMP margin (m)
            # Aggregate across motions
            'mean_rate':         float,      # end-to-end success rate, pre-fall counts as failure
            'std_rate':          float,
            'mean_zmp':          float,
            'std_zmp':           float,
            # Pooled CI (binomial, pooled n_total)
            'ci_rate':           float,         # ±1 std, Bernoulli
        }
        """
        if not named_summaries:
            raise ValueError("named_summaries is empty")

        _, ref_summary = named_summaries[0]
        n_eps  = len(ref_summary)
        n_pvel = len(ref_summary[0])

        merged: dict[int, dict[int, dict]] = {}

        for ei in range(n_eps):
            merged[ei] = {}
            for pi in range(n_pvel):
                rates, conditional_rates, pre_fall_rates, zmps, names = [], [], [], [], []
                for name, summary in named_summaries:
                    cell = summary[ei][pi]
                    if not np.isnan(cell["end_to_end_success_rate"]):
                        rates.append(cell["end_to_end_success_rate"])
                        conditional_rates.append(cell["conditional_recovery_rate"])
                        pre_fall_rates.append(cell["pre_fall_rate"])
                        zmps.append(cell["mean_zmp_settle"])
                        names.append(name)

                rates_arr = np.array(rates)
                cond_arr = np.array(conditional_rates, dtype=float)
                pre_arr = np.array(pre_fall_rates, dtype=float)
                zmps_arr  = np.array(zmps)

                # Pooled Bernoulli CI: use end-to-end success rate and total n.
                # Pre-push falls are failures, so all completed trials contribute.
                total_n = sum(
                    summary[ei][pi]["n_total"]
                    for _, summary in named_summaries
                )
                mean_r = float(np.mean(rates_arr)) if len(rates_arr) else float("nan")
                ci = (float(np.sqrt(mean_r * (1 - mean_r) / max(total_n, 1)))
                      if not np.isnan(mean_r) else 0.0)

                merged[ei][pi] = {
                    "epsilon":          ref_summary[ei][pi]["epsilon"],
                    "push_velocity":    ref_summary[ei][pi]["push_velocity"],
                    "motion_names":     names,
                    "rates_per_motion": rates,
                    "conditional_rates_per_motion": conditional_rates,
                    "pre_fall_rates_per_motion": pre_fall_rates,
                    "zmp_per_motion":   zmps,
                    "mean_rate":        mean_r,
                    "std_rate":         float(np.std(rates_arr)) if len(rates_arr) > 1 else 0.0,
                    "mean_conditional_rate": (
                        float(np.nanmean(cond_arr))
                        if len(cond_arr) and np.isfinite(cond_arr).any() else float("nan")
                    ),
                    "mean_pre_fall_rate": (
                        float(np.nanmean(pre_arr))
                        if len(pre_arr) and np.isfinite(pre_arr).any() else float("nan")
                    ),
                    "mean_zmp":         float(np.mean(zmps_arr)) if len(zmps_arr) else float("nan"),
                    "std_zmp":          float(np.std(zmps_arr))  if len(zmps_arr) > 1 else 0.0,
                    "ci_rate":          ci,
                }

        return merged

    # ── Summary ───────────────────────────────────────────────────────────────

    def to_summary(self, mode_idx: int | None = None, perturbation_mode: str | None = None) -> dict:
        """
        Returns a nested dict:
          summary[epsilon_idx][push_vel_idx] = {
              'end_to_end_success_rate': float,  # all trials; pre-push falls count as failure
              'conditional_recovery_rate': float, # only trials that reached the push
              'n_valid':       int,        # trials that reached the push (not pre-fallen)
              'n_total':       int,
              'n_fallen_before': int,
              'pre_fall_rate': float,
              'mean_zmp_settle': float,    # mean ZMP margin during settle phase (m)
              'std_zmp_settle':  float,
          }
        """
        if perturbation_mode is not None:
            modes = self._mode_names()
            if perturbation_mode not in modes:
                raise ValueError(f"Unknown perturbation mode {perturbation_mode!r}; available={modes}")
            mode_idx = modes.index(perturbation_mode)
        elif mode_idx is None and self._has_mode_axis():
            modes = self._mode_names()
            mode_idx = modes.index("composite") if "composite" in modes else 0

        eps_vals  = self.meta["epsilon_values"]
        pvel_vals = self.meta["push_velocities"]
        n_eps  = len(eps_vals)
        n_pvel = len(pvel_vals)

        summary: dict[int, dict[int, dict]] = {}

        for ei in range(n_eps):
            summary[ei] = {}
            for pi in range(n_pvel):
                success_after_push, end_to_end_success, fallen_list, settle_zmp_all = [], [], [], []

                for ti in range(self.meta["n_trials"]):
                    key = (mode_idx, ei, pi, ti) if self._has_mode_axis() else (ei, pi, ti)
                    if key not in self._data:
                        continue
                    r = self._data[key]
                    fallen_list.append(r.fallen_before_push)
                    end_to_end_success.append(int((not r.fallen_before_push) and r.success))
                    if not r.fallen_before_push:
                        success_after_push.append(int(r.success))
                    settle_zmp_all.extend(r.zmp_margins_settle)

                n_total  = len(fallen_list)
                n_valid  = len(success_after_push)
                n_fallen = len(fallen_list) - n_valid
                end_rate = float(np.mean(end_to_end_success)) if n_total > 0 else float("nan")
                rec_rate = float(np.mean(success_after_push)) if n_valid > 0 else float("nan")
                pre_rate = float(n_fallen / n_total) if n_total > 0 else float("nan")
                mean_zmp = float(np.mean(settle_zmp_all)) if settle_zmp_all else float("nan")
                std_zmp  = float(np.std(settle_zmp_all))  if settle_zmp_all else float("nan")

                summary[ei][pi] = {
                    "epsilon":          eps_vals[ei],
                    "push_velocity":    pvel_vals[pi],
                    "end_to_end_success_rate": end_rate,
                    "conditional_recovery_rate": rec_rate,
                    # Backward-compatible alias.  Plotting now treats pre-push
                    # falls as failures through end_to_end_success_rate.
                    "recovery_rate":    end_rate,
                    "n_valid":          n_valid,
                    "n_total":          n_total,
                    "n_fallen_before":  n_fallen,
                    "pre_fall_rate":    pre_rate,
                    "mean_zmp_settle":  mean_zmp,
                    "std_zmp_settle":   std_zmp,
                }

        return summary

    def _has_mode_axis(self) -> bool:
        return any(len(key) == 4 for key in self._data)

    def _mode_names(self) -> list[str]:
        modes = self.meta.get("perturbation_modes")
        if modes:
            return list(modes)
        return [self.meta.get("perturbation_mode", "composite")]

    def save_summary_csv(self, output_dir: str, filename: str = "summary.csv") -> None:
        """Write a compact per-condition summary next to the raw results."""
        motion_name = self.meta.get("motion_name") or os.path.splitext(os.path.basename(self.meta.get("motion", "")))[0]
        motion_group = self.meta.get("motion_group", "Ungrouped")
        modes = self._mode_names()

        rows: list[dict[str, Any]] = []
        for idx, perturbation_mode in enumerate(modes):
            summary = self.to_summary(mode_idx=idx if self._has_mode_axis() else None)
            for ei in sorted(summary):
                for pi in sorted(summary[ei]):
                    cell = summary[ei][pi]
                    rows.append({
                        "motion_group": motion_group,
                        "motion_name": motion_name,
                        "perturbation_mode": perturbation_mode,
                        "epsilon_idx": ei,
                        "push_velocity_idx": pi,
                        "epsilon": cell["epsilon"],
                        "push_velocity": cell["push_velocity"],
                        "n_total": cell["n_total"],
                        "n_valid": cell["n_valid"],
                        "n_fallen_before": cell["n_fallen_before"],
                        "end_to_end_success_rate": cell["end_to_end_success_rate"],
                        "conditional_recovery_rate": cell["conditional_recovery_rate"],
                        "pre_fall_rate": cell["pre_fall_rate"],
                        "mean_zmp_settle": cell["mean_zmp_settle"],
                        "std_zmp_settle": cell["std_zmp_settle"],
                    })

        path = os.path.join(output_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
                "motion_group", "motion_name", "perturbation_mode", "epsilon_idx",
                "push_velocity_idx", "epsilon", "push_velocity", "n_total", "n_valid",
                "n_fallen_before", "end_to_end_success_rate", "conditional_recovery_rate",
                "pre_fall_rate", "mean_zmp_settle", "std_zmp_settle",
            ])
            writer.writeheader()
            writer.writerows(rows)
