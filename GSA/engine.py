"""Adapter between Saltelli samples and the existing benchmark runner.

This module contains the code executed by worker processes.  For every row of
the sampling design it copies the selected benchmark cases, applies the
sampled values, runs the unmodified muscle model and returns compact arrays to
the parent process.  Keeping this layer separate from :mod:`GSA.run_gsa`
avoids sending full benchmark objects back and forth for every evaluation.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any

# Each GSA worker runs one independent model evaluation.  Restricting BLAS to
# one thread prevents nested parallelism when several workers are active.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_trials import all_trials
from run_benchmark import build_case, run_case

from GSA.config import GSABlock
from GSA.curves import CurveBasis, apply_curve_modes


# The order is part of the saved-array format and is recorded in metadata.
METRIC_NAMES = (
    "mAE_pct_ref",
    "MAE_pct_ref",
    "R2",
    "peak",
    "time_to_peak_s",
    "impulse",
    "rise_10_90_s",
    "half_decay_s",
)

# Worker-local state.  ``ProcessPoolExecutor`` calls ``initialise_worker`` once
# when a process starts, so benchmark cases are built once rather than for
# every Saltelli row.
_BLOCK: GSABlock | None = None
_CASES: dict[str, dict[str, Any]] = {}
_CURVE_BASIS: CurveBasis | None = None
_ANALYSIS_DT: float = 0.002


def initialise_worker(
    block: GSABlock,
    curve_basis_dict: dict[str, Any] | None,
    analysis_dt: float,
) -> None:
    """Build benchmark cases and cache immutable run settings in a worker."""

    global _BLOCK, _CASES, _CURVE_BASIS, _ANALYSIS_DT
    _BLOCK = block
    _CURVE_BASIS = CurveBasis.from_dict(curve_basis_dict) if curve_basis_dict else None
    _ANALYSIS_DT = float(analysis_dt)
    _CASES = {name: build_case(name, all_trials[name].copy()) for name in block.trials}


def analysis_indices(time: np.ndarray, analysis_dt: float) -> np.ndarray:
    """Select a compact, approximately uniform time grid for saved traces.

    Simulations retain their native integration/output grid for scalar error
    metrics.  Only the arrays used by the time-resolved analysis are thinned,
    which substantially reduces disk use for large Saltelli designs.  The
    final sample is always retained.
    """

    if len(time) < 2:
        return np.array([0], dtype=int)
    native_dt = float(np.median(np.diff(time)))
    stride = max(1, int(round(analysis_dt / native_dt)))
    idx = np.arange(0, len(time), stride, dtype=int)
    if idx[-1] != len(time) - 1:
        idx = np.append(idx, len(time) - 1)
    return idx


def trial_shapes(block: GSABlock, analysis_dt: float) -> dict[str, dict[str, Any]]:
    """Determine output-array dimensions before memory maps are allocated."""

    shapes: dict[str, dict[str, Any]] = {}
    for name in block.trials:
        case = build_case(name, all_trials[name].copy())
        idx = analysis_indices(case["time"], analysis_dt)
        shapes[name] = {
            "time": np.asarray(case["time"])[idx],
            "n_time": int(len(idx)),
            "signal": block.signal,
        }
    return shapes


def _apply_values(case: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    """Return an isolated case with one physical parameter sample applied.

    Parameters ending in ``_scale`` or ``_delta_deg`` describe uncertainty
    relative to a trial-specific baseline; all other names map directly to a
    state or a model parameter.  Only the mutable dictionaries are copied,
    leaving experimental arrays shared and read-only.
    """

    new_case = case.copy()
    new_case["parameters"] = case["parameters"].copy()
    new_case["states"] = case["states"].copy()

    for name, value in values.items():
        if name.startswith("f1_mode_") or name.startswith("f2_mode_"):
            continue
        value = float(value)
        if name == "l_M_opt_scale":
            new_case["parameters"]["l_M_opt"] = case["parameters"]["l_M_opt"] * value
        elif name == "l_T_slack_scale":
            new_case["parameters"]["l_T_slack"] = case["parameters"]["l_T_slack"] * value
        elif name == "l_M_0_scale":
            new_case["states"]["l_M_0"] = case["states"]["l_M_0"] * value
        elif name == "alpha_0_deg":
            new_case["parameters"]["alpha_0"] = np.deg2rad(value)
        elif name == "alpha_0_delta_deg":
            new_case["parameters"]["alpha_0"] = max(
                0.0, case["parameters"]["alpha_0"] + np.deg2rad(value)
            )
        elif name in new_case["states"]:
            new_case["states"][name] = value
        else:
            new_case["parameters"][name] = value

    if _CURVE_BASIS is not None:
        apply_curve_modes(new_case["parameters"], values, _CURVE_BASIS)
    return new_case


def _experimental_on_model_time(case: dict[str, Any], signal_name: str) -> np.ndarray:
    """Place experimental force or calcium data on the model output grid."""

    time = np.asarray(case["time"], dtype=float)
    if signal_name == "force":
        exp = np.asarray(case["exp_force"], dtype=float).ravel()
        n = min(len(time), len(exp))
        aligned = np.full(len(time), np.nan)
        aligned[:n] = exp[:n]
        return aligned

    exp_ca = np.asarray(case["exp_ca"], dtype=float)
    exp_time = (exp_ca[:, 0] - exp_ca[0, 0]) * 1e-3
    return np.interp(time, exp_time, exp_ca[:, 1], left=np.nan, right=np.nan)


def _benchmark_signals(case: dict[str, Any], out: dict[str, Any], signal_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return comparable simulated/experimental signals and an error scale.

    Calcium is converted to the units of the digitised measurements.  Dynamic
    force trials follow the same MVC normalisation rules as the benchmark
    plotting and error code, including the special handling of the 120 Hz
    caudofemoralis condition.
    """

    if signal_name == "Ca":
        sim = np.asarray(out["Ca"], dtype=float) * 1e6
        exp = _experimental_on_model_time(case, signal_name)
        reference = float(np.nanmax(np.abs(exp)))
        return sim, exp, reference

    sim = np.asarray(out["force"], dtype=float).copy()
    exp = _experimental_on_model_time(case, signal_name)
    if case["normalise_dynamic_force"]:
        i = int(case["mvc_sample"])
        if np.isclose(sim[i], 0.0):
            raise FloatingPointError("Cannot normalise dynamic force by a zero MVC sample.")
        sim /= sim[i]
        if not np.isclose(float(case["config"]["freq"]), 120.0):
            if np.isclose(exp[i], 0.0):
                raise FloatingPointError("Cannot normalise experimental force by a zero MVC sample.")
            exp /= exp[i]
        return sim, exp, 1.0
    return sim, exp, float(case["config"]["MVC"])


def _crossing_time(time: np.ndarray, signal: np.ndarray, threshold: float, stop: int) -> float:
    """Return the first threshold crossing before ``stop``, or ``NaN``."""

    idx = np.flatnonzero(signal[: stop + 1] >= threshold)
    return float(time[idx[0]]) if idx.size else np.nan


def compute_metrics(time: np.ndarray, sim: np.ndarray, exp: np.ndarray, reference: float) -> np.ndarray:
    """Compute waveform and timing descriptors for one simulation.

    Error metrics use samples at which either trace is active and are reported
    as percentages of the trial reference force/concentration.  Shape metrics
    are derived from the simulated trace because they become the scalar model
    outputs in the Sobol analysis.
    """

    finite = np.isfinite(sim) & np.isfinite(exp)
    active = finite & ((sim != 0) | (exp != 0))
    if not np.any(active) or not np.isfinite(reference) or np.isclose(reference, 0.0):
        return np.full(len(METRIC_NAMES), np.nan)

    abs_error = np.abs(sim[active] - exp[active]) / reference * 100.0
    m_a_e = float(np.mean(abs_error))
    max_a_e = float(np.max(abs_error))
    exp_active = exp[active]
    sim_active = sim[active]
    ss_res = float(np.sum((sim_active - exp_active) ** 2))
    ss_tot = float(np.sum((exp_active - np.mean(exp_active)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    finite_sim = np.where(np.isfinite(sim), sim, -np.inf)
    peak_idx = int(np.argmax(finite_sim))
    peak = float(sim[peak_idx])
    time_to_peak = float(time[peak_idx])
    valid_sim = np.where(np.isfinite(sim), sim, 0.0)
    impulse = float(np.trapezoid(valid_sim, time))

    baseline = float(valid_sim[0])
    amplitude = peak - baseline
    if amplitude > 0:
        t10 = _crossing_time(time, valid_sim, baseline + 0.1 * amplitude, peak_idx)
        t90 = _crossing_time(time, valid_sim, baseline + 0.9 * amplitude, peak_idx)
        rise = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan
        after = np.flatnonzero(valid_sim[peak_idx:] <= baseline + 0.5 * amplitude)
        half_decay = float(time[peak_idx + after[0]] - time[peak_idx]) if after.size else np.nan
    else:
        rise = np.nan
        half_decay = np.nan

    return np.array([m_a_e, max_a_e, r2, peak, time_to_peak, impulse, rise, half_decay])


def evaluate_sample(task: tuple[int, np.ndarray]) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    """Evaluate one physical Saltelli row for every trial in the active block."""

    sample_index, sample = task
    if _BLOCK is None:
        raise RuntimeError("Worker was not initialised.")
    values = {spec.name: float(value) for spec, value in zip(_BLOCK.parameters, sample)}
    results: dict[str, dict[str, np.ndarray]] = {}

    for name in _BLOCK.trials:
        case = _apply_values(_CASES[name], values)
        # The benchmark runner prints one status line per simulation.  Those
        # messages would overwhelm the useful progress report during a GSA.
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_case(case)
        sim, exp, reference = _benchmark_signals(case, out, _BLOCK.signal)
        if not np.all(np.isfinite(sim)):
            raise FloatingPointError(f"Non-finite {_BLOCK.signal} in trial {name}.")
        idx = analysis_indices(case["time"], _ANALYSIS_DT)
        # MVC normalisation removes the trivial force-scale contribution from
        # the temporal fingerprints while leaving the raw signal available.
        if _BLOCK.signal == "force" and not case["normalise_dynamic_force"]:
            shape_reference = float(case["parameters"]["MVC"])
        else:
            shape_reference = reference
        if np.isclose(shape_reference, 0.0):
            raise FloatingPointError(f"Zero normalization reference in trial {name}.")
        results[name] = {
            "signal": sim[idx],
            "signal_normalized": sim[idx] / shape_reference,
            "metrics": compute_metrics(np.asarray(case["time"]), sim, exp, reference),
        }
    return sample_index, results
