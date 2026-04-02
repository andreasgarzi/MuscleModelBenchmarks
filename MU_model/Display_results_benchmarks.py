"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBME
___________________________________

Display MU-model benchmarks results among:
Slow_M: ("slow_M_max", "slow_M_sub", slow_M_len) 
Fast_M: (fast_M_iso, fast_M_len, fast_M_dyn)
MU: (slow_MU, fast_MU)

"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from sklearn.metrics import r2_score
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

####################################################################################
# Helper functions
####################################################################################

def summarize_trials(mae_list, maxae_list, label="", unit="% MVC"):
    """
    mae_list   : iterable di MAE (mean absolute error) per trial
    maxae_list : iterable di MaxAE (max absolute error) per trial
    """
    mae_arr   = np.asarray(mae_list, dtype=float)
    maxae_arr = np.asarray(maxae_list, dtype=float)

    if mae_arr.size == 0 or maxae_arr.size == 0:
        print(f"{label} (no trials)")
        return

    print(f"{label}")
    print(
        f"  MAE   = {mae_arr.mean():6.2f} ± {mae_arr.std(ddof=0):6.2f} {unit}   "
        f"[{mae_arr.min():6.2f} – {mae_arr.max():6.2f}]"
    )
    print(
        f"  MaxAE = {maxae_arr.mean():6.2f} ± {maxae_arr.std(ddof=0):6.2f} {unit}   "
        f"[{maxae_arr.min():6.2f} – {maxae_arr.max():6.2f}]"
    )
    print()

def load_series(folder: Path, names):
    return [np.load(folder / f"{name}.npy") for name in names]

def pct_errors(exp: np.ndarray, sim: np.ndarray, F0: float):
    """
    Compute:
    - mAE  : mean absolute error, in %F0
    - MAE  : maximum absolute error, in %F0
    - stde : standard deviation of absolute error, in %F0

    Only samples where at least one signal is non-zero are considered.
    """
    exp = np.asarray(exp, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()

    mask = (exp != 0) | (sim != 0)
    if not np.any(mask):
        return 0.0, 0.0, 0.0

    exp_nz = exp[mask]
    sim_nz = sim[mask]

    abs_err_pctF0 = np.abs(sim_nz - exp_nz) / F0 * 100.0
    mae = float(np.mean(abs_err_pctF0))
    maxae = float(np.max(abs_err_pctF0))
    stde = float(np.std(abs_err_pctF0))
    return mae, maxae, stde

def compute_r2(exp: np.ndarray, sim: np.ndarray) -> float:
    """
    Compute coefficient of determination (R^2).
    Only samples where at least one signal is non-zero are considered.
    """
    exp = np.asarray(exp, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()

    mask = (exp != 0) | (sim != 0)
    if not np.any(mask):
        return np.nan

    exp_m = exp[mask]
    sim_m = sim[mask]

    ss_res = np.sum((exp_m - sim_m) ** 2)
    ss_tot = np.sum((exp_m - np.mean(exp_m)) ** 2)

    if ss_tot <= 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)

def summarize_trials(mae_list, maxae_list, label="", unit="% F0"):
    """
    Print summary statistics for mAE and MAE across trials.
    """
    mae_arr = np.asarray(mae_list, dtype=float)
    maxae_arr = np.asarray(maxae_list, dtype=float)

    if mae_arr.size == 0 or maxae_arr.size == 0:
        print(f"{label} (no trials)")
        return

    print(f"{label}")
    print(
        f"  mAE  = {mae_arr.mean():6.2f} ± {mae_arr.std(ddof=0):6.2f} {unit}   "
        f"[{mae_arr.min():6.2f} – {mae_arr.max():6.2f}]"
    )
    print(
        f"  MAE  = {maxae_arr.mean():6.2f} ± {maxae_arr.std(ddof=0):6.2f} {unit}   "
        f"[{maxae_arr.min():6.2f} – {maxae_arr.max():6.2f}]"
    )
    print()

def summarize_metric_list(values, label="", unit=""):
    """
    Print summary statistics for a generic metric list, ignoring NaNs.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        print(f"{label} (no valid values)")
        return

    print(
        f"{label}: {arr.mean():6.4f} ± {arr.std(ddof=0):6.4f} {unit}   "
        f"[{arr.min():6.4f} – {arr.max():6.4f}]"
    )

def first_crossing_time(
    time: np.ndarray,
    signal: np.ndarray,
    level: float,
    start_idx: int = 0,
    end_idx: int | None = None,
    direction: str = "rising",
):
    """
    Return the first interpolated crossing time of 'level' in 'signal'.

    Parameters
    ----------
    time : array
        Time vector.
    signal : array
        Signal vector.
    level : float
        Threshold to be crossed.
    start_idx : int
        Index from which to start searching.
    end_idx : int or None
        Final index for the search.
    direction : str
        'rising' for upward crossing, 'falling' for downward crossing.

    Returns
    -------
    crossing_time : float
        Interpolated crossing time, or np.nan if not found.
    """
    time = np.asarray(time, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float).ravel()

    if end_idx is None:
        end_idx = len(signal)

    start_idx = max(1, int(start_idx))
    end_idx = min(int(end_idx), len(signal))

    for i in range(start_idx, end_idx):
        y0 = signal[i - 1]
        y1 = signal[i]

        if direction == "rising":
            crossed = (y0 < level <= y1)
        elif direction == "falling":
            crossed = (y0 > level >= y1)
        else:
            raise ValueError("direction must be 'rising' or 'falling'")

        if crossed:
            if y1 == y0:
                return float(time[i])
            alpha = (level - y0) / (y1 - y0)
            return float(time[i - 1] + alpha * (time[i] - time[i - 1]))

    return np.nan

def first_positive_sample(time: np.ndarray, force: np.ndarray, threshold: float = 0.01):
    """
    Return time, force, and index of the first sample above threshold.
    """
    time = np.asarray(time, dtype=float).ravel()
    force = np.asarray(force, dtype=float).ravel()

    idx = np.where(force > threshold)[0]
    if idx.size == 0:
        return np.nan, np.nan, np.nan

    i0 = int(idx[0])
    return float(time[i0]), float(force[i0]), i0

def peak_metrics(force: np.ndarray, time: np.ndarray, F0: float):
    """
    Return peak force, peak time, peak index, and peak normalized to F0 (%F0).
    """
    force = np.asarray(force, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()

    if force.size == 0:
        return {
            "peak_force": np.nan,
            "peak_time": np.nan,
            "peak_idx": np.nan,
            "peak_force_pctF0": np.nan,
        }

    peak_idx = int(np.argmax(force))
    peak_force = float(force[peak_idx])
    peak_time = float(time[peak_idx])

    return {
        "peak_force": peak_force,
        "peak_time": peak_time,
        "peak_idx": peak_idx,
        "peak_force_pctF0": np.nan if np.isclose(F0, 0.0) else float(peak_force / F0 * 100.0),
    }

def compute_twitch_metrics(
    force: np.ndarray,
    time: np.ndarray,
    F0: float,
    force_threshold: float = 0.01,
):
    """
    Metrics for twitch trials:
    - peak force
    - rise time: from first non-zero sample to peak
    - half decay time: from peak to half-peak after peak
    """
    force = np.asarray(force, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()

    peak = peak_metrics(force, time, F0)
    onset_time, onset_force, onset_idx = first_positive_sample(time, force, threshold=force_threshold)

    out = {
        "trial_type": "twitch",
        "onset_time": onset_time,
        "onset_force": onset_force,
        "onset_idx": onset_idx,
        "peak_force": peak["peak_force"],
        "peak_time": peak["peak_time"],
        "peak_idx": peak["peak_idx"],
        "peak_force_pctF0": peak["peak_force_pctF0"],
        "rise_time_twitch": np.nan,
        "half_decay_force": np.nan,
        "half_decay_time": np.nan,
        "half_decay_time_twitch": np.nan,
    }

    if np.isnan(onset_time) or np.isnan(peak["peak_time"]):
        return out

    out["rise_time_twitch"] = float(peak["peak_time"] - onset_time)

    if peak["peak_force"] <= 0:
        return out

    half_peak = 0.5 * peak["peak_force"]
    half_decay_time = first_crossing_time(
        time=time,
        signal=force,
        level=half_peak,
        start_idx=int(peak["peak_idx"]) + 1,
        end_idx=len(force),
        direction="falling",
    )

    out["half_decay_force"] = half_peak
    out["half_decay_time"] = half_decay_time

    if not np.isnan(half_decay_time):
        out["half_decay_time_twitch"] = float(half_decay_time - peak["peak_time"])

    return out

def compute_first_order_tetanus_metrics(
    force: np.ndarray,
    time: np.ndarray,
    F0: float,
):
    """
    First-order-style metrics for non-twitch trials:
    - peak force = last detected local peak used for fusion index
    - tau_rise_1st: time at 63.2% of that peak before the peak
    - tau_decay_1st: time at 36.8% of that peak after the peak
    """
    force = np.asarray(force, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()

    fusion = compute_fusion_index(force, time, F0)

    peak_force = fusion["fusion_peak_force"]
    peak_time = fusion["fusion_peak_time"]
    peak_idx = fusion["fusion_peak_idx"]

    out = {
        "peak_force": peak_force,
        "peak_time": peak_time,
        "peak_idx": peak_idx,
        "peak_force_pctF0": np.nan if np.isclose(F0, 0.0) else float(peak_force / F0 * 100.0),
        "rise_force_1st": np.nan,
        "rise_time_1st": np.nan,
        "tau_rise_1st": np.nan,
        "decay_force_1st": np.nan,
        "decay_time_1st": np.nan,
        "tau_decay_1st": np.nan,
    }

    if np.isnan(peak_idx) or peak_force <= 0:
        return out

    peak_idx = int(peak_idx)

    rise_force = 0.632 * peak_force
    decay_force = 0.368 * peak_force

    rise_time = first_crossing_time(
        time=time,
        signal=force,
        level=rise_force,
        start_idx=1,
        end_idx=peak_idx + 1,
        direction="rising",
    )

    decay_time = first_crossing_time(
        time=time,
        signal=force,
        level=decay_force,
        start_idx=peak_idx + 1,
        end_idx=len(force),
        direction="falling",
    )

    out["rise_force_1st"] = rise_force
    out["rise_time_1st"] = rise_time
    out["tau_rise_1st"] = rise_time

    out["decay_force_1st"] = decay_force
    out["decay_time_1st"] = decay_time
    out["tau_decay_1st"] = decay_time

    return out

def compute_fusion_index(
    force: np.ndarray,
    time: np.ndarray,
    F0: float,
    min_peak_prominence: float | None = None,
    min_peak_distance_s: float | None = None,
):
    """
    Fusion index:
        FI = minimum force between the last two detected peaks / last peak force

    Fallback:
    If fewer than two peaks are found, the contraction is considered fully fused
    and the fusion index is set to 1.0.
    """
    force = np.asarray(force, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()

    if force.size < 3 or time.size != force.size:
        return {
            "fusion_peak_force": np.nan,
            "fusion_peak_time": np.nan,
            "fusion_prev_peak_force": np.nan,
            "fusion_prev_peak_time": np.nan,
            "fusion_min_force": np.nan,
            "fusion_min_time": np.nan,
            "fusion_index_ratio": np.nan,
            "fusion_peak_idx": np.nan,
            "fusion_prev_peak_idx": np.nan,
            "fusion_min_idx": np.nan,
        }

    frange = np.max(force) - np.min(force)

    if min_peak_prominence is None:
        min_peak_prominence = 0.01 * frange if frange > 0 else 0.0

    dt = float(np.mean(np.diff(time)))
    if min_peak_distance_s is None:
        min_peak_distance_s = 0.005

    min_peak_distance_samples = max(1, int(round(min_peak_distance_s / dt)))

    peak_idx, _ = find_peaks(
        force,
        prominence=min_peak_prominence,
        distance=min_peak_distance_samples
    )

    if peak_idx.size < 2:
        last_idx = int(np.argmax(force))
        last_force = float(force[last_idx])
        last_time = float(time[last_idx])
        return {
            "fusion_peak_force": last_force,
            "fusion_peak_time": last_time,
            "fusion_prev_peak_force": np.nan,
            "fusion_prev_peak_time": np.nan,
            "fusion_min_force": last_force,
            "fusion_min_time": last_time,
            "fusion_index_ratio": 1.0,
            "fusion_peak_idx": last_idx,
            "fusion_prev_peak_idx": np.nan,
            "fusion_min_idx": last_idx,
        }

    prev_peak_idx = int(peak_idx[-2])
    last_peak_idx = int(peak_idx[-1])

    seg_force = force[prev_peak_idx:last_peak_idx + 1]
    min_rel_idx = int(np.argmin(seg_force))
    min_idx = prev_peak_idx + min_rel_idx

    last_force = float(force[last_peak_idx])
    fi = np.nan if np.isclose(last_force, 0.0) else float(force[min_idx] / last_force)

    return {
        "fusion_peak_force": float(force[last_peak_idx]),
        "fusion_peak_time": float(time[last_peak_idx]),
        "fusion_prev_peak_force": float(force[prev_peak_idx]),
        "fusion_prev_peak_time": float(time[prev_peak_idx]),
        "fusion_min_force": float(force[min_idx]),
        "fusion_min_time": float(time[min_idx]),
        "fusion_index_ratio": fi,
        "fusion_peak_idx": last_peak_idx,
        "fusion_prev_peak_idx": prev_peak_idx,
        "fusion_min_idx": min_idx,
    }

def compute_nontwitch_metrics(
    force: np.ndarray,
    time: np.ndarray,
    F0: float,
    peak_window_s: float = 0.05,
):
    global_peak = peak_metrics(force, time, F0)
    first_order = compute_first_order_tetanus_metrics(force, time, F0)
    fusion = compute_fusion_index(force, time, F0)

    out = {"trial_type": "nontwitch"}

    # global peak kept separate for peak error
    out["peak_force_global"] = global_peak["peak_force"]
    out["peak_time_global"] = global_peak["peak_time"]
    out["peak_idx_global"] = global_peak["peak_idx"]
    out["peak_force_global_pctF0"] = global_peak["peak_force_pctF0"]

    # first-order / fusion metrics
    out.update(first_order)
    out.update(fusion)
    return out

def compute_isometric_trial_metrics(
    force: np.ndarray,
    time: np.ndarray,
    F0: float,
    trial_type: str,
    peak_window_s: float = 0.05,
    force_threshold: float = 0.001,
):
    """
    trial_type:
    - 'twitch'
    - 'nontwitch'
    """
    if trial_type == "twitch":
        return compute_twitch_metrics(
            force=force,
            time=time,
            F0=F0,
            force_threshold=force_threshold,
        )
    elif trial_type == "nontwitch":
        return compute_nontwitch_metrics(
            force=force,
            time=time,
            F0=F0,
            peak_window_s=peak_window_s,
        )
    else:
        raise ValueError("trial_type must be 'twitch' or 'nontwitch'")

def add_peak_ratio_to_twitch(
    metrics: dict,
    twitch_peak_force_ref: float,
):
    """
    Add peak/twitch ratio to a metrics dictionary.

    Returned values:
    - peak_force
    - twitch_peak_force_ref
    - peak_twitch_ratio
    """
    out = dict(metrics)
    peak_force = out.get("peak_force", np.nan)

    out["twitch_peak_force_ref"] = twitch_peak_force_ref

    if np.isnan(peak_force) or np.isnan(twitch_peak_force_ref) or np.isclose(twitch_peak_force_ref, 0.0):
        out["peak_twitch_ratio"] = np.nan
    else:
        out["peak_twitch_ratio"] = float(peak_force / twitch_peak_force_ref)

    return out

def error_time(sim_val: float, exp_val: float) -> float:
    """
    Signed error in seconds:
        simulated - experimental
    """
    if np.isnan(sim_val) or np.isnan(exp_val):
        return np.nan
    return float(sim_val - exp_val)

def error_force_pctF0(sim_force: float, exp_force: float, F0: float) -> float:
    """
    Signed force error expressed as %F0:
        (simulated - experimental) / F0 * 100
    """
    if np.isnan(sim_force) or np.isnan(exp_force) or np.isclose(F0, 0.0):
        return np.nan
    return float((sim_force - exp_force) / F0 * 100.0)

def error_scalar(sim_val: float, exp_val: float) -> float:
    """
    Signed error for dimensionless scalar metrics:
        simulated - experimental
    """
    if np.isnan(sim_val) or np.isnan(exp_val):
        return np.nan
    return float(sim_val - exp_val)

def compute_isometric_metric_errors(
    metrics_exp: dict,
    metrics_sim: dict,
    F0: float,
):
    out = {}

    # Peak error: use global peak if available, otherwise fallback
    exp_peak_for_error = metrics_exp.get("peak_force_global", metrics_exp.get("peak_force", np.nan))
    sim_peak_for_error = metrics_sim.get("peak_force_global", metrics_sim.get("peak_force", np.nan))

    out["peak_force_err_pctF0"] = error_force_pctF0(
        sim_peak_for_error,
        exp_peak_for_error,
        F0
    )

    out["peak_twitch_ratio_err"] = error_scalar(
        metrics_sim.get("peak_twitch_ratio", np.nan),
        metrics_exp.get("peak_twitch_ratio", np.nan),
    )

    out["rise_time_twitch_err_s"] = error_time(
        metrics_sim.get("rise_time_twitch", np.nan),
        metrics_exp.get("rise_time_twitch", np.nan),
    )

    out["half_decay_time_twitch_err_s"] = error_time(
        metrics_sim.get("half_decay_time_twitch", np.nan),
        metrics_exp.get("half_decay_time_twitch", np.nan),
    )

    out["tau_rise_1st_err_s"] = error_time(
        metrics_sim.get("tau_rise_1st", np.nan),
        metrics_exp.get("tau_rise_1st", np.nan),
    )

    out["tau_decay_1st_err_s"] = error_time(
        metrics_sim.get("tau_decay_1st", np.nan),
        metrics_exp.get("tau_decay_1st", np.nan),
    )

    out["fusion_index_ratio_err"] = error_scalar(
        metrics_sim.get("fusion_index_ratio", np.nan),
        metrics_exp.get("fusion_index_ratio", np.nan),
    )

    return out

def compute_full_isometric_evaluation(
    exp: np.ndarray,
    sim: np.ndarray,
    time: np.ndarray,
    F0: float,
    trial_type: str,
    twitch_peak_exp: float,
    twitch_peak_sim: float,
    peak_window_s: float = 0.05,
    force_threshold: float = 0.001,
):
    """
    Full evaluation for an isometric trial:
    - mAE, MAE, std (%F0)
    - R²
    - trial-specific metrics
    - peak/twitch ratio
    - metric errors
    """
    out = {}

    mae, maxae, stde = pct_errors(exp, sim, F0)
    out["mae"] = mae
    out["maxae"] = maxae
    out["stde"] = stde
    out["R2"] = compute_r2(exp, sim)

    metrics_exp = compute_isometric_trial_metrics(
        force=exp,
        time=time,
        F0=F0,
        trial_type=trial_type,
        peak_window_s=peak_window_s,
        force_threshold=force_threshold,
    )
    metrics_sim = compute_isometric_trial_metrics(
        force=sim,
        time=time,
        F0=F0,
        trial_type=trial_type,
        peak_window_s=peak_window_s,
        force_threshold=force_threshold,
    )

    metrics_exp = add_peak_ratio_to_twitch(metrics_exp, twitch_peak_exp)
    metrics_sim = add_peak_ratio_to_twitch(metrics_sim, twitch_peak_sim)

    metric_errors = compute_isometric_metric_errors(metrics_exp, metrics_sim, F0)

    out["metrics_exp"] = metrics_exp
    out["metrics_sim"] = metrics_sim
    out["metric_errors"] = metric_errors

    return out

def plot_twitch_metric_points(
    time: np.ndarray,
    force_exp: np.ndarray,
    force_sim: np.ndarray,
    metrics_exp: dict,
    metrics_sim: dict,
    title: str = "",
    figsize=(7, 4),
):
    """
    Plot twitch traces with onset, peak, and half-decay points.
    """
    plt.figure(figsize=figsize)
    plt.plot(time, force_exp, 'k', label='Experimental')
    plt.plot(time, force_sim, 'r', label='Simulated')

    if not np.isnan(metrics_exp.get("onset_time", np.nan)):
        plt.plot(metrics_exp["onset_time"], metrics_exp["onset_force"], 'ko', label='Exp onset')
    if not np.isnan(metrics_exp.get("peak_time", np.nan)):
        plt.plot(metrics_exp["peak_time"], metrics_exp["peak_force"], 'kD', label='Exp peak')
    if not np.isnan(metrics_exp.get("half_decay_time", np.nan)):
        plt.plot(metrics_exp["half_decay_time"], metrics_exp["half_decay_force"], 'kP', label='Exp half decay')

    if not np.isnan(metrics_sim.get("onset_time", np.nan)):
        plt.plot(metrics_sim["onset_time"], metrics_sim["onset_force"], 'ro', label='Sim onset')
    if not np.isnan(metrics_sim.get("peak_time", np.nan)):
        plt.plot(metrics_sim["peak_time"], metrics_sim["peak_force"], 'rD', label='Sim peak')
    if not np.isnan(metrics_sim.get("half_decay_time", np.nan)):
        plt.plot(metrics_sim["half_decay_time"], metrics_sim["half_decay_force"], 'rP', label='Sim half decay')

    plt.title(title, weight='bold')
    plt.xlabel('Time [s]', weight='bold')
    plt.ylabel('Force', weight='bold')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

def plot_nontwitch_metric_points(
    time: np.ndarray,
    force_exp: np.ndarray,
    force_sim: np.ndarray,
    metrics_exp: dict,
    metrics_sim: dict,
    title: str = "",
    figsize=(7, 4),
):
    """
    Plot non-twitch traces with:
    - peak
    - first-order points
    - fusion previous peak
    - fusion minimum
    - fusion last peak
    """
    plt.figure(figsize=figsize)
    plt.plot(time, force_exp, 'k', label='Experimental')
    plt.plot(time, force_sim, 'r', label='Simulated')

    # Experimental
    if not np.isnan(metrics_exp.get("peak_time", np.nan)):
        plt.plot(metrics_exp["peak_time"], metrics_exp["peak_force"], 'kD', label='Exp peak')
    if not np.isnan(metrics_exp.get("rise_time_1st", np.nan)):
        plt.plot(metrics_exp["rise_time_1st"], metrics_exp["rise_force_1st"], 'k^', label='Exp 63.2% peak')
    if not np.isnan(metrics_exp.get("decay_time_1st", np.nan)):
        plt.plot(metrics_exp["decay_time_1st"], metrics_exp["decay_force_1st"], 'kP', label='Exp 36.8% peak')

    if not np.isnan(metrics_exp.get("fusion_prev_peak_time", np.nan)):
        plt.plot(metrics_exp["fusion_prev_peak_time"], metrics_exp["fusion_prev_peak_force"], 'kv', label='Exp prev peak')
    if not np.isnan(metrics_exp.get("fusion_min_time", np.nan)):
        plt.plot(metrics_exp["fusion_min_time"], metrics_exp["fusion_min_force"], 'ks', label='Exp fusion min')
    if not np.isnan(metrics_exp.get("fusion_peak_time", np.nan)):
        plt.plot(metrics_exp["fusion_peak_time"], metrics_exp["fusion_peak_force"], 'ko', fillstyle='none', label='Exp last peak')

    # Simulated
    if not np.isnan(metrics_sim.get("peak_time", np.nan)):
        plt.plot(metrics_sim["peak_time"], metrics_sim["peak_force"], 'rD', label='Sim peak')
    if not np.isnan(metrics_sim.get("rise_time_1st", np.nan)):
        plt.plot(metrics_sim["rise_time_1st"], metrics_sim["rise_force_1st"], 'r^', label='Sim 63.2% peak')
    if not np.isnan(metrics_sim.get("decay_time_1st", np.nan)):
        plt.plot(metrics_sim["decay_time_1st"], metrics_sim["decay_force_1st"], 'rP', label='Sim 36.8% peak')

    if not np.isnan(metrics_sim.get("fusion_prev_peak_time", np.nan)):
        plt.plot(metrics_sim["fusion_prev_peak_time"], metrics_sim["fusion_prev_peak_force"], 'rv', label='Sim prev peak')
    if not np.isnan(metrics_sim.get("fusion_min_time", np.nan)):
        plt.plot(metrics_sim["fusion_min_time"], metrics_sim["fusion_min_force"], 'rs', label='Sim fusion min')
    if not np.isnan(metrics_sim.get("fusion_peak_time", np.nan)):
        plt.plot(metrics_sim["fusion_peak_time"], metrics_sim["fusion_peak_force"], 'ro', fillstyle='none', label='Sim last peak')

    plt.title(title, weight='bold')
    plt.xlabel('Time [s]', weight='bold')
    plt.ylabel('Force', weight='bold')
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()

def print_twitch_metric_summary(label: str, metrics_exp: dict, metrics_sim: dict, metric_errors: dict, align: int = 24):
    print(
        f"{label:>{align}}   "
        f"peak_exp = {metrics_exp.get('peak_force_pctF0', np.nan):.2f}%F0   "
        f"peak_sim = {metrics_sim.get('peak_force_pctF0', np.nan):.2f}%F0   "
        f"peak/twitch_exp = {metrics_exp.get('peak_twitch_ratio', np.nan):.4f}   "
        f"peak/twitch_sim = {metrics_sim.get('peak_twitch_ratio', np.nan):.4f}"
    )
    print(
        f"{'':>{align}}   "
        f"peak err = {metric_errors['peak_force_err_pctF0']:.2f}%F0   "
        f"peak/twitch err = {metric_errors['peak_twitch_ratio_err']:.4f}   "
        f"rise err = {metric_errors['rise_time_twitch_err_s']:.4f} s   "
        f"half-decay err = {metric_errors['half_decay_time_twitch_err_s']:.4f} s"
    )

def print_nontwitch_metric_summary(label: str, metrics_exp: dict, metrics_sim: dict, metric_errors: dict, align: int = 24):
    print(
        f"{label:>{align}}   "
        f"peak_exp = {metrics_exp.get('peak_force_pctF0', np.nan):.2f}%F0   "
        f"peak_sim = {metrics_sim.get('peak_force_pctF0', np.nan):.2f}%F0   "
        f"peak/twitch_exp = {metrics_exp.get('peak_twitch_ratio', np.nan):.4f}   "
        f"peak/twitch_sim = {metrics_sim.get('peak_twitch_ratio', np.nan):.4f}"
    )
    print(
        f"{'':>{align}}   "
        f"prev_peak_exp = {metrics_exp.get('fusion_prev_peak_force', np.nan):.4f}   "
        f"min_exp = {metrics_exp.get('fusion_min_force', np.nan):.4f}   "
        f"last_peak_exp = {metrics_exp.get('fusion_peak_force', np.nan):.4f}   "
        f"FI_exp = {metrics_exp.get('fusion_index_ratio', np.nan):.4f}"
    )
    print(
        f"{'':>{align}}   "
        f"prev_peak_sim = {metrics_sim.get('fusion_prev_peak_force', np.nan):.4f}   "
        f"min_sim = {metrics_sim.get('fusion_min_force', np.nan):.4f}   "
        f"last_peak_sim = {metrics_sim.get('fusion_peak_force', np.nan):.4f}   "
        f"FI_sim = {metrics_sim.get('fusion_index_ratio', np.nan):.4f}"
    )
    print(
        f"{'':>{align}}   "
        f"peak err = {metric_errors['peak_force_err_pctF0']:.2f}%F0   "
        f"peak/twitch err = {metric_errors['peak_twitch_ratio_err']:.4f}   "
        f"τrise err = {metric_errors['tau_rise_1st_err_s']:.4f} s   "
        f"τdecay err = {metric_errors['tau_decay_1st_err_s']:.4f} s   "
        f"FI err = {metric_errors['fusion_index_ratio_err']:.4f}"
    )

#############################################################################
"PLOT MU_model benchmark results and COMPUTE ERRORS & STATISTICS"
"Files were kept separated to avoid confusion in error and statistics computation"
#############################################################################

benchmark = 'fast_M_dyn' # specify benchmark among [MU, slow_M_max, slow_M_sub, slow_M_len, fast_M_iso, fast_M_dyn]
base_path = Path('..') / 'Results_benchmarks'
dt = 1e-4

if benchmark == 'slow_M_max':  # Krylow & Sandercock 1997 experiments

    sim_path = base_path / 'slow_M' / 'sim'
    exp_path = base_path / 'slow_M' / 'exp'

    exp1 = np.load(exp_path / 'max_0.npy')
    exp2 = np.load(exp_path / 'max_1.npy')
    exp3 = np.load(exp_path / 'max_2.npy')
    exp4 = np.load(exp_path / 'max_3.npy')
    exp5 = np.load(exp_path / 'max_4.npy')
    exp6 = np.load(exp_path / 'max_5.npy')

    sim1 = np.load(sim_path / 'max_0.npy')
    sim2 = np.load(sim_path / 'max_1.npy')
    sim3 = np.load(sim_path / 'max_2.npy')
    sim4 = np.load(sim_path / 'max_3.npy')
    sim5 = np.load(sim_path / 'max_4.npy')
    sim6 = np.load(sim_path / 'max_5.npy')

    t_end = 2
    time_dt = np.arange(0, t_end, dt)
    MVC = 1.32

    fig = plt.figure(figsize=(7, 8))

    plt.subplot(6, 1, 1)
    plt.plot(time_dt, exp1, 'k')
    plt.plot(time_dt, sim1, 'r')
    plt.title(u"\u00B1 0.05 mm", x=0.1, y=0.97, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.xlim((-0.03, 2.03))
    plt.ylim((0, 2))

    plt.subplot(6, 1, 2)
    plt.plot(time_dt, exp2, 'k')
    plt.plot(time_dt, sim2, 'r')
    plt.title(u"\u00B1 0.10 mm", x=0.1, y=0.97, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.xlim((-0.03, 2.03))
    plt.ylim((0, 2))

    plt.subplot(6, 1, 3)
    plt.plot(time_dt, exp3, 'k')
    plt.plot(time_dt, sim3, 'r')
    plt.title(u"\u00B1 0.25 mm", x=0.1, y=0.97, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.xlim((-0.03, 2.03))
    plt.ylim((0, 2))

    plt.subplot(6, 1, 4)
    plt.plot(time_dt, exp4, 'k')
    plt.plot(time_dt, sim4, 'r')
    plt.title(u"\u00B1 0.50 mm", x=0.1, y=0.97, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.xlim((-0.03, 2.03))
    plt.ylim((0, 2))

    plt.subplot(6, 1, 5)
    plt.plot(time_dt, exp5, 'k')
    plt.plot(time_dt, sim5, 'r')
    plt.title(u"\u00B1 1.00 mm", x=0.1, y=0.97, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.xlim((-0.03, 2.03))
    plt.ylim((0, 2))

    plt.subplot(6, 1, 6)
    plt.plot(time_dt, exp6, 'k')
    plt.plot(time_dt, sim6, 'r')
    plt.title(u"\u00B1 2.00 mm", x=0.1, y=0.97, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.xlim((-0.03, 2.03))
    plt.ylim((0, 2))

    fig.text(0.01, 0.5, 'Rat Soleus (Slow) Force [N]', va='center', rotation='vertical',
             weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Errors and R²
    # -------------------------------------------------------------------------
    exp_all = [exp1, exp2, exp3, exp4, exp5, exp6]
    sim_all = [sim1, sim2, sim3, sim4, sim5, sim6]
    displacements = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]

    mean_err_list = []
    max_err_list = []
    std_err_list = []
    r2_list = []

    print("\n=== Benchmark: slow_M_max ===")
    print(f"MVC = {MVC:.2f} N\n")

    for exp, sim, disp in zip(exp_all, sim_all, displacements):
        mae, maxae, stde = pct_errors(exp, sim, MVC)
        r2 = compute_r2(exp, sim)

        mean_err_list.append(mae)
        max_err_list.append(maxae)
        std_err_list.append(stde)
        r2_list.append(r2)

        print(f"Displacement ±{disp:.2f} mm:")
        print(f"  mAE = {mae:.2f}% F0   (std = {stde:.2f})")
        print(f"  MAE = {maxae:.2f}% F0")
        print(f"  R²  = {r2:.4f}\n")

    mmae = np.mean(mean_err_list)
    mstde = np.std(mean_err_list)
    mr2 = np.nanmean(r2_list)
    sr2 = np.nanstd(r2_list)

    print(f"  Mean mAE = {mmae:.2f}% F0   (std = {mstde:.2f})")
    print(f"  Mean R²  = {mr2:.4f}   (std = {sr2:.4f})\n")

    summarize_trials(
        mean_err_list,
        max_err_list,
        label="— slow_M_max summary (all displacements) —",
        unit="% F0"
    )
    summarize_metric_list(r2_list, "— slow_M_max summary: R² (all displacements) —", "")

    # -------------------------------------------------------------------------
    # Error plot (unchanged)
    # -------------------------------------------------------------------------
    x = np.arange(1, len(displacements) + 1)

    mean_err_arr = np.array(mean_err_list)
    max_err_arr = np.array(max_err_list)
    std_err_arr = np.array(std_err_list)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(x, mean_err_arr, '-o', color='k', label='mAE ± SD')
    ax.fill_between(
        x,
        mean_err_arr - std_err_arr,
        mean_err_arr + std_err_arr,
        color='k',
        alpha=0.2
    )

    ax.plot(x, max_err_arr, '--*', color='k', label='MAE')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.2f}' for d in displacements])
    ax.set_xlabel('Max. length variation amplitude [mm]', fontweight='bold')
    ax.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.tight_layout()
    plt.show()


elif benchmark == 'slow_M_sub':  # Perreault 2003 experiments

    sim_path = base_path / 'slow_M' / 'sim'
    exp_path = base_path / 'slow_M' / 'exp'

    # -------------------------------------------------------------------------
    # Load isometric trials
    # -------------------------------------------------------------------------
    sim_iso_c_10 = np.load(sim_path / 'sub_iso_c_10.npy')
    sim_iso_c_20 = np.load(sim_path / 'sub_iso_c_20.npy')
    sim_iso_c_30 = np.load(sim_path / 'sub_iso_c_30.npy')
    sim_iso_v_10 = np.load(sim_path / 'sub_iso_v_10.npy')
    sim_iso_v_20 = np.load(sim_path / 'sub_iso_v_20.npy')
    sim_iso_v_30 = np.load(sim_path / 'sub_iso_v_30.npy')

    exp_iso_c_10 = np.load(exp_path / 'sub_iso_c_10.npy')
    exp_iso_c_20 = np.load(exp_path / 'sub_iso_c_20.npy')
    exp_iso_c_30 = np.load(exp_path / 'sub_iso_c_30.npy')
    exp_iso_v_10 = np.load(exp_path / 'sub_iso_v_10.npy')
    exp_iso_v_20 = np.load(exp_path / 'sub_iso_v_20.npy')
    exp_iso_v_30 = np.load(exp_path / 'sub_iso_v_30.npy')
    
    t_end = 2
    time_dt = np.arange(0, t_end, dt)
    MVC = 26.13


    # Legend handles
    legend_handles = [
        Line2D([0], [0], color='k', lw=2, label='Experimental'),
        Line2D([0], [0], color='r', lw=2, label='Simulated (yielding)'),
        Line2D([0], [0], color='r', lw=2, label='Simulated (no yielding)', linestyle='dashed'),
    ]

    # Create empty figure
    fig = plt.figure(figsize=(6, 1.2))  # largo e basso (tipo strip)

    # Add legend
    fig.legend(handles=legend_handles,
               loc='center',
               ncol=3,
               frameon=True,
               fancybox=False,
               edgecolor='black',
               fontsize=12)

    # Remove axes completely
    plt.axis('off')
    plt.show()


    # -------------------------------------------------------------------------
    # ISO plots (unchanged)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 9))

    plt.subplot(3, 2, 1)
    plt.plot(time_dt, exp_iso_c_10, 'k')
    plt.plot(time_dt, sim_iso_c_10, 'r')
    plt.title(u"Constant 10 Hz", x=0.2, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 2)
    plt.plot(time_dt, exp_iso_v_10, 'k')
    plt.plot(time_dt, sim_iso_v_10, 'r')
    plt.title(u"Random 10 Hz", x=0.2, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 3)
    plt.plot(time_dt, exp_iso_c_20, 'k')
    plt.plot(time_dt, sim_iso_c_20, 'r')
    plt.title(u"Constant 20 Hz", x=0.2, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel('Rat Soleus (Slow) Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 4)
    plt.plot(time_dt, exp_iso_v_20, 'k')
    plt.plot(time_dt, sim_iso_v_20, 'r')
    plt.title(u"Random 20 Hz", x=0.2, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 5)
    plt.plot(time_dt, exp_iso_c_30, 'k')
    plt.plot(time_dt, sim_iso_c_30, 'r')
    plt.title(u"Constant 30 Hz", x=0.2, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 6)
    plt.plot(time_dt, exp_iso_v_30, 'k')
    plt.plot(time_dt, sim_iso_v_30, 'r')
    plt.title(u"Random 30 Hz", x=0.2, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 30))
    
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Load dynamic constant trials
    # -------------------------------------------------------------------------
    sim_dyn_c_10_1 = np.load(sim_path / 'sub_dyn_c_10_1.npy')
    sim_dyn_c_20_1 = np.load(sim_path / 'sub_dyn_c_20_1.npy')
    sim_dyn_c_30_1 = np.load(sim_path / 'sub_dyn_c_30_1.npy')
    sim_dyn_c_10_8 = np.load(sim_path / 'sub_dyn_c_10_8.npy')
    sim_dyn_c_20_8 = np.load(sim_path / 'sub_dyn_c_20_8.npy')
    sim_dyn_c_30_8 = np.load(sim_path / 'sub_dyn_c_30_8.npy')

    sim_dyn_c_10_1_noy = np.load(sim_path / 'sub_dyn_c_10_1_noy.npy')
    sim_dyn_c_20_1_noy = np.load(sim_path / 'sub_dyn_c_20_1_noy.npy')
    sim_dyn_c_30_1_noy = np.load(sim_path / 'sub_dyn_c_30_1_noy.npy')
    sim_dyn_c_10_8_noy = np.load(sim_path / 'sub_dyn_c_10_8_noy.npy')
    sim_dyn_c_20_8_noy = np.load(sim_path / 'sub_dyn_c_20_8_noy.npy')
    sim_dyn_c_30_8_noy = np.load(sim_path / 'sub_dyn_c_30_8_noy.npy')

    exp_dyn_c_10_1 = np.load(exp_path / 'sub_dyn_c_10_1.npy')
    exp_dyn_c_20_1 = np.load(exp_path / 'sub_dyn_c_20_1.npy')
    exp_dyn_c_30_1 = np.load(exp_path / 'sub_dyn_c_30_1.npy')
    exp_dyn_c_10_8 = np.load(exp_path / 'sub_dyn_c_10_8.npy')
    exp_dyn_c_20_8 = np.load(exp_path / 'sub_dyn_c_20_8.npy')
    exp_dyn_c_30_8 = np.load(exp_path / 'sub_dyn_c_30_8.npy')

    # -------------------------------------------------------------------------
    # Dynamic constant plots (unchanged)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 9))

    plt.subplot(3, 2, 1)
    plt.plot(time_dt, exp_dyn_c_10_1, 'k')
    plt.plot(time_dt, sim_dyn_c_10_1, 'r')
    plt.plot(time_dt, sim_dyn_c_10_1_noy, 'r--')
    plt.title(u"Constant 10 Hz, \u00B1 1 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 2)
    plt.plot(time_dt, exp_dyn_c_10_8, 'k')
    plt.plot(time_dt, sim_dyn_c_10_8, 'r')
    plt.plot(time_dt, sim_dyn_c_10_8_noy, 'r--')
    plt.title(u"Constant 10 Hz, \u00B1 8 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 3)
    plt.plot(time_dt, exp_dyn_c_20_1, 'k')
    plt.plot(time_dt, sim_dyn_c_20_1, 'r')
    plt.plot(time_dt, sim_dyn_c_20_1_noy, 'r--')
    plt.title(u"Constant 20 Hz, \u00B1 1 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel('Rat Soleus (Slow) Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 4)
    plt.plot(time_dt, exp_dyn_c_20_8, 'k')
    plt.plot(time_dt, sim_dyn_c_20_8, 'r')
    plt.plot(time_dt, sim_dyn_c_20_8_noy, 'r--')
    plt.title(u"Constant 20 Hz, \u00B1 8 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 5)
    plt.plot(time_dt, exp_dyn_c_30_1, 'k')
    plt.plot(time_dt, sim_dyn_c_30_1, 'r')
    plt.plot(time_dt, sim_dyn_c_30_1_noy, 'r--')
    plt.title(u"Constant 30 Hz, \u00B1 1 mm", x=0.21, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 6)
    plt.plot(time_dt, exp_dyn_c_30_8, 'k')
    plt.plot(time_dt, sim_dyn_c_30_8, 'r')
    plt.plot(time_dt, sim_dyn_c_30_8_noy, 'r--')
    plt.title(u"Constant 30 Hz, \u00B1 8 mm", x=0.21, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Load dynamic random trials
    # -------------------------------------------------------------------------
    sim_dyn_v_10_1 = np.load(sim_path / 'sub_dyn_v_10_1.npy')
    sim_dyn_v_20_1 = np.load(sim_path / 'sub_dyn_v_20_1.npy')
    sim_dyn_v_30_1 = np.load(sim_path / 'sub_dyn_v_30_1.npy')
    sim_dyn_v_10_8 = np.load(sim_path / 'sub_dyn_v_10_8.npy')
    sim_dyn_v_20_8 = np.load(sim_path / 'sub_dyn_v_20_8.npy')
    sim_dyn_v_30_8 = np.load(sim_path / 'sub_dyn_v_30_8.npy')

    sim_dyn_v_10_1_noy = np.load(sim_path / 'sub_dyn_v_10_1_noy.npy')
    sim_dyn_v_20_1_noy = np.load(sim_path / 'sub_dyn_v_20_1_noy.npy')
    sim_dyn_v_30_1_noy = np.load(sim_path / 'sub_dyn_v_30_1_noy.npy')
    sim_dyn_v_10_8_noy = np.load(sim_path / 'sub_dyn_v_10_8_noy.npy')
    sim_dyn_v_20_8_noy = np.load(sim_path / 'sub_dyn_v_20_8_noy.npy')
    sim_dyn_v_30_8_noy = np.load(sim_path / 'sub_dyn_v_30_8_noy.npy')

    exp_dyn_v_10_1 = np.load(exp_path / 'sub_dyn_v_10_1.npy')
    exp_dyn_v_20_1 = np.load(exp_path / 'sub_dyn_v_20_1.npy')
    exp_dyn_v_30_1 = np.load(exp_path / 'sub_dyn_v_30_1.npy')
    exp_dyn_v_10_8 = np.load(exp_path / 'sub_dyn_v_10_8.npy')
    exp_dyn_v_20_8 = np.load(exp_path / 'sub_dyn_v_20_8.npy')
    exp_dyn_v_30_8 = np.load(exp_path / 'sub_dyn_v_30_8.npy')

    # -------------------------------------------------------------------------
    # Dynamic random plots 
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 9))

    plt.subplot(3, 2, 1)
    plt.plot(time_dt, exp_dyn_v_10_1, 'k')
    plt.plot(time_dt, sim_dyn_v_10_1, 'r')
    plt.plot(time_dt, sim_dyn_v_10_1_noy, 'r--')
    plt.title(u"Random 10 Hz, \u00B1 1 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 2)
    plt.plot(time_dt, exp_dyn_v_10_8, 'k')
    plt.plot(time_dt, sim_dyn_v_10_8, 'r')
    plt.plot(time_dt, sim_dyn_v_10_8_noy, 'r--')
    plt.title(u"Random 10 Hz, \u00B1 8 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 3)
    plt.plot(time_dt, exp_dyn_v_20_1, 'k')
    plt.plot(time_dt, sim_dyn_v_20_1, 'r')
    plt.plot(time_dt, sim_dyn_v_20_1_noy, 'r--')
    plt.title(u"Random 20 Hz, \u00B1 1 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel('Rat Soleus (Slow) Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 4)
    plt.plot(time_dt, exp_dyn_v_20_8, 'k')
    plt.plot(time_dt, sim_dyn_v_20_8, 'r')
    plt.plot(time_dt, sim_dyn_v_20_8_noy, 'r--')
    plt.title(u"Random 20 Hz, \u00B1 8 mm", x=0.21, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 5)
    plt.plot(time_dt, exp_dyn_v_30_1, 'k')
    plt.plot(time_dt, sim_dyn_v_30_1, 'r')
    plt.plot(time_dt, sim_dyn_v_30_1_noy, 'r--')
    plt.title(u"Random 30 Hz, \u00B1 1 mm", x=0.21, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 6)
    plt.plot(time_dt, exp_dyn_v_30_8, 'k')
    plt.plot(time_dt, sim_dyn_v_30_8, 'r')
    plt.plot(time_dt, sim_dyn_v_30_8_noy, 'r--')
    plt.title(u"Random 30 Hz, \u00B1 8 mm", x=0.21, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Trial definitions
    # -------------------------------------------------------------------------
    print("\n=== Benchmark: slow_M_sub ===")
    print(f"MVC = {MVC:.2f} N\n")

    iso_trials = [
        ("ISO Const 10 Hz", "nontwitch", exp_iso_c_10, sim_iso_c_10, MVC, time_dt),
        ("ISO Rand  10 Hz", "nontwitch", exp_iso_v_10, sim_iso_v_10, MVC, time_dt),
        ("ISO Const 20 Hz", "nontwitch", exp_iso_c_20, sim_iso_c_20, MVC, time_dt),
        ("ISO Rand  20 Hz", "nontwitch", exp_iso_v_20, sim_iso_v_20, MVC, time_dt),
        ("ISO Const 30 Hz", "nontwitch", exp_iso_c_30, sim_iso_c_30, MVC, time_dt),
        ("ISO Rand  30 Hz", "nontwitch", exp_iso_v_30, sim_iso_v_30, MVC, time_dt),
    ]

    dyn_c_trials = [
        ("DYN Const 10 Hz ±1 mm", exp_dyn_c_10_1, sim_dyn_c_10_1),
        ("DYN Const 10 Hz ±8 mm", exp_dyn_c_10_8, sim_dyn_c_10_8),
        ("DYN Const 20 Hz ±1 mm", exp_dyn_c_20_1, sim_dyn_c_20_1),
        ("DYN Const 20 Hz ±8 mm", exp_dyn_c_20_8, sim_dyn_c_20_8),
        ("DYN Const 30 Hz ±1 mm", exp_dyn_c_30_1, sim_dyn_c_30_1),
        ("DYN Const 30 Hz ±8 mm", exp_dyn_c_30_8, sim_dyn_c_30_8),
    ]

    dyn_v_trials = [
        ("DYN Rand 10 Hz ±1 mm", exp_dyn_v_10_1, sim_dyn_v_10_1),
        ("DYN Rand 10 Hz ±8 mm", exp_dyn_v_10_8, sim_dyn_v_10_8),
        ("DYN Rand 20 Hz ±1 mm", exp_dyn_v_20_1, sim_dyn_v_20_1),
        ("DYN Rand 20 Hz ±8 mm", exp_dyn_v_20_8, sim_dyn_v_20_8),
        ("DYN Rand 30 Hz ±1 mm", exp_dyn_v_30_1, sim_dyn_v_30_1),
        ("DYN Rand 30 Hz ±8 mm", exp_dyn_v_30_8, sim_dyn_v_30_8),
    ]

    dyn_c_trials_noy = [
        ("DYN Const 10 Hz ±1 mm (NOY)", exp_dyn_c_10_1, sim_dyn_c_10_1_noy),
        ("DYN Const 10 Hz ±8 mm (NOY)", exp_dyn_c_10_8, sim_dyn_c_10_8_noy),
        ("DYN Const 20 Hz ±1 mm (NOY)", exp_dyn_c_20_1, sim_dyn_c_20_1_noy),
        ("DYN Const 20 Hz ±8 mm (NOY)", exp_dyn_c_20_8, sim_dyn_c_20_8_noy),
        ("DYN Const 30 Hz ±1 mm (NOY)", exp_dyn_c_30_1, sim_dyn_c_30_1_noy),
        ("DYN Const 30 Hz ±8 mm (NOY)", exp_dyn_c_30_8, sim_dyn_c_30_8_noy),
    ]

    dyn_v_trials_noy = [
        ("DYN Rand 10 Hz ±1 mm (NOY)", exp_dyn_v_10_1, sim_dyn_v_10_1_noy),
        ("DYN Rand 10 Hz ±8 mm (NOY)", exp_dyn_v_10_8, sim_dyn_v_10_8_noy),
        ("DYN Rand 20 Hz ±1 mm (NOY)", exp_dyn_v_20_1, sim_dyn_v_20_1_noy),
        ("DYN Rand 20 Hz ±8 mm (NOY)", exp_dyn_v_20_8, sim_dyn_v_20_8_noy),
        ("DYN Rand 30 Hz ±1 mm (NOY)", exp_dyn_v_30_1, sim_dyn_v_30_1_noy),
        ("DYN Rand 30 Hz ±8 mm (NOY)", exp_dyn_v_30_8, sim_dyn_v_30_8_noy),
    ]

    # -------------------------------------------------------------------------
    # Helper for isometric trials
    # No twitch reference available in this dataset -> peak/twitch ratio = NaN
    # -------------------------------------------------------------------------
    def process_iso_trials(
        trials,
        block_title,
        avg_label,
        align=16,
        plot_points=True,
        peak_window_s=0.05
    ):
        mae_list, maxae_list, std_list = [], [], []
        r2_list = []

        peak_err_list = []
        peak_twitch_ratio_err_list = []
        tau_rise_1st_err_list = []
        tau_decay_1st_err_list = []
        fusion_index_ratio_err_list = []

        print(block_title)

        for name, trial_type, exp, sim, F0, time_dt_local in trials:
            eval_out = compute_full_isometric_evaluation(
                exp=exp,
                sim=sim,
                time=time_dt_local,
                F0=F0,
                trial_type=trial_type,
                twitch_peak_exp=np.nan,
                twitch_peak_sim=np.nan,
                peak_window_s=peak_window_s,
                force_threshold=0.0,
            )

            mae = eval_out["mae"]
            maxae = eval_out["maxae"]
            stde = eval_out["stde"]
            r2 = eval_out["R2"]

            metrics_exp = eval_out["metrics_exp"]
            metrics_sim = eval_out["metrics_sim"]
            metric_errors = eval_out["metric_errors"]

            mae_list.append(mae)
            maxae_list.append(maxae)
            std_list.append(stde)
            r2_list.append(r2)

            peak_err_list.append(metric_errors["peak_force_err_pctF0"])
            peak_twitch_ratio_err_list.append(metric_errors["peak_twitch_ratio_err"])
            tau_rise_1st_err_list.append(metric_errors["tau_rise_1st_err_s"])
            tau_decay_1st_err_list.append(metric_errors["tau_decay_1st_err_s"])
            fusion_index_ratio_err_list.append(metric_errors["fusion_index_ratio_err"])

            print(
                f"{name:>{align}}:  mAE = {mae:6.2f}% F0  (std = {stde:6.2f})   "
                f"MAE = {maxae:6.2f}% F0   R² = {r2:.3f}"
            )

            print_nontwitch_metric_summary(
                label="",
                metrics_exp=metrics_exp,
                metrics_sim=metrics_sim,
                metric_errors=metric_errors,
                align=align
            )

            if plot_points:
                plot_nontwitch_metric_points(
                    time=time_dt_local,
                    force_exp=exp,
                    force_sim=sim,
                    metrics_exp=metrics_exp,
                    metrics_sim=metrics_sim,
                    title=name
                )

        print(
            f"{avg_label:>{align}}:  "
            f"mAE = {np.nanmean(mae_list):6.2f}% F0  "
            f"(std = {np.nanmean(std_list):6.2f})   "
            f"MAE = {np.nanmean(maxae_list):6.2f}% F0   "
            f"R² = {np.nanmean(r2_list):.4f}  (std = {np.nanstd(r2_list):.4f})\n"
        )

        return {
            "mae": mae_list,
            "maxae": maxae_list,
            "std": std_list,
            "r2": r2_list,
            "peak_err_pctF0": peak_err_list,
            "peak_twitch_ratio_err": peak_twitch_ratio_err_list,
            "tau_rise_1st_err_s": tau_rise_1st_err_list,
            "tau_decay_1st_err_s": tau_decay_1st_err_list,
            "fusion_index_ratio_err": fusion_index_ratio_err_list,
        }

    # -------------------------------------------------------------------------
    # Helper for dynamic trials
    # -------------------------------------------------------------------------
    def process_dynamic_trials(trials, block_title, avg_label, align=24):
        mae_list, maxae_list, std_list = [], [], []
        r2_list = []

        print(block_title)
        for name, exp, sim in trials:
            mae, maxae, stde = pct_errors(exp, sim, MVC)
            r2 = compute_r2(exp, sim)

            mae_list.append(mae)
            maxae_list.append(maxae)
            std_list.append(stde)
            r2_list.append(r2)

            print(
                f"{name:>{align}}:  mAE = {mae:6.2f}% F0  "
                f"(std = {stde:6.2f})   MAE = {maxae:6.2f}% F0   R² = {r2:.3f}"
            )

        print(
            f"{avg_label:>{align}}:  "
            f"mAE = {np.nanmean(mae_list):6.2f}% F0  "
            f"(std = {np.nanmean(std_list):6.2f})   "
            f"MAE = {np.nanmean(maxae_list):6.2f}% F0   "
            f"R² = {np.nanmean(r2_list):.4f}  (std = {np.nanstd(r2_list):.4f})\n"
        )

        return {
            "mae": mae_list,
            "maxae": maxae_list,
            "std": std_list,
            "r2": r2_list,
        }

    # -------------------------------------------------------------------------
    # Helper for paired statistics: WITH vs WITHOUT yielding
    # -------------------------------------------------------------------------

    def compare_paired_metrics(with_vals, without_vals, label, unit=""):
        x = np.asarray(with_vals, dtype=float)
        y = np.asarray(without_vals, dtype=float)

        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            print(f"{label}: not enough paired data\n")
            return np.nan

        if np.allclose(x, y):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
            except ValueError:
                p = np.nan

        print(f"{label}:")
        print(f"  with yielding    = {np.mean(x):.4f} ± {np.std(x, ddof=0):.4f} {unit}")
        print(f"  without yielding = {np.mean(y):.4f} ± {np.std(y, ddof=0):.4f} {unit}")
        print(f"  Wilcoxon p       = {p:.4f}\n")
        return p

    def run_yielding_statistics(with_res, without_res, title):
        print(f"\n=== {title} ===\n")

        compare_paired_metrics(with_res["mae"], without_res["mae"], "mAE", "%F0")
        compare_paired_metrics(with_res["maxae"], without_res["maxae"], "MAE", "%F0")
        compare_paired_metrics(with_res["r2"], without_res["r2"], r"$R^2$", "")

    # -------------------------------------------------------------------------
    # ISO
    # -------------------------------------------------------------------------
    iso_res = process_iso_trials(
        iso_trials,
        block_title="— ISO trials —",
        avg_label="ISO Average",
        align=16,
        plot_points=True
    )

    summarize_metric_list(iso_res["r2"], "ISO R²", "")
    summarize_metric_list(iso_res["peak_err_pctF0"], "ISO peak force absolute error", "%F0")
    summarize_metric_list(iso_res["peak_twitch_ratio_err"], "ISO peak/twitch ratio absolute error", "")
    summarize_metric_list(iso_res["tau_rise_1st_err_s"], "ISO tau_rise absolute error", "s")
    summarize_metric_list(iso_res["tau_decay_1st_err_s"], "ISO tau_decay absolute error", "s")
    summarize_metric_list(iso_res["fusion_index_ratio_err"], "ISO fusion index absolute error", "")
    print()

    # -------------------------------------------------------------------------
    # Dynamic constant
    # -------------------------------------------------------------------------
    dync_res = process_dynamic_trials(
        dyn_c_trials,
        block_title="— Dynamic (constant) trials —",
        avg_label="DYN Const Average",
        align=24
    )

    # -------------------------------------------------------------------------
    # Dynamic random
    # -------------------------------------------------------------------------
    dynv_res = process_dynamic_trials(
        dyn_v_trials,
        block_title="— Dynamic (random) trials —",
        avg_label="DYN Rand Average",
        align=23
    )

    # -------------------------------------------------------------------------
    # Dynamic constant - no yielding
    # -------------------------------------------------------------------------
    dync_noy_res = process_dynamic_trials(
        dyn_c_trials_noy,
        block_title="— Dynamic (constant) trials — NO YIELDING —",
        avg_label="DYN Const Average (NOY)",
        align=34
    )

    # -------------------------------------------------------------------------
    # Dynamic random - no yielding
    # -------------------------------------------------------------------------
    dynv_noy_res = process_dynamic_trials(
        dyn_v_trials_noy,
        block_title="— Dynamic (random) trials — NO YIELDING —",
        avg_label="DYN Rand Average (NOY)",
        align=33
    )

    # -------------------------------------------------------------------------
    # Paired statistical tests: WITH yielding vs WITHOUT yielding
    # Combine all dynamic trials
    # -------------------------------------------------------------------------
    dyn_all_with_res = {
        "mae": dync_res["mae"] + dynv_res["mae"],
        "maxae": dync_res["maxae"] + dynv_res["maxae"],
        "std": dync_res["std"] + dynv_res["std"],
        "r2": dync_res["r2"] + dynv_res["r2"],
    }

    dyn_all_noy_res = {
        "mae": dync_noy_res["mae"] + dynv_noy_res["mae"],
        "maxae": dync_noy_res["maxae"] + dynv_noy_res["maxae"],
        "std": dync_noy_res["std"] + dynv_noy_res["std"],
        "r2": dync_noy_res["r2"] + dynv_noy_res["r2"],
    }

    run_yielding_statistics(
        dyn_all_with_res,
        dyn_all_noy_res,
        title="YIELDING EFFECT STATISTICS — all dynamic trials combined"
    )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    summarize_trials(
        iso_res["mae"],
        iso_res["maxae"],
        label="— slow_M_sub summary: ISO (all trials) —",
        unit="% F0"
    )
    summarize_metric_list(iso_res["r2"], "— slow_M_sub summary: ISO R² (all trials) —", "")

    summarize_trials(
        list(dync_res["mae"]) + list(dynv_res["mae"]),
        list(dync_res["maxae"]) + list(dynv_res["maxae"]),
        label="— slow_M_sub summary: DYN (all trials) —",
        unit="% F0"
    )
    summarize_metric_list(
        list(dync_res["r2"]) + list(dynv_res["r2"]),
        "— slow_M_sub summary: DYN R² (all trials) —",
        ""
    )

    summarize_trials(
        list(dync_noy_res["mae"]) + list(dynv_noy_res["mae"]),
        list(dync_noy_res["maxae"]) + list(dynv_noy_res["maxae"]),
        label="— slow_M_sub summary: DYN NO YIELDING (all trials) —",
        unit="% F0"
    )
    summarize_metric_list(
        list(dync_noy_res["r2"]) + list(dynv_noy_res["r2"]),
        "— slow_M_sub summary: DYN NO YIELDING R² (all trials) —",
        ""
    )

    # -------------------------------------------------------------------------
    # Error plots (unchanged)
    # -------------------------------------------------------------------------
    def build_err(trials, MVC):
        mean_list, max_list, std_list = [], [], []
        for _, exp, sim in trials:
            mae, maxae, stde = pct_errors(exp, sim, MVC)
            mean_list.append(mae)
            max_list.append(maxae)
            std_list.append(stde)
        return np.array(mean_list), np.array(max_list), np.array(std_list)

    iso_trials_plot = [
        ("Const 10 Hz", exp_iso_c_10, sim_iso_c_10),
        ("Const 20 Hz", exp_iso_c_20, sim_iso_c_20),
        ("Const 30 Hz", exp_iso_c_30, sim_iso_c_30),
        ("Rand 10 Hz",  exp_iso_v_10, sim_iso_v_10),
        ("Rand 20 Hz",  exp_iso_v_20, sim_iso_v_20),
        ("Rand 30 Hz",  exp_iso_v_30, sim_iso_v_30),
    ]
    mean_iso, max_iso, std_iso = build_err(iso_trials_plot, MVC)

    dyn_1mm_trials = [
        ("Const 10 Hz", exp_dyn_c_10_1, sim_dyn_c_10_1),
        ("Const 20 Hz", exp_dyn_c_20_1, sim_dyn_c_20_1),
        ("Const 30 Hz", exp_dyn_c_30_1, sim_dyn_c_30_1),
        ("Rand 10 Hz",  exp_dyn_v_10_1, sim_dyn_v_10_1),
        ("Rand 20 Hz",  exp_dyn_v_20_1, sim_dyn_v_20_1),
        ("Rand 30 Hz",  exp_dyn_v_30_1, sim_dyn_v_30_1),
    ]
    mean_1mm, max_1mm, std_1mm = build_err(dyn_1mm_trials, MVC)

    dyn_8mm_trials = [
        ("Const 10 Hz", exp_dyn_c_10_8, sim_dyn_c_10_8),
        ("Const 20 Hz", exp_dyn_c_20_8, sim_dyn_c_20_8),
        ("Const 30 Hz", exp_dyn_c_30_8, sim_dyn_c_30_8),
        ("Rand 10 Hz",  exp_dyn_v_10_8, sim_dyn_v_10_8),
        ("Rand 20 Hz",  exp_dyn_v_20_8, sim_dyn_v_20_8),
        ("Rand 30 Hz",  exp_dyn_v_30_8, sim_dyn_v_30_8),
    ]
    mean_8mm, max_8mm, std_8mm = build_err(dyn_8mm_trials, MVC)

    x = np.arange(1, 6 + 1)
    x_labels = ["Constant 10 Hz", "Constant 20 Hz", "Constant 30 Hz", "Random 10 Hz", "Random 20 Hz", "Random 30 Hz"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    def plot_panel(ax, x, mean_arr, max_arr, std_arr, title):
        lower = np.maximum(mean_arr - std_arr, 0)
        upper = mean_arr + std_arr

        ax.plot(x, mean_arr, '-o', color='k', label='mAE ± SD')
        ax.fill_between(x, lower, upper, color='k', alpha=0.2)
        ax.plot(x, max_arr, '--*', color='k', label='MAE')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim([0, 45])

    # plot_panel(axes[0], x, mean_iso, max_iso, std_iso, "Isometric")
    plot_panel(axes[0], x, mean_1mm, max_1mm, std_1mm, "Dynamic ±1 mm")
    plot_panel(axes[1], x, mean_8mm, max_8mm, std_8mm, "Dynamic ±8 mm")

    axes[0].set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Stimulation', fontweight='bold')
    axes[0].legend()

    plt.tight_layout()
    plt.show()


elif benchmark == 'slow_M_len':  # Kim et al. 2015 from Perreault 2003 experiments

    sim_path = base_path / 'slow_M' / 'sim'
    exp_path = base_path / 'slow_M' / 'exp'

    exp_twitch_0 = np.load(exp_path / 'len_iso_twitch_0.npy')
    exp_iso_0_10 = np.load(exp_path / 'len_iso_10_0.npy')
    exp_iso_0_20 = np.load(exp_path / 'len_iso_20_0.npy')
    exp_iso_0_40 = np.load(exp_path / 'len_iso_40_0.npy')
    exp_twitch_8 = np.load(exp_path / 'len_iso_twitch_8.npy')
    exp_iso_8_10 = np.load(exp_path / 'len_iso_10_8.npy')
    exp_iso_8_20 = np.load(exp_path / 'len_iso_20_8.npy')
    exp_iso_8_40 = np.load(exp_path / 'len_iso_40_8.npy')
    exp_twitch_16 = np.load(exp_path / 'len_iso_twitch_16.npy')
    exp_iso_16_10 = np.load(exp_path / 'len_iso_10_16.npy')
    exp_iso_16_20 = np.load(exp_path / 'len_iso_20_16.npy')
    exp_iso_16_40 = np.load(exp_path / 'len_iso_40_16.npy')

    sim_twitch_0 = np.load(sim_path / 'len_iso_twitch_0.npy')
    sim_iso_0_10 = np.load(sim_path / 'len_iso_10_0.npy')
    sim_iso_0_20 = np.load(sim_path / 'len_iso_20_0.npy')
    sim_iso_0_40 = np.load(sim_path / 'len_iso_40_0.npy')
    sim_twitch_8 = np.load(sim_path / 'len_iso_twitch_8.npy')
    sim_iso_8_10 = np.load(sim_path / 'len_iso_10_8.npy')
    sim_iso_8_20 = np.load(sim_path / 'len_iso_20_8.npy')
    sim_iso_8_40 = np.load(sim_path / 'len_iso_40_8.npy')
    sim_twitch_16 = np.load(sim_path / 'len_iso_twitch_16.npy')
    sim_iso_16_10 = np.load(sim_path / 'len_iso_10_16.npy')
    sim_iso_16_20 = np.load(sim_path / 'len_iso_20_16.npy')
    sim_iso_16_40 = np.load(sim_path / 'len_iso_40_16.npy')

    t_end = 1.4
    time_dt = np.arange(0, t_end, dt)
    MVC = 30.25

    # -------------------------------------------------------------------------
    # Plots 
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 8))

    plt.subplot(4, 3, 1)
    plt.plot(time_dt, exp_twitch_0, 'k')
    plt.plot(time_dt, sim_twitch_0, 'r')
    plt.title(r"1 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 10))

    plt.subplot(4, 3, 2)
    plt.plot(time_dt, exp_twitch_8, 'k')
    plt.plot(time_dt, sim_twitch_8, 'r')
    plt.title(r"1 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 10))

    plt.subplot(4, 3, 3)
    plt.plot(time_dt, exp_twitch_16, 'k')
    plt.plot(time_dt, sim_twitch_16, 'r')
    plt.title(r"1 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 10))

    plt.subplot(4, 3, 4)
    plt.plot(time_dt, exp_iso_0_10, 'k')
    plt.plot(time_dt, sim_iso_0_10, 'r')
    plt.title(r"10 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 5)
    plt.plot(time_dt, exp_iso_8_10, 'k')
    plt.plot(time_dt, sim_iso_8_10, 'r')
    plt.title(r"10 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 6)
    plt.plot(time_dt, exp_iso_16_10, 'k')
    plt.plot(time_dt, sim_iso_16_10, 'r')
    plt.title(r"10 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 7)
    plt.plot(time_dt, exp_iso_0_20, 'k')
    plt.plot(time_dt, sim_iso_0_20, 'r')
    plt.title(r"20 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 8)
    plt.plot(time_dt, exp_iso_8_20, 'k')
    plt.plot(time_dt, sim_iso_8_20, 'r')
    plt.title(r"20 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 9)
    plt.plot(time_dt, exp_iso_16_20, 'k')
    plt.plot(time_dt, sim_iso_16_20, 'r')
    plt.title(r"20 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm", x=0.4, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 10)
    plt.plot(time_dt, exp_iso_0_40, 'k')
    plt.plot(time_dt, sim_iso_0_40, 'r')
    plt.title(r"40 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm", x=0.4, y=0.99, weight='bold')
    plt.ylim((0, 35))

    plt.subplot(4, 3, 11)
    plt.plot(time_dt, exp_iso_8_40, 'k')
    plt.plot(time_dt, sim_iso_8_40, 'r')
    plt.title(r"40 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.4, y=0.99, weight='bold')
    plt.ylim((0, 35))
    plt.xlabel('Time [s]', weight='bold', fontsize=14)

    plt.subplot(4, 3, 12)
    plt.plot(time_dt, exp_iso_16_40, 'k')
    plt.plot(time_dt, sim_iso_16_40, 'r')
    plt.title(r"40 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm", x=0.4, y=0.99, weight='bold')
    plt.ylim((0, 35))

    fig.text(0.01, 0.5, 'Rat Soleus (Slow) Force [N]', va='center', rotation='vertical',
             weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    print("\n=== Benchmark: slow_M_len (Perreault/Kim) ===")
    print(f"MVC = {MVC:.2f} N\n")

    # -------------------------------------------------------------------------
    # Trial definitions grouped by length condition
    # -------------------------------------------------------------------------
    trials_0 = [
        ("Twitch 1 Hz, 0 mm",  "twitch",    exp_twitch_0,  sim_twitch_0,  MVC, time_dt),
        ("10 Hz,  -0 mm",      "nontwitch", exp_iso_0_10,  sim_iso_0_10,  MVC, time_dt),
        ("20 Hz,  -0 mm",      "nontwitch", exp_iso_0_20,  sim_iso_0_20,  MVC, time_dt),
        ("40 Hz,  -0 mm",      "nontwitch", exp_iso_0_40,  sim_iso_0_40,  MVC, time_dt),
    ]

    trials_8 = [
        ("Twitch 1 Hz, -8 mm", "twitch",    exp_twitch_8,  sim_twitch_8,  MVC, time_dt),
        ("10 Hz,  -8 mm",      "nontwitch", exp_iso_8_10,  sim_iso_8_10,  MVC, time_dt),
        ("20 Hz,  -8 mm",      "nontwitch", exp_iso_8_20,  sim_iso_8_20,  MVC, time_dt),
        ("40 Hz,  -8 mm",      "nontwitch", exp_iso_8_40,  sim_iso_8_40,  MVC, time_dt),
    ]

    trials_16 = [
        ("Twitch 1 Hz, -16 mm", "twitch",    exp_twitch_16, sim_twitch_16, MVC, time_dt),
        ("10 Hz,  -16 mm",      "nontwitch", exp_iso_16_10, sim_iso_16_10, MVC, time_dt),
        ("20 Hz,  -16 mm",      "nontwitch", exp_iso_16_20, sim_iso_16_20, MVC, time_dt),
        ("40 Hz,  -16 mm",      "nontwitch", exp_iso_16_40, sim_iso_16_40, MVC, time_dt),
    ]

    all_trials = trials_0 + trials_8 + trials_16

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------
    def process_len_trials(
        trials,
        block_title,
        avg_label,
        align=18,
        plot_points=True,
        peak_window_s=0.05,
        twitch_ref=None,   # tuple: (twitch_exp, twitch_sim, F0, time)
    ):
        mae_list, maxae_list, std_list = [], [], []
        r2_list = []

        peak_err_list = []
        peak_twitch_ratio_err_list = []

        rise_time_twitch_err_list = []
        half_decay_time_twitch_err_list = []

        tau_rise_1st_err_list = []
        tau_decay_1st_err_list = []
        fusion_index_ratio_err_list = []

        print(block_title)

        # -----------------------------
        # reference twitch peaks
        # -----------------------------
        twitch_peak_exp = np.nan
        twitch_peak_sim = np.nan

        if twitch_ref is not None:
            twitch_exp, twitch_sim, twitch_F0, twitch_time = twitch_ref
            twitch_metrics_exp = compute_isometric_trial_metrics(
                force=twitch_exp, time=twitch_time, F0=twitch_F0, trial_type="twitch"
            )
            twitch_metrics_sim = compute_isometric_trial_metrics(
                force=twitch_sim, time=twitch_time, F0=twitch_F0, trial_type="twitch"
            )
            twitch_peak_exp = twitch_metrics_exp["peak_force"]
            twitch_peak_sim = twitch_metrics_sim["peak_force"]
        else:
            raise ValueError("A twitch reference trial is required for this block.")

        for name, trial_type, exp, sim, F0, time_dt_local in trials:
            eval_out = compute_full_isometric_evaluation(
                exp=exp,
                sim=sim,
                time=time_dt_local,
                F0=F0,
                trial_type=trial_type,
                twitch_peak_exp=twitch_peak_exp,
                twitch_peak_sim=twitch_peak_sim,
                peak_window_s=peak_window_s,
                force_threshold=0.01,
            )

            mae = eval_out["mae"]
            maxae = eval_out["maxae"]
            stde = eval_out["stde"]
            r2 = eval_out["R2"]

            metrics_exp = eval_out["metrics_exp"]
            metrics_sim = eval_out["metrics_sim"]
            metric_errors = eval_out["metric_errors"]

            mae_list.append(mae)
            maxae_list.append(maxae)
            std_list.append(stde)
            r2_list.append(r2)

            peak_err_list.append(metric_errors["peak_force_err_pctF0"])
            peak_twitch_ratio_err_list.append(metric_errors["peak_twitch_ratio_err"])

            rise_time_twitch_err_list.append(metric_errors["rise_time_twitch_err_s"])
            half_decay_time_twitch_err_list.append(metric_errors["half_decay_time_twitch_err_s"])

            tau_rise_1st_err_list.append(metric_errors["tau_rise_1st_err_s"])
            tau_decay_1st_err_list.append(metric_errors["tau_decay_1st_err_s"])
            fusion_index_ratio_err_list.append(metric_errors["fusion_index_ratio_err"])

            print(
                f"{name:>{align}}:  mAE = {mae:6.2f}% F0  (std = {stde:6.2f})   "
                f"MAE = {maxae:6.2f}% F0   R² = {r2:.3f}"
            )

            if trial_type == "twitch":
                print_twitch_metric_summary(
                    label="",
                    metrics_exp=metrics_exp,
                    metrics_sim=metrics_sim,
                    metric_errors=metric_errors,
                    align=align
                )
            else:
                print_nontwitch_metric_summary(
                    label="",
                    metrics_exp=metrics_exp,
                    metrics_sim=metrics_sim,
                    metric_errors=metric_errors,
                    align=align
                )

            if plot_points:
                if trial_type == "twitch":
                    plot_twitch_metric_points(
                        time=time_dt_local,
                        force_exp=exp,
                        force_sim=sim,
                        metrics_exp=metrics_exp,
                        metrics_sim=metrics_sim,
                        title=name
                    )
                else:
                    plot_nontwitch_metric_points(
                        time=time_dt_local,
                        force_exp=exp,
                        force_sim=sim,
                        metrics_exp=metrics_exp,
                        metrics_sim=metrics_sim,
                        title=name
                    )

        print(
            f"{avg_label:>{align}}:  "
            f"mAE = {np.nanmean(mae_list):6.2f}% F0  "
            f"(std = {np.nanmean(std_list):6.2f})   "
            f"MAE = {np.nanmean(maxae_list):6.2f}% F0   "
            f"R² = {np.nanmean(r2_list):.4f}  (std = {np.nanstd(r2_list):.4f})\n"
        )

        return {
            "mae": mae_list,
            "maxae": maxae_list,
            "std": std_list,
            "r2": r2_list,
            "peak_err_pctF0": peak_err_list,
            "peak_twitch_ratio_err": peak_twitch_ratio_err_list,
            "rise_time_twitch_err_s": rise_time_twitch_err_list,
            "half_decay_time_twitch_err_s": half_decay_time_twitch_err_list,
            "tau_rise_1st_err_s": tau_rise_1st_err_list,
            "tau_decay_1st_err_s": tau_decay_1st_err_list,
            "fusion_index_ratio_err": fusion_index_ratio_err_list,
        }

    # -------------------------------------------------------------------------
    # Process each ΔL group with its own twitch reference
    # -------------------------------------------------------------------------
    res_0 = process_len_trials(
        trials_0,
        block_title="— ΔL = 0 mm trials —",
        avg_label="Average ΔL = 0 mm",
        align=18,
        plot_points=True,
        twitch_ref=(exp_twitch_0, sim_twitch_0, MVC, time_dt)
    )

    res_8 = process_len_trials(
        trials_8,
        block_title="— ΔL = -8 mm trials —",
        avg_label="Average ΔL = -8 mm",
        align=18,
        plot_points=True,
        twitch_ref=(exp_twitch_8, sim_twitch_8, MVC, time_dt)
    )

    res_16 = process_len_trials(
        trials_16,
        block_title="— ΔL = -16 mm trials —",
        avg_label="Average ΔL = -16 mm",
        align=18,
        plot_points=True,
        twitch_ref=(exp_twitch_16, sim_twitch_16, MVC, time_dt)
    )

    # -------------------------------------------------------------------------
    # Overall summaries
    # -------------------------------------------------------------------------
    all_res = {
        "mae": res_0["mae"] + res_8["mae"] + res_16["mae"],
        "maxae": res_0["maxae"] + res_8["maxae"] + res_16["maxae"],
        "std": res_0["std"] + res_8["std"] + res_16["std"],
        "r2": res_0["r2"] + res_8["r2"] + res_16["r2"],
        "peak_err_pctF0": res_0["peak_err_pctF0"] + res_8["peak_err_pctF0"] + res_16["peak_err_pctF0"],
        "peak_twitch_ratio_err": res_0["peak_twitch_ratio_err"] + res_8["peak_twitch_ratio_err"] + res_16["peak_twitch_ratio_err"],
        "rise_time_twitch_err_s": res_0["rise_time_twitch_err_s"] + res_8["rise_time_twitch_err_s"] + res_16["rise_time_twitch_err_s"],
        "half_decay_time_twitch_err_s": res_0["half_decay_time_twitch_err_s"] + res_8["half_decay_time_twitch_err_s"] + res_16["half_decay_time_twitch_err_s"],
        "tau_rise_1st_err_s": res_0["tau_rise_1st_err_s"] + res_8["tau_rise_1st_err_s"] + res_16["tau_rise_1st_err_s"],
        "tau_decay_1st_err_s": res_0["tau_decay_1st_err_s"] + res_8["tau_decay_1st_err_s"] + res_16["tau_decay_1st_err_s"],
        "fusion_index_ratio_err": res_0["fusion_index_ratio_err"] + res_8["fusion_index_ratio_err"] + res_16["fusion_index_ratio_err"],
    }

    summarize_trials(
        all_res["mae"],
        all_res["maxae"],
        label="— slow_M_len summary (all trials) —",
        unit="% F0"
    )

    summarize_metric_list(all_res["r2"], "Overall R²", "")
    summarize_metric_list(all_res["peak_err_pctF0"], "Overall peak force absolute error", "%F0")
    summarize_metric_list(all_res["peak_twitch_ratio_err"], "Overall peak/twitch ratio absolute error", "")
    summarize_metric_list(all_res["rise_time_twitch_err_s"], "Overall twitch rise time absolute error", "s")
    summarize_metric_list(all_res["half_decay_time_twitch_err_s"], "Overall twitch half-decay time absolute error", "s")
    summarize_metric_list(all_res["tau_rise_1st_err_s"], "Overall tau_rise absolute error", "s")
    summarize_metric_list(all_res["tau_decay_1st_err_s"], "Overall tau_decay absolute error", "s")
    summarize_metric_list(all_res["fusion_index_ratio_err"], "Overall fusion index absolute error", "")
    print()

    # -------------------------------------------------------------------------
    # Error panels (unchanged)
    # -------------------------------------------------------------------------
    def build_len_err(exp_list, sim_list, MVC):
        mean_list, max_list, std_list = [], [], []
        for exp, sim in zip(exp_list, sim_list):
            mae, maxae, stde = pct_errors(exp, sim, MVC)
            mean_list.append(mae)
            max_list.append(maxae)
            std_list.append(stde)
        return np.array(mean_list), np.array(max_list), np.array(std_list)

    mean_0, max_0, std_0 = build_len_err(
        [exp_twitch_0, exp_iso_0_10, exp_iso_0_20, exp_iso_0_40],
        [sim_twitch_0, sim_iso_0_10, sim_iso_0_20, sim_iso_0_40],
        MVC
    )

    mean_8, max_8, std_8 = build_len_err(
        [exp_twitch_8, exp_iso_8_10, exp_iso_8_20, exp_iso_8_40],
        [sim_twitch_8, sim_iso_8_10, sim_iso_8_20, sim_iso_8_40],
        MVC
    )

    mean_16, max_16, std_16 = build_len_err(
        [exp_twitch_16, exp_iso_16_10, exp_iso_16_20, exp_iso_16_40],
        [sim_twitch_16, sim_iso_16_10, sim_iso_16_20, sim_iso_16_40],
        MVC
    )

    x = np.arange(1, 5)
    x_labels = ["1 Hz", "10 Hz", "20 Hz", "40 Hz"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)

    def plot_panel(ax, x, mean_arr, max_arr, std_arr, title):
        lower = np.maximum(mean_arr - std_arr, 0)
        upper = mean_arr + std_arr

        ax.plot(x, mean_arr, '-o', color='k', label='mAE ± SD')
        ax.fill_between(x, lower, upper, color='k', alpha=0.2)
        ax.plot(x, max_arr, '--*', color='k', label='MAE')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(bottom=0)

    plot_panel(axes[0], x, mean_0,  max_0,  std_0,  r"$\boldsymbol{\Delta}\mathbf{L}$ = 0 mm")
    plot_panel(axes[1], x, mean_8,  max_8,  std_8,  r"$\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm")
    plot_panel(axes[2], x, mean_16, max_16, std_16, r"$\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm")

    axes[0].set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Stimulation frequency', fontweight='bold')
    axes[0].legend()

    plt.tight_layout()
    plt.show()


elif benchmark == 'MU':  # Burke 1974 & Celichowski 1999 experiments

    sim_path = base_path / 'MU' / 'sim'
    exp_path = base_path / 'MU' / 'exp'

    # -------------------------------------------------------------------------
    # Load experimental traces
    # -------------------------------------------------------------------------
    exp_S_twitch = np.load(exp_path / 'slow_twitch.npy')
    exp_S_unfused = np.load(exp_path / 'slow_unfused.npy')
    exp_S_fused = np.load(exp_path / 'slow_fused.npy')

    exp_F_twitch = np.load(exp_path / 'fast_twitch.npy')
    exp_F_unfused = np.load(exp_path / 'fast_unfused.npy')
    exp_F_fused = np.load(exp_path / 'fast_fused.npy')

    exp_F_25 = np.load(exp_path / 'fast_25.npy')
    exp_F_30 = np.load(exp_path / 'fast_30.npy')
    exp_F_35 = np.load(exp_path / 'fast_35.npy')
    exp_F_40 = np.load(exp_path / 'fast_40.npy')
    exp_F_150 = np.load(exp_path / 'fast_150.npy')

    # -------------------------------------------------------------------------
    # Load simulated traces
    # -------------------------------------------------------------------------
    sim_S_twitch = np.load(sim_path / 'slow_twitch.npy')
    sim_S_unfused = np.load(sim_path / 'slow_unfused.npy')
    sim_S_fused = np.load(sim_path / 'slow_fused.npy')

    sim_F_twitch = np.load(sim_path / 'fast_twitch_nosag.npy')  # sag not active for twitch
    sim_F_twitch_nosag = np.load(sim_path / 'fast_twitch_nosag.npy')
    sim_F_unfused = np.load(sim_path / 'fast_unfused.npy')
    sim_F_unfused_nosag = np.load(sim_path / 'fast_unfused_nosag.npy')
    sim_F_fused = np.load(sim_path / 'fast_fused.npy')

    sim_F_25 = np.load(sim_path / 'fast_25.npy')
    sim_F_30 = np.load(sim_path / 'fast_30.npy')
    sim_F_35 = np.load(sim_path / 'fast_35.npy')
    sim_F_40 = np.load(sim_path / 'fast_40.npy')
    sim_F_150 = np.load(sim_path / 'fast_150.npy')

    sim_F_25_nosag = np.load(sim_path / 'fast_25_nosag.npy')
    sim_F_30_nosag = np.load(sim_path / 'fast_30_nosag.npy')
    sim_F_35_nosag = np.load(sim_path / 'fast_35_nosag.npy')
    sim_F_40_nosag = np.load(sim_path / 'fast_40_nosag.npy')

    # -------------------------------------------------------------------------
    # Time vectors and F0
    # -------------------------------------------------------------------------
    t_end_S = 1.8
    t_end_F1 = 1.2
    t_end_F2 = 0.7

    time_dt_S = np.arange(0, t_end_S, dt)
    time_dt_F1 = np.arange(0, t_end_F1, dt)
    time_dt_F2 = np.arange(0, t_end_F2, dt)

    MVC_S = 0.04
    MVC_F1 = 0.40
    MVC_F2 = 0.073

    # -------------------------------------------------------------------------
    # Force traces
    # -------------------------------------------------------------------------
    # Common legend handles
    legend_handles = [
        Line2D([0], [0], color='k', lw=2, label='Experimental'),
        Line2D([0], [0], color='r', lw=2, label='Simulated (sag)'),
        Line2D([0], [0], color='r', lw=2, ls='--', label='Simulated (no sag)')
    ]
    fig, axs = plt.subplots(2, 3, figsize=(12, 5))

    # --- Row 1: Cat LG (S) ---
    axs[0, 0].plot(time_dt_S, exp_S_twitch, 'k')
    axs[0, 0].plot(time_dt_S, sim_S_twitch, 'r')
    axs[0, 0].text(0.8, 0.95, '1 Hz', transform=axs[0, 0].transAxes,
                   ha='left', va='top', weight='bold')
    axs[0, 0].set_ylabel('Cat LG (S) \nMU Force [N]', weight='bold', fontsize=14)
    axs[0, 0].set_ylim((0, MVC_S + 0.01))

    axs[0, 1].plot(time_dt_S, exp_S_unfused, 'k')
    axs[0, 1].plot(time_dt_S, sim_S_unfused, 'r')
    axs[0, 1].text(0.8, 0.95, '12.5 Hz', transform=axs[0, 1].transAxes,
                   ha='left', va='top', weight='bold')
    axs[0, 1].tick_params(axis='y', which='both', labelleft=False)
    axs[0, 1].set_ylim((0, MVC_S + 0.01))

    axs[0, 2].plot(time_dt_S, exp_S_fused, 'k')
    axs[0, 2].plot(time_dt_S, sim_S_fused, 'r')
    axs[0, 2].plot(time_dt_S, sim_S_fused, 'r--')
    axs[0, 2].text(0.8, 0.95, '40 Hz', transform=axs[0, 2].transAxes,
                   ha='left', va='top', weight='bold')
    axs[0, 2].tick_params(axis='y', which='both', labelleft=False)
    axs[0, 2].set_ylim((0, MVC_S + 0.01))

    # --- Row 2: Cat MG (F) ---
    axs[1, 0].plot(time_dt_F1, exp_F_twitch, 'k')
    axs[1, 0].plot(time_dt_F1, sim_F_twitch, 'r')
    axs[1, 0].text(0.8, 0.95, '1 Hz', transform=axs[1, 0].transAxes,
                   ha='left', va='top', weight='bold')
    axs[1, 0].set_ylabel('Cat MG (F) \nMU Force [N]', weight='bold', fontsize=14)
    axs[1, 0].set_ylim((0, MVC_F1))

    axs[1, 1].plot(time_dt_F1, exp_F_unfused, 'k')
    axs[1, 1].plot(time_dt_F1, sim_F_unfused, 'r')
    axs[1, 1].plot(time_dt_F1, sim_F_unfused_nosag, 'r--')
    axs[1, 1].text(0.8, 0.95, '25 Hz', transform=axs[1, 1].transAxes,
                   ha='left', va='top', weight='bold')
    axs[1, 1].set_xlabel('Time [s]', weight='bold', fontsize=14)
    axs[1, 1].tick_params(axis='y', which='both', labelleft=False)
    axs[1, 1].set_ylim((0, MVC_F1))

    axs[1, 2].plot(time_dt_F1, exp_F_fused, 'k')
    axs[1, 2].plot(time_dt_F1, sim_F_fused, 'r')
    axs[1, 2].text(0.8, 0.95, '40 Hz', transform=axs[1, 2].transAxes,
               ha='left', va='top', weight='bold')
    axs[1, 2].tick_params(axis='y', which='both', labelleft=False)
    axs[1, 2].set_ylim((0, MVC_F1))

    fig.legend(handles=legend_handles,
              loc='upper center',
              ncol=3,
              frameon=True,
              fancybox=False,      
              edgecolor='black',  
              fontsize=12,
              bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    fig, axs = plt.subplots(1, 5, figsize=(14, 5))

    axs[0].plot(time_dt_F2, exp_F_25, 'k')
    axs[0].plot(time_dt_F2, sim_F_25, 'r')
    axs[0].plot(time_dt_F2, sim_F_25_nosag, 'r--')
    axs[0].text(0.76, 0.95, '25 Hz', transform=axs[0].transAxes,
            ha='left', va='top', weight='bold')
    axs[0].set_ylim((0, MVC_F2 + 0.006))
    axs[0].set_ylabel('Rat MG (F) \nMU Force [N]', weight='bold', fontsize=14)

    axs[1].plot(time_dt_F2, exp_F_30, 'k')
    axs[1].plot(time_dt_F2, sim_F_30, 'r')
    axs[1].plot(time_dt_F2, sim_F_30_nosag, 'r--')
    axs[1].text(0.76, 0.95, '30 Hz', transform=axs[1].transAxes,
            ha='left', va='top', weight='bold')
    axs[1].tick_params(axis='y', which='both', labelleft=False)
    axs[1].set_ylim((0, MVC_F2 + 0.006))

    axs[2].plot(time_dt_F2, exp_F_35, 'k')
    axs[2].plot(time_dt_F2, sim_F_35, 'r')
    axs[2].plot(time_dt_F2, sim_F_35_nosag, 'r--')
    axs[2].text(0.76, 0.95, '35 Hz', transform=axs[2].transAxes,
            ha='left', va='top', weight='bold')
    axs[2].tick_params(axis='y', which='both', labelleft=False)
    axs[2].set_ylim((0, MVC_F2 + 0.006))

    axs[3].plot(time_dt_F2, exp_F_40, 'k')
    axs[3].plot(time_dt_F2, sim_F_40, 'r')
    axs[3].plot(time_dt_F2, sim_F_40_nosag, 'r--')
    axs[3].text(0.76, 0.95, '40 Hz', transform=axs[3].transAxes,
            ha='left', va='top', weight='bold')
    axs[3].tick_params(axis='y', which='both', labelleft=False)
    axs[3].set_ylim((0, MVC_F2 + 0.006))

    axs[4].plot(time_dt_F2, exp_F_150, 'k')
    axs[4].plot(time_dt_F2, sim_F_150, 'r')
    axs[4].text(0.76, 0.95, '150 Hz', transform=axs[4].transAxes,
            ha='left', va='top', weight='bold')
    axs[4].tick_params(axis='y', which='both', labelleft=False)
    axs[4].set_ylim((0, MVC_F2 + 0.006))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    print("\n=== Benchmark: MU (Burke 1974 & Celichowski 1999) ===")
    print(f"MVC_S (slow) = {MVC_S:.4f} N   |   MVC_F1 (fast) = {MVC_F1:.4f} N  |  MVC_F2 (fast) = {MVC_F2:.4f}\n")

    # -------------------------------------------------------------------------
    # Trial definitions
    # -------------------------------------------------------------------------
    slow_trials = [
        ("Slow Twitch   (1 Hz)",   "twitch",    exp_S_twitch,   sim_S_twitch,   MVC_S,  time_dt_S),
        ("Slow Unfused (12.5 Hz)", "nontwitch", exp_S_unfused,  sim_S_unfused,  MVC_S,  time_dt_S),
        ("Slow Fused   (40 Hz)",   "nontwitch", exp_S_fused,    sim_S_fused,    MVC_S,  time_dt_S),
    ]

    fast_cat_trials = [
        ("Fast Twitch  (1 Hz)",    "twitch",    exp_F_twitch,   sim_F_twitch,   MVC_F1, time_dt_F1),
        ("Fast Unfused (25 Hz)",   "nontwitch", exp_F_unfused,  sim_F_unfused,  MVC_F1, time_dt_F1),
        ("Fast Fused   (40 Hz)",   "nontwitch", exp_F_fused,    sim_F_fused,    MVC_F1, time_dt_F1),
    ]

    fast_cat_trials_nosag = [
        ("Fast Twitch  (1 Hz, no sag)",   "twitch",    exp_F_twitch,  sim_F_twitch_nosag,   MVC_F1, time_dt_F1),
        ("Fast Unfused (25 Hz, no sag)",  "nontwitch", exp_F_unfused, sim_F_unfused_nosag,  MVC_F1, time_dt_F1),
    ]

    fast_rat_trials = [
        ("Fast 25 Hz  (rat MG)",  "nontwitch", exp_F_25,  sim_F_25,  MVC_F2, time_dt_F2),
        ("Fast 30 Hz  (rat MG)",  "nontwitch", exp_F_30,  sim_F_30,  MVC_F2, time_dt_F2),
        ("Fast 35 Hz  (rat MG)",  "nontwitch", exp_F_35,  sim_F_35,  MVC_F2, time_dt_F2),
        ("Fast 40 Hz  (rat MG)",  "nontwitch", exp_F_40,  sim_F_40,  MVC_F2, time_dt_F2),
        ("Fast 150 Hz (rat MG)",  "nontwitch", exp_F_150, sim_F_150, MVC_F2, time_dt_F2),
    ]

    fast_rat_trials_nosag = [
        ("Fast 25 Hz  (rat MG, no sag)", "nontwitch", exp_F_25, sim_F_25_nosag, MVC_F2, time_dt_F2),
        ("Fast 30 Hz  (rat MG, no sag)", "nontwitch", exp_F_30, sim_F_30_nosag, MVC_F2, time_dt_F2),
        ("Fast 35 Hz  (rat MG, no sag)", "nontwitch", exp_F_35, sim_F_35_nosag, MVC_F2, time_dt_F2),
        ("Fast 40 Hz  (rat MG, no sag)", "nontwitch", exp_F_40, sim_F_40_nosag, MVC_F2, time_dt_F2),
    ]

    # -------------------------------------------------------------------------
    # Helper to process one MU group
    # -------------------------------------------------------------------------
    def process_mu_trials(
        trials,
        block_title,
        avg_label,
        align=22,
        plot_points=True,
        peak_window_s=0.05,
        twitch_ref=None,
    ):
        mae_list, maxae_list, std_list = [], [], []
        r2_list = []

        peak_err_list = []
        peak_twitch_ratio_err_list = []

        rise_time_twitch_err_list = []
        half_decay_time_twitch_err_list = []

        tau_rise_1st_err_list = []
        tau_decay_1st_err_list = []
        fusion_index_ratio_err_list = []

        print(block_title)

        twitch_peak_exp = np.nan
        twitch_peak_sim = np.nan

        if twitch_ref is not None:
            twitch_exp, twitch_sim, twitch_F0, twitch_time = twitch_ref
            twitch_metrics_exp = compute_isometric_trial_metrics(
                force=twitch_exp, time=twitch_time, F0=twitch_F0, trial_type="twitch"
            )
            twitch_metrics_sim = compute_isometric_trial_metrics(
                force=twitch_sim, time=twitch_time, F0=twitch_F0, trial_type="twitch"
            )
            twitch_peak_exp = twitch_metrics_exp["peak_force"]
            twitch_peak_sim = twitch_metrics_sim["peak_force"]
        else:
            for trial in trials:
                if trial[1] == "twitch":
                    _, _, twitch_exp, twitch_sim, twitch_F0, twitch_time = trial
                    twitch_metrics_exp = compute_isometric_trial_metrics(
                        force=twitch_exp, time=twitch_time, F0=twitch_F0, trial_type="twitch"
                    )
                    twitch_metrics_sim = compute_isometric_trial_metrics(
                        force=twitch_sim, time=twitch_time, F0=twitch_F0, trial_type="twitch"
                    )
                    twitch_peak_exp = twitch_metrics_exp["peak_force"]
                    twitch_peak_sim = twitch_metrics_sim["peak_force"]
                    break

        for name, trial_type, exp, sim, F0, time_dt_local in trials:
            eval_out = compute_full_isometric_evaluation(
                exp=exp,
                sim=sim,
                time=time_dt_local,
                F0=F0,
                trial_type=trial_type,
                twitch_peak_exp=twitch_peak_exp,
                twitch_peak_sim=twitch_peak_sim,
                peak_window_s=peak_window_s,
                force_threshold=0.0,
            )

            mae = eval_out["mae"]
            maxae = eval_out["maxae"]
            stde = eval_out["stde"]
            r2 = eval_out["R2"]

            metrics_exp = eval_out["metrics_exp"]
            metrics_sim = eval_out["metrics_sim"]
            metric_errors = eval_out["metric_errors"]

            mae_list.append(mae)
            maxae_list.append(maxae)
            std_list.append(stde)
            r2_list.append(r2)

            peak_err_list.append(metric_errors["peak_force_err_pctF0"])
            peak_twitch_ratio_err_list.append(metric_errors["peak_twitch_ratio_err"])

            rise_time_twitch_err_list.append(metric_errors["rise_time_twitch_err_s"])
            half_decay_time_twitch_err_list.append(metric_errors["half_decay_time_twitch_err_s"])

            tau_rise_1st_err_list.append(metric_errors["tau_rise_1st_err_s"])
            tau_decay_1st_err_list.append(metric_errors["tau_decay_1st_err_s"])
            fusion_index_ratio_err_list.append(metric_errors["fusion_index_ratio_err"])

            print(
                f"{name:>{align}}:  mAE = {mae:6.2f}% F0  (std = {stde:6.2f})   "
                f"MAE = {maxae:6.2f}% F0   R² = {r2:.3f}"
            )

            if trial_type == "twitch":
                print_twitch_metric_summary(
                    label="",
                    metrics_exp=metrics_exp,
                    metrics_sim=metrics_sim,
                    metric_errors=metric_errors,
                    align=align
                )
            else:
                print_nontwitch_metric_summary(
                    label="",
                    metrics_exp=metrics_exp,
                    metrics_sim=metrics_sim,
                    metric_errors=metric_errors,
                    align=align
                )

            if plot_points:
                if trial_type == "twitch":
                    plot_twitch_metric_points(
                        time=time_dt_local,
                        force_exp=exp,
                        force_sim=sim,
                        metrics_exp=metrics_exp,
                        metrics_sim=metrics_sim,
                        title=name
                    )
                else:
                    plot_nontwitch_metric_points(
                        time=time_dt_local,
                        force_exp=exp,
                        force_sim=sim,
                        metrics_exp=metrics_exp,
                        metrics_sim=metrics_sim,
                        title=name
                    )

        print(
            f"{avg_label:>{align}}:  "
            f"mAE = {np.nanmean(mae_list):6.2f}% F0  "
            f"(std = {np.nanmean(std_list):6.2f})   "
            f"MAE = {np.nanmean(maxae_list):6.2f}% F0\n"
        )

        return {
            "mae": mae_list,
            "maxae": maxae_list,
            "std": std_list,
            "r2": r2_list,
            "peak_err_pctF0": peak_err_list,
            "peak_twitch_ratio_err": peak_twitch_ratio_err_list,
            "rise_time_twitch_err_s": rise_time_twitch_err_list,
            "half_decay_time_twitch_err_s": half_decay_time_twitch_err_list,
            "tau_rise_1st_err_s": tau_rise_1st_err_list,
            "tau_decay_1st_err_s": tau_decay_1st_err_list,
            "fusion_index_ratio_err": fusion_index_ratio_err_list,
        }

    # -------------------------------------------------------------------------
    # Statistical comparison helper: WITH vs WITHOUT sag (paired Wilcoxon)
    # -------------------------------------------------------------------------
    def compare_paired_metrics(with_vals, without_vals, label, unit=""):
        x = np.asarray(with_vals, dtype=float)
        y = np.asarray(without_vals, dtype=float)

        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            print(f"{label}: not enough paired data\n")
            return np.nan

        if np.allclose(x, y):
            p = 1.0
            stat = 0.0
        else:
            try:
                stat, p = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
            except ValueError:
                stat, p = np.nan, np.nan

        print(f"{label}:")
        print(f"  with sag    = {np.mean(x):.4f} ± {np.std(x, ddof=0):.4f} {unit}")
        print(f"  without sag = {np.mean(y):.4f} ± {np.std(y, ddof=0):.4f} {unit}")
        print(f"  Wilcoxon p  = {p:.4f}\n")
        return p

    def run_sag_statistics(with_res, without_res, title):
        print(f"\n=== {title} ===\n")

        compare_paired_metrics(with_res["mae"], without_res["mae"], "mAE", "%F0")
        compare_paired_metrics(with_res["maxae"], without_res["maxae"], "MAE", "%F0")
        compare_paired_metrics(with_res["r2"], without_res["r2"], r"$R^2$", "")
        compare_paired_metrics(with_res["peak_err_pctF0"], without_res["peak_err_pctF0"], "Peak force error", "%F0")
        compare_paired_metrics(with_res["peak_twitch_ratio_err"], without_res["peak_twitch_ratio_err"], "Peak/twitch ratio error", "")
        compare_paired_metrics(with_res["rise_time_twitch_err_s"], without_res["rise_time_twitch_err_s"], "Twitch rise time error", "s")
        compare_paired_metrics(with_res["half_decay_time_twitch_err_s"], without_res["half_decay_time_twitch_err_s"], "Twitch half-decay time error", "s")
        compare_paired_metrics(with_res["tau_rise_1st_err_s"], without_res["tau_rise_1st_err_s"], "Tetanus tau_rise error", "s")
        compare_paired_metrics(with_res["tau_decay_1st_err_s"], without_res["tau_decay_1st_err_s"], "Tetanus tau_decay error", "s")
        compare_paired_metrics(with_res["fusion_index_ratio_err"], without_res["fusion_index_ratio_err"], "Fusion index error", "")

    # -------------------------------------------------------------------------
    # Slow MU
    # -------------------------------------------------------------------------
    slow_res = process_mu_trials(
        slow_trials,
        block_title="— Slow MU trials —",
        avg_label="Slow Average",
        align=22,
        plot_points=True
    )

    summarize_metric_list(slow_res["r2"], "Slow MU R²", "")
    summarize_metric_list(slow_res["peak_err_pctF0"], "Slow MU peak force absolute error", "%F0")
    summarize_metric_list(slow_res["peak_twitch_ratio_err"], "Slow MU peak/twitch ratio absolute error", "")
    summarize_metric_list(slow_res["rise_time_twitch_err_s"], "Slow MU twitch rise time absolute error", "s")
    summarize_metric_list(slow_res["half_decay_time_twitch_err_s"], "Slow MU twitch half-decay time absolute error", "s")
    summarize_metric_list(slow_res["tau_rise_1st_err_s"], "Slow MU tetanus tau_rise absolute error", "s")
    summarize_metric_list(slow_res["tau_decay_1st_err_s"], "Slow MU tetanus tau_decay absolute error", "s")
    summarize_metric_list(slow_res["fusion_index_ratio_err"], "Slow MU fusion index absolute error", "")
    print()

    # -------------------------------------------------------------------------
    # Fast MU - cat MG
    # -------------------------------------------------------------------------
    fast_cat_res = process_mu_trials(
        fast_cat_trials,
        block_title="— Fast MU trials (cat MG) —",
        avg_label="Fast Average (cat MG)",
        align=22,
        plot_points=True
    )

    summarize_metric_list(fast_cat_res["r2"], "Fast MU cat MG R²", "")
    summarize_metric_list(fast_cat_res["peak_err_pctF0"], "Fast MU cat MG peak force absolute error", "%F0")
    summarize_metric_list(fast_cat_res["peak_twitch_ratio_err"], "Fast MU cat MG peak/twitch ratio absolute error", "")
    summarize_metric_list(fast_cat_res["rise_time_twitch_err_s"], "Fast MU cat MG twitch rise time absolute error", "s")
    summarize_metric_list(fast_cat_res["half_decay_time_twitch_err_s"], "Fast MU cat MG twitch half-decay time absolute error", "s")
    summarize_metric_list(fast_cat_res["tau_rise_1st_err_s"], "Fast MU cat MG tetanus tau_rise absolute error", "s")
    summarize_metric_list(fast_cat_res["tau_decay_1st_err_s"], "Fast MU cat MG tetanus tau_decay absolute error", "s")
    summarize_metric_list(fast_cat_res["fusion_index_ratio_err"], "Fast MU cat MG fusion index absolute error", "")
    print()

    # -------------------------------------------------------------------------
    # Fast MU - cat MG, no sag
    # -------------------------------------------------------------------------
    fast_cat_nosag_res = process_mu_trials(
        fast_cat_trials_nosag,
        block_title="— Fast MU trials (cat MG) — NO SAG —",
        avg_label="Fast Average (cat MG, no sag)",
        align=30,
        plot_points=True
    )

    summarize_metric_list(fast_cat_nosag_res["r2"], "Fast MU cat MG NO SAG R²", "")
    summarize_metric_list(fast_cat_nosag_res["peak_err_pctF0"], "Fast MU cat MG NO SAG peak force absolute error", "%F0")
    summarize_metric_list(fast_cat_nosag_res["peak_twitch_ratio_err"], "Fast MU cat MG NO SAG peak/twitch ratio absolute error", "")
    summarize_metric_list(fast_cat_nosag_res["rise_time_twitch_err_s"], "Fast MU cat MG NO SAG twitch rise time absolute error", "s")
    summarize_metric_list(fast_cat_nosag_res["half_decay_time_twitch_err_s"], "Fast MU cat MG NO SAG twitch half-decay time absolute error", "s")
    summarize_metric_list(fast_cat_nosag_res["tau_rise_1st_err_s"], "Fast MU cat MG NO SAG tetanus tau_rise absolute error", "s")
    summarize_metric_list(fast_cat_nosag_res["tau_decay_1st_err_s"], "Fast MU cat MG NO SAG tetanus tau_decay absolute error", "s")
    summarize_metric_list(fast_cat_nosag_res["fusion_index_ratio_err"], "Fast MU cat MG NO SAG fusion index absolute error", "")
    print()

    # -------------------------------------------------------------------------
    # Fast MU - rat MG
    # -------------------------------------------------------------------------
    fast_rat_res = process_mu_trials(
        fast_rat_trials,
        block_title="— Fast MU trials (rat MG) —",
        avg_label="Fast Average (rat MG)",
        align=24,
        plot_points=True,
        twitch_ref=None
    )

    summarize_metric_list(fast_rat_res["r2"], "Fast MU rat MG R²", "")
    summarize_metric_list(fast_rat_res["peak_err_pctF0"], "Fast MU rat MG peak force absolute error", "%F0")
    summarize_metric_list(fast_rat_res["peak_twitch_ratio_err"], "Fast MU rat MG peak/twitch ratio absolute error", "")
    summarize_metric_list(fast_rat_res["tau_rise_1st_err_s"], "Fast MU rat MG tetanus tau_rise absolute error", "s")
    summarize_metric_list(fast_rat_res["tau_decay_1st_err_s"], "Fast MU rat MG tetanus tau_decay absolute error", "s")
    summarize_metric_list(fast_rat_res["fusion_index_ratio_err"], "Fast MU rat MG fusion index absolute error", "")
    print()

    # -------------------------------------------------------------------------
    # Fast MU - rat MG, no sag
    # -------------------------------------------------------------------------
    fast_rat_nosag_res = process_mu_trials(
        fast_rat_trials_nosag,
        block_title="— Fast MU trials (rat MG) — NO SAG —",
        avg_label="Fast Average (rat MG, no sag)",
        align=30,
        plot_points=True,
        twitch_ref=None
    )

    summarize_metric_list(fast_rat_nosag_res["r2"], "Fast MU rat MG NO SAG R²", "")
    summarize_metric_list(fast_rat_nosag_res["peak_err_pctF0"], "Fast MU rat MG NO SAG peak force absolute error", "%F0")
    summarize_metric_list(fast_rat_nosag_res["peak_twitch_ratio_err"], "Fast MU rat MG NO SAG peak/twitch ratio absolute error", "")
    summarize_metric_list(fast_rat_nosag_res["tau_rise_1st_err_s"], "Fast MU rat MG NO SAG tetanus tau_rise absolute error", "s")
    summarize_metric_list(fast_rat_nosag_res["tau_decay_1st_err_s"], "Fast MU rat MG NO SAG tetanus tau_decay absolute error", "s")
    summarize_metric_list(fast_rat_nosag_res["fusion_index_ratio_err"], "Fast MU rat MG NO SAG fusion index absolute error", "")
    print()

     # -------------------------------------------------------------------------
    # Paired statistical tests: WITH sag vs WITHOUT sag
    # Combined cat MG + rat MG matched trials
    # -------------------------------------------------------------------------
    fast_cat_match_trials = [
        ("Fast Twitch  (1 Hz)",    "twitch",    exp_F_twitch,   sim_F_twitch,   MVC_F1, time_dt_F1),
        ("Fast Unfused (25 Hz)",   "nontwitch", exp_F_unfused,  sim_F_unfused,  MVC_F1, time_dt_F1),
    ]
    fast_rat_match_trials = [
        ("Fast 25 Hz  (rat MG)",   "nontwitch", exp_F_25, sim_F_25, MVC_F2, time_dt_F2),
        ("Fast 30 Hz  (rat MG)",   "nontwitch", exp_F_30, sim_F_30, MVC_F2, time_dt_F2),
        ("Fast 35 Hz  (rat MG)",   "nontwitch", exp_F_35, sim_F_35, MVC_F2, time_dt_F2),
        ("Fast 40 Hz  (rat MG)",   "nontwitch", exp_F_40, sim_F_40, MVC_F2, time_dt_F2),
    ]

    fast_cat_match_res = process_mu_trials(
        fast_cat_match_trials,
        block_title="— Fast MU matched trials for sag statistics (cat MG) —",
        avg_label="Fast matched average (cat MG)",
        align=22,
        plot_points=False
    )

    fast_rat_match_res = process_mu_trials(
        fast_rat_match_trials,
        block_title="— Fast MU matched trials for sag statistics (rat MG) —",
        avg_label="Fast matched average (rat MG)",
        align=24,
        plot_points=False,
        twitch_ref=None
    )

    # combine matched WITH-sag results
    fast_all_match_res = {
        "mae": fast_cat_match_res["mae"] + fast_rat_match_res["mae"],
        "maxae": fast_cat_match_res["maxae"] + fast_rat_match_res["maxae"],
        "std": fast_cat_match_res["std"] + fast_rat_match_res["std"],
        "r2": fast_cat_match_res["r2"] + fast_rat_match_res["r2"],
        "peak_err_pctF0": fast_cat_match_res["peak_err_pctF0"] + fast_rat_match_res["peak_err_pctF0"],
        "peak_twitch_ratio_err": fast_cat_match_res["peak_twitch_ratio_err"] + fast_rat_match_res["peak_twitch_ratio_err"],
        "rise_time_twitch_err_s": fast_cat_match_res["rise_time_twitch_err_s"] + fast_rat_match_res["rise_time_twitch_err_s"],
        "half_decay_time_twitch_err_s": fast_cat_match_res["half_decay_time_twitch_err_s"] + fast_rat_match_res["half_decay_time_twitch_err_s"],
        "tau_rise_1st_err_s": fast_cat_match_res["tau_rise_1st_err_s"] + fast_rat_match_res["tau_rise_1st_err_s"],
        "tau_decay_1st_err_s": fast_cat_match_res["tau_decay_1st_err_s"] + fast_rat_match_res["tau_decay_1st_err_s"],
        "fusion_index_ratio_err": fast_cat_match_res["fusion_index_ratio_err"] + fast_rat_match_res["fusion_index_ratio_err"],
    }

    # combine matched WITHOUT-sag results
    fast_all_nosag_res = {
        "mae": fast_cat_nosag_res["mae"] + fast_rat_nosag_res["mae"],
        "maxae": fast_cat_nosag_res["maxae"] + fast_rat_nosag_res["maxae"],
        "std": fast_cat_nosag_res["std"] + fast_rat_nosag_res["std"],
        "r2": fast_cat_nosag_res["r2"] + fast_rat_nosag_res["r2"],
        "peak_err_pctF0": fast_cat_nosag_res["peak_err_pctF0"] + fast_rat_nosag_res["peak_err_pctF0"],
        "peak_twitch_ratio_err": fast_cat_nosag_res["peak_twitch_ratio_err"] + fast_rat_nosag_res["peak_twitch_ratio_err"],
        "rise_time_twitch_err_s": fast_cat_nosag_res["rise_time_twitch_err_s"] + fast_rat_nosag_res["rise_time_twitch_err_s"],
        "half_decay_time_twitch_err_s": fast_cat_nosag_res["half_decay_time_twitch_err_s"] + fast_rat_nosag_res["half_decay_time_twitch_err_s"],
        "tau_rise_1st_err_s": fast_cat_nosag_res["tau_rise_1st_err_s"] + fast_rat_nosag_res["tau_rise_1st_err_s"],
        "tau_decay_1st_err_s": fast_cat_nosag_res["tau_decay_1st_err_s"] + fast_rat_nosag_res["tau_decay_1st_err_s"],
        "fusion_index_ratio_err": fast_cat_nosag_res["fusion_index_ratio_err"] + fast_rat_nosag_res["fusion_index_ratio_err"],
    }

    run_sag_statistics(
        fast_all_match_res,
        fast_all_nosag_res,
        title="SAG EFFECT STATISTICS — combined cat MG + rat MG"
    )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    summarize_trials(
        slow_res["mae"],
        slow_res["maxae"],
        label="— MU summary: S MU (all trials) —",
        unit="% F0"
    )
    summarize_trials(
        fast_cat_res["mae"],
        fast_cat_res["maxae"],
        label="— MU summary: F MU cat MG (all trials) —",
        unit="% F0"
    )
    summarize_trials(
        fast_cat_nosag_res["mae"],
        fast_cat_nosag_res["maxae"],
        label="— MU summary: F MU cat MG NO SAG (all trials) —",
        unit="% F0"
    )
    summarize_trials(
        fast_rat_res["mae"],
        fast_rat_res["maxae"],
        label="— MU summary: F MU rat MG (all trials) —",
        unit="% F0"
    )
    summarize_trials(
        fast_rat_nosag_res["mae"],
        fast_rat_nosag_res["maxae"],
        label="— MU summary: F MU rat MG NO SAG (all trials) —",
        unit="% F0"
    )


elif benchmark == 'fast_M_iso':  # Millard 2025 experiments

    sim_path = base_path / 'fast_M' / 'sim'
    exp_path = base_path / 'fast_M' / 'exp'

    # -------------------------------------------------------------------------
    # Load FFR twitch (separate dataset)
    # -------------------------------------------------------------------------
    exp_FFR_1 = np.load(exp_path / 'iso_FFR_twitch.npy')
    sim_FFR_1 = np.load(sim_path / 'iso_FFR_twitch.npy')

    # -------------------------------------------------------------------------
    # Load FFR tetani
    # -------------------------------------------------------------------------
    exp_FFR_30  = np.load(exp_path / 'iso_FFR_30.npy')
    exp_FFR_50  = np.load(exp_path / 'iso_FFR_50.npy')
    exp_FFR_60  = np.load(exp_path / 'iso_FFR_60.npy')
    exp_FFR_70  = np.load(exp_path / 'iso_FFR_70.npy')
    exp_FFR_80  = np.load(exp_path / 'iso_FFR_80.npy')
    exp_FFR_90  = np.load(exp_path / 'iso_FFR_90.npy')
    exp_FFR_100 = np.load(exp_path / 'iso_FFR_100.npy')
    exp_FFR_120 = np.load(exp_path / 'iso_FFR_120.npy')

    sim_FFR_30  = np.load(sim_path / 'iso_FFR_30.npy')
    sim_FFR_50  = np.load(sim_path / 'iso_FFR_50.npy')
    sim_FFR_60  = np.load(sim_path / 'iso_FFR_60.npy')
    sim_FFR_70  = np.load(sim_path / 'iso_FFR_70.npy')
    sim_FFR_80  = np.load(sim_path / 'iso_FFR_80.npy')
    sim_FFR_90  = np.load(sim_path / 'iso_FFR_90.npy')
    sim_FFR_100 = np.load(sim_path / 'iso_FFR_100.npy')
    sim_FFR_120 = np.load(sim_path / 'iso_FFR_120.npy')

    freqs = [30, 50, 60, 70, 80, 90, 100, 120]
    exp_FFR_series = [
        exp_FFR_30, exp_FFR_50, exp_FFR_60, exp_FFR_70,
        exp_FFR_80, exp_FFR_90, exp_FFR_100, exp_FFR_120
    ]
    sim_FFR_series = [
        sim_FFR_30, sim_FFR_50, sim_FFR_60, sim_FFR_70,
        sim_FFR_80, sim_FFR_90, sim_FFR_100, sim_FFR_120
    ]

    # -------------------------------------------------------------------------
    # Load FLR
    # -------------------------------------------------------------------------
    exp_FLR_050 = np.load(exp_path / 'iso_FLR_0.50.npy')
    exp_FLR_100 = np.load(exp_path / 'iso_FLR_1.00.npy')
    exp_FLR_150 = np.load(exp_path / 'iso_FLR_1.50.npy')
    exp_FLR_200 = np.load(exp_path / 'iso_FLR_2.00.npy')
    exp_FLR_250 = np.load(exp_path / 'iso_FLR_2.50.npy')
    exp_FLR_300 = np.load(exp_path / 'iso_FLR_3.00.npy')
    exp_FLR_350 = np.load(exp_path / 'iso_FLR_3.50.npy')
    exp_FLR_400 = np.load(exp_path / 'iso_FLR_4.00.npy')

    sim_FLR_050 = np.load(sim_path / 'iso_FLR_0.50.npy')
    sim_FLR_100 = np.load(sim_path / 'iso_FLR_1.00.npy')
    sim_FLR_150 = np.load(sim_path / 'iso_FLR_1.50.npy')
    sim_FLR_200 = np.load(sim_path / 'iso_FLR_2.00.npy')
    sim_FLR_250 = np.load(sim_path / 'iso_FLR_2.50.npy')
    sim_FLR_300 = np.load(sim_path / 'iso_FLR_3.00.npy')
    sim_FLR_350 = np.load(sim_path / 'iso_FLR_3.50.npy')
    sim_FLR_400 = np.load(sim_path / 'iso_FLR_4.00.npy')

    disp_mm = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    exp_FLR_series = [
        exp_FLR_050, exp_FLR_100, exp_FLR_150, exp_FLR_200,
        exp_FLR_250, exp_FLR_300, exp_FLR_350, exp_FLR_400
    ]
    sim_FLR_series = [
        sim_FLR_050, sim_FLR_100, sim_FLR_150, sim_FLR_200,
        sim_FLR_250, sim_FLR_300, sim_FLR_350, sim_FLR_400
    ]

    # -------------------------------------------------------------------------
    # Time vectors
    # -------------------------------------------------------------------------
    t_end_twitch = len(exp_FFR_1) * dt
    time_dt_twitch = np.arange(0, t_end_twitch, dt)

    t_end = 0.9991
    time_dt = np.arange(0, t_end, dt)

    MVC = 2.49

    # -------------------------------------------------------------------------
    # Plots (updated: twitch separated)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    ax_twitch_exp = axes[0, 0]
    ax_ffr_exp    = axes[0, 1]
    ax_ffr_sim    = axes[0, 2]
    ax_blank      = axes[1, 0]
    ax_flr_exp    = axes[1, 1]
    ax_flr_sim    = axes[1, 2]

    # Twitch subplot
    ax_twitch_exp.plot(time_dt_twitch, exp_FFR_1, 'k', label='Experimental')
    ax_twitch_exp.plot(time_dt_twitch, sim_FFR_1, 'r', label='Simulated')
    ax_twitch_exp.set_title("Twitch (1 Hz)", weight='bold')
    ax_twitch_exp.set_ylabel("Rat EDL (Fast)\nForce [N]", weight='bold')
    ax_twitch_exp.set_xlabel("Time [s]", weight='bold')
    ax_twitch_exp.legend(loc='upper right', fontsize=7)

    # FFR panels
    n_ffr = len(freqs)
    gray_levels = np.linspace(0.75, 0.15, n_ffr)
    red_levels = np.linspace(0.4, 1.0, n_ffr)

    for i, (f, y) in enumerate(zip(freqs, exp_FFR_series)):
        ax_ffr_exp.plot(time_dt, y, color=str(gray_levels[i]), label=f"{f} Hz")
    ax_ffr_exp.set_title("Experimental Tetani", weight='bold')
    ax_ffr_exp.set_ylim([-0.08, 1.7])
    ax_ffr_exp.set_xlabel("Time [s]", weight='bold')
    ax_ffr_exp.legend(loc='upper right', fontsize=7)

    for i, (f, y) in enumerate(zip(freqs, sim_FFR_series)):
        ax_ffr_sim.plot(time_dt, y, color=(red_levels[i], 0, 0), label=f"{f} Hz")
    ax_ffr_sim.set_ylim([-0.08, 1.7])
    ax_ffr_sim.set_title("Simulated Tetani", weight='bold')
    ax_ffr_sim.set_xlabel("Time [s]", weight='bold')
    ax_ffr_sim.legend(loc='upper right', fontsize=7)

    # FLR panels
    n_flr = len(disp_mm)
    gray_levels_flr = np.linspace(0.15, 0.75, n_flr)
    red_levels_flr = np.linspace(0.4, 1.0, n_flr)

    ax_blank.axis('off')

    for i, (d, y) in enumerate(zip(disp_mm, exp_FLR_series)):
        ax_flr_exp.plot(time_dt, y, color=str(gray_levels_flr[i]), label=f"$\Delta L$ = + {d:.1f} mm")
    ax_flr_exp.set_title("Experimental Tetani (80 Hz)", weight='bold')
    ax_flr_exp.set_ylabel("Rat EDL (Fast)\nForce [N]", weight='bold')
    ax_flr_exp.set_xlabel("Time [s]", weight='bold')
    ax_flr_exp.legend(loc='upper right', fontsize=7)

    for i, (d, y) in enumerate(zip(disp_mm, sim_FLR_series)):
        ax_flr_sim.plot(time_dt, y, color=(red_levels_flr[i], 0, 0), label=f"$\Delta L$ = + {d:.1f} mm")
    ax_flr_sim.set_title("Simulated Tetani (80 Hz)", weight='bold')
    ax_flr_sim.set_xlabel("Time [s]", weight='bold')
    ax_flr_sim.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.show()

    print("\n=== Benchmark: fast_M_iso ===")
    print(f"MVC = {MVC:.2f} N\n")

    # -------------------------------------------------------------------------
    # Trial definitions
    # -------------------------------------------------------------------------
    twitch_trial = ("1 Hz", "twitch", exp_FFR_1, sim_FFR_1, MVC, time_dt_twitch)

    ffr_trials = [
        ("30 Hz",  "nontwitch", exp_FFR_30,  sim_FFR_30,  MVC, time_dt),
        ("50 Hz",  "nontwitch", exp_FFR_50,  sim_FFR_50,  MVC, time_dt),
        ("60 Hz",  "nontwitch", exp_FFR_60,  sim_FFR_60,  MVC, time_dt),
        ("70 Hz",  "nontwitch", exp_FFR_70,  sim_FFR_70,  MVC, time_dt),
        ("80 Hz",  "nontwitch", exp_FFR_80,  sim_FFR_80,  MVC, time_dt),
        ("90 Hz",  "nontwitch", exp_FFR_90,  sim_FFR_90,  MVC, time_dt),
        ("100 Hz", "nontwitch", exp_FFR_100, sim_FFR_100, MVC, time_dt),
        ("120 Hz", "nontwitch", exp_FFR_120, sim_FFR_120, MVC, time_dt),
    ]

    flr_trials = [
        ("+0.5 mm", "nontwitch", exp_FLR_050, sim_FLR_050, MVC, time_dt),
        ("+1.0 mm", "nontwitch", exp_FLR_100, sim_FLR_100, MVC, time_dt),
        ("+1.5 mm", "nontwitch", exp_FLR_150, sim_FLR_150, MVC, time_dt),
        ("+2.0 mm", "nontwitch", exp_FLR_200, sim_FLR_200, MVC, time_dt),
        ("+2.5 mm", "nontwitch", exp_FLR_250, sim_FLR_250, MVC, time_dt),
        ("+3.0 mm", "nontwitch", exp_FLR_300, sim_FLR_300, MVC, time_dt),
        ("+3.5 mm", "nontwitch", exp_FLR_350, sim_FLR_350, MVC, time_dt),
        ("+4.0 mm", "nontwitch", exp_FLR_400, sim_FLR_400, MVC, time_dt),
    ]

    # -------------------------------------------------------------------------
    # Helper to process one fast_M_iso group
    # -------------------------------------------------------------------------
    def process_fast_iso_trials(
        trials,
        block_title,
        avg_label,
        align=10,
        plot_points=True,
        peak_window_s=0.05,
        twitch_ref=None,   # tuple: (twitch_exp, twitch_sim, F0, time)
    ):
        mae_list, maxae_list, std_list = [], [], []
        r2_list = []

        peak_err_list = []
        peak_twitch_ratio_err_list = []

        rise_time_twitch_err_list = []
        half_decay_time_twitch_err_list = []

        tau_rise_1st_err_list = []
        tau_decay_1st_err_list = []
        fusion_index_ratio_err_list = []

        print(block_title)

        # -----------------------------
        # reference twitch peaks
        # -----------------------------
        twitch_peak_exp = np.nan
        twitch_peak_sim = np.nan

        if twitch_ref is not None:
            twitch_exp, twitch_sim, twitch_F0, twitch_time = twitch_ref
            twitch_metrics_exp = compute_isometric_trial_metrics(
                force=twitch_exp, time=twitch_time, F0=twitch_F0, trial_type="twitch"
            )
            twitch_metrics_sim = compute_isometric_trial_metrics(
                force=twitch_sim, time=twitch_time, F0=twitch_F0, trial_type="twitch"
            )
            twitch_peak_exp = twitch_metrics_exp["peak_force"]
            twitch_peak_sim = twitch_metrics_sim["peak_force"]
        else:
            raise ValueError("A twitch reference trial is required to compute peak/twitch ratios.")

        for name, trial_type, exp, sim, F0, time_dt_local in trials:
            eval_out = compute_full_isometric_evaluation(
                exp=exp,
                sim=sim,
                time=time_dt_local,
                F0=F0,
                trial_type=trial_type,
                twitch_peak_exp=twitch_peak_exp,
                twitch_peak_sim=twitch_peak_sim,
                peak_window_s=peak_window_s,
                force_threshold=0.0,
            )

            mae = eval_out["mae"]
            maxae = eval_out["maxae"]
            stde = eval_out["stde"]
            r2 = eval_out["R2"]

            metrics_exp = eval_out["metrics_exp"]
            metrics_sim = eval_out["metrics_sim"]
            metric_errors = eval_out["metric_errors"]

            mae_list.append(mae)
            maxae_list.append(maxae)
            std_list.append(stde)
            r2_list.append(r2)

            peak_err_list.append(metric_errors["peak_force_err_pctF0"])
            peak_twitch_ratio_err_list.append(metric_errors["peak_twitch_ratio_err"])

            rise_time_twitch_err_list.append(metric_errors["rise_time_twitch_err_s"])
            half_decay_time_twitch_err_list.append(metric_errors["half_decay_time_twitch_err_s"])

            tau_rise_1st_err_list.append(metric_errors["tau_rise_1st_err_s"])
            tau_decay_1st_err_list.append(metric_errors["tau_decay_1st_err_s"])
            fusion_index_ratio_err_list.append(metric_errors["fusion_index_ratio_err"])

            print(
                f"{name:>{align}}:  mAE = {mae:6.2f}% F0  (std = {stde:6.2f})   "
                f"MAE = {maxae:6.2f}% F0   R² = {r2:.3f}"
            )

            print_nontwitch_metric_summary(
                label="",
                metrics_exp=metrics_exp,
                metrics_sim=metrics_sim,
                metric_errors=metric_errors,
                align=align
            )

            if plot_points:
                plot_nontwitch_metric_points(
                    time=time_dt_local,
                    force_exp=exp,
                    force_sim=sim,
                    metrics_exp=metrics_exp,
                    metrics_sim=metrics_sim,
                    title=name
                )

        mean_mae = np.nanmean(mae_list)
        std_mae = np.nanstd(mae_list)
        print(f"{avg_label}: {mean_mae:6.2f}% F0  (std = {std_mae:6.2f})\n")

        return {
            "mae": mae_list,
            "maxae": maxae_list,
            "std": std_list,
            "r2": r2_list,
            "peak_err_pctF0": peak_err_list,
            "peak_twitch_ratio_err": peak_twitch_ratio_err_list,
            "rise_time_twitch_err_s": rise_time_twitch_err_list,
            "half_decay_time_twitch_err_s": half_decay_time_twitch_err_list,
            "tau_rise_1st_err_s": tau_rise_1st_err_list,
            "tau_decay_1st_err_s": tau_decay_1st_err_list,
            "fusion_index_ratio_err": fusion_index_ratio_err_list,
        }

    # -------------------------------------------------------------------------
    # Twitch dataset handled separately
    # -------------------------------------------------------------------------
    twitch_eval = compute_full_isometric_evaluation(
        exp=twitch_trial[2],
        sim=twitch_trial[3],
        time=twitch_trial[5],
        F0=twitch_trial[4],
        trial_type="twitch",
        twitch_peak_exp=np.nan,   # not meaningful for twitch itself
        twitch_peak_sim=np.nan,
        peak_window_s=0.05,
        force_threshold=0.0,
    )

    print("— FFR twitch —")
    print(
        f"{'1 Hz':>7}:  mAE = {twitch_eval['mae']:6.2f}% F0  (std = {twitch_eval['stde']:6.2f})   "
        f"MAE = {twitch_eval['maxae']:6.2f}% F0   R² = {twitch_eval['R2']:.3f}"
    )
    print_twitch_metric_summary(
        label="",
        metrics_exp=twitch_eval["metrics_exp"],
        metrics_sim=twitch_eval["metrics_sim"],
        metric_errors=twitch_eval["metric_errors"],
        align=7
    )
    plot_twitch_metric_points(
        time=twitch_trial[5],
        force_exp=twitch_trial[2],
        force_sim=twitch_trial[3],
        metrics_exp=twitch_eval["metrics_exp"],
        metrics_sim=twitch_eval["metrics_sim"],
        title="1 Hz"
    )
    print()

    summarize_metric_list([twitch_eval["R2"]], "FFR twitch R²", "")
    summarize_metric_list([twitch_eval["metric_errors"]["peak_force_err_pctF0"]], "FFR twitch peak force absolute error", "%F0")
    summarize_metric_list([twitch_eval["metric_errors"]["rise_time_twitch_err_s"]], "FFR twitch rise time absolute error", "s")
    summarize_metric_list([twitch_eval["metric_errors"]["half_decay_time_twitch_err_s"]], "FFR twitch half-decay time absolute error", "s")
    print()

    # -------------------------------------------------------------------------
    # FFR tetani
    # -------------------------------------------------------------------------
    ffr_res = process_fast_iso_trials(
        ffr_trials,
        block_title="— FFR tetani —",
        avg_label="average mAE",
        align=7,
        plot_points=True,
        twitch_ref=(twitch_trial[2], twitch_trial[3], twitch_trial[4], twitch_trial[5])
    )

    summarize_metric_list(ffr_res["r2"], "FFR R²", "")
    summarize_metric_list(ffr_res["peak_err_pctF0"], "FFR peak force absolute error", "%F0")
    summarize_metric_list(ffr_res["peak_twitch_ratio_err"], "FFR peak/twitch ratio absolute error", "")
    summarize_metric_list(ffr_res["tau_rise_1st_err_s"], "FFR tetanus tau_rise absolute error", "s")
    summarize_metric_list(ffr_res["tau_decay_1st_err_s"], "FFR tetanus tau_decay absolute error", "s")
    summarize_metric_list(ffr_res["fusion_index_ratio_err"], "FFR fusion index absolute error", "")
    print()

    # -------------------------------------------------------------------------
    # FLR
    # Use FFR twitch as reference twitch
    # -------------------------------------------------------------------------
    flr_res = process_fast_iso_trials(
        flr_trials,
        block_title="— FLR errors —",
        avg_label="average mAE",
        align=10,
        plot_points=True,
        twitch_ref=(twitch_trial[2], twitch_trial[3], twitch_trial[4], twitch_trial[5])
    )

    summarize_metric_list(flr_res["r2"], "FLR R²", "")
    summarize_metric_list(flr_res["peak_err_pctF0"], "FLR peak force absolute error", "%F0")
    summarize_metric_list(flr_res["peak_twitch_ratio_err"], "FLR peak/twitch ratio absolute error", "")
    summarize_metric_list(flr_res["tau_rise_1st_err_s"], "FLR tetanus tau_rise absolute error", "s")
    summarize_metric_list(flr_res["tau_decay_1st_err_s"], "FLR tetanus tau_decay absolute error", "s")
    summarize_metric_list(flr_res["fusion_index_ratio_err"], "FLR fusion index absolute error", "")
    print()

    summarize_trials(
        [twitch_eval["mae"]] + ffr_res["mae"],
        [twitch_eval["maxae"]] + ffr_res["maxae"],
        label="— fast_M_iso summary: FFR (twitch + tetani) —",
        unit="% F0"
    )

    summarize_trials(
        flr_res["mae"],
        flr_res["maxae"],
        label="— fast_M_iso summary: FLR (all ΔL) —",
        unit="% F0"
    )

    # -------------------------------------------------------------------------
    # Error plots (updated: include twitch in FFR panel)
    # -------------------------------------------------------------------------
    ffr_x = [1] + freqs
    ffr_labels = ['1'] + [str(f) for f in freqs]

    ffr_mean_arr = np.array([twitch_eval["mae"]] + ffr_res["mae"])
    ffr_max_arr  = np.array([twitch_eval["maxae"]] + ffr_res["maxae"])
    ffr_std_arr  = np.array([twitch_eval["stde"]] + ffr_res["std"])

    flr_mean_arr = np.array(flr_res["mae"])
    flr_max_arr  = np.array(flr_res["maxae"])
    flr_std_arr  = np.array(flr_res["std"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # --- FFR: twitch + tetani ---
    lower = np.maximum(ffr_mean_arr - ffr_std_arr, 0)
    upper = ffr_mean_arr + ffr_std_arr
    ax1.plot(ffr_x, ffr_mean_arr, '-o', color='k', label='mAE ± SD')
    ax1.fill_between(ffr_x, lower, upper, color='k', alpha=0.2)
    ax1.plot(ffr_x, ffr_max_arr, '--*', color='k', label='MAE')
    ax1.set_xticks(ffr_x)
    ax1.set_xticklabels(ffr_labels)
    ax1.set_xlabel('Stimulation frequency [Hz]', fontweight='bold')
    ax1.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    ax1.set_ylim([0, 50])
    ax1.legend(loc='upper left')

    # --- FLR ---
    lower_f = np.maximum(flr_mean_arr - flr_std_arr, 0)
    upper_f = flr_mean_arr + flr_std_arr
    ax2.plot(disp_mm, flr_mean_arr, '-o', color='k', label='mAE ± SD')
    ax2.fill_between(disp_mm, lower_f, upper_f, color='k', alpha=0.2)
    ax2.plot(disp_mm, flr_max_arr, '--*', color='k', label='MAE')
    ax2.set_xlabel(r"$\boldsymbol{\Delta}\mathbf{L}$ [mm]", fontweight='bold')
    ax2.set_ylim([0, 40])

    plt.tight_layout()
    plt.show()


elif benchmark == 'fast_M_dyn':  # Brown 1999 experiments

    sim_path = base_path / 'fast_M' / 'sim'
    exp_path = base_path / 'fast_M' / 'exp'

    exp_dyn_120_095_s = np.load(exp_path / 'dyn_120_0.95_length.npy')
    exp_dyn_120_095_l = np.load(exp_path / 'dyn_120_0.95_short.npy')
    exp_dyn_20_095_s = np.load(exp_path / 'dyn_20_0.95_length.npy')
    exp_dyn_20_095_l = np.load(exp_path / 'dyn_20_0.95_short.npy')
    exp_dyn_40_08_s = np.load(exp_path / 'dyn_40_0.8_length.npy')
    exp_dyn_40_08_l = np.load(exp_path / 'dyn_40_0.8_short.npy')
    exp_dyn_40_11_s = np.load(exp_path / 'dyn_40_1.1_length.npy')
    exp_dyn_40_11_l = np.load(exp_path / 'dyn_40_1.1_short.npy')
    exp_dyn_60_095_s = np.load(exp_path / 'dyn_60_0.95_length.npy')
    exp_dyn_60_095_l = np.load(exp_path / 'dyn_60_0.95_short.npy')

    sim_dyn_120_095_s = np.load(sim_path / 'dyn_120_0.95_length.npy')
    sim_dyn_120_095_l = np.load(sim_path / 'dyn_120_0.95_short.npy')
    sim_dyn_20_095_s = np.load(sim_path / 'dyn_20_0.95_length.npy')
    sim_dyn_20_095_l = np.load(sim_path / 'dyn_20_0.95_short.npy')
    sim_dyn_40_08_s = np.load(sim_path / 'dyn_40_0.8_length.npy')
    sim_dyn_40_08_l = np.load(sim_path / 'dyn_40_0.8_short.npy')
    sim_dyn_40_11_s = np.load(sim_path / 'dyn_40_1.1_length.npy')
    sim_dyn_40_11_l = np.load(sim_path / 'dyn_40_1.1_short.npy')
    sim_dyn_60_095_s = np.load(sim_path / 'dyn_60_0.95_length.npy')
    sim_dyn_60_095_l = np.load(sim_path / 'dyn_60_0.95_short.npy')

    disp_sub_s = np.load(exp_path / 'disp_max_length_interp.npy')
    disp_sub_l = np.load(exp_path / 'disp_max_short_interp.npy')
    disp_max_s = np.load(exp_path / 'disp_sub_length_interp.npy')
    disp_max_l = np.load(exp_path / 'disp_sub_short_interp.npy')

    t_end_max = 0.16
    t_end_sub = 0.17
    time_dt_max = np.arange(0, t_end_max, dt)
    time_dt_sub = np.arange(0, t_end_sub, dt)
    MVC = 2.49

    # -------------------------------------------------------------------------
    # Plots (unchanged)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 7))

    plt.subplot(3, 3, 1)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_20_095_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_20_095_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_20_095_l[0:len(disp_sub_l)], 'red')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_20_095_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"20 Hz, at 0.95$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    plt.subplot(3, 3, 2)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_08_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_08_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_08_l[0:len(disp_sub_l)], 'red')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_08_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"40 Hz, at 0.8$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    plt.subplot(3, 3, 3)
    plt.plot(time_dt_max, exp_dyn_120_095_l, 'k', label='Experimental-Shortening')
    plt.plot(time_dt_max, exp_dyn_120_095_s, 'gray', label='Experimental-Lengthening')
    plt.plot(time_dt_max, sim_dyn_120_095_l, 'r', label='Simulated-Shortening')
    plt.plot(time_dt_max, sim_dyn_120_095_s, 'orange', label='Simulated-Lengthening')
    plt.axvline(time_dt_sub[1100], color='gray', linestyle='--', linewidth=1)
    plt.title(r"120 Hz, at 0.95$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.legend(loc=(0.2, 1.3), fontsize=12)
    plt.ylim((0, 1.7))
    plt.xlim((0.08, 0.16))

    plt.subplot(3, 3, 4)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_60_095_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_60_095_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_60_095_l[0:len(disp_sub_l)], 'r')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_60_095_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"60 Hz, at 0.95$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    plt.subplot(3, 3, 5)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_11_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_11_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_11_l[0:len(disp_sub_l)], 'r')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_11_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"40 Hz, at 1.1$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    plt.subplot(3, 3, 7)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], disp_sub_l, 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_s)], disp_sub_s, 'gray')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(u"Displacement", x=0.3, y=0.99, weight='bold')
    plt.ylabel(r'$\boldsymbol{\Delta}\mathbf{L^{CE}_{0}}$', weight='bold', fontsize=14)
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((-0.1, 0.12))

    plt.subplot(3, 3, 8)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], disp_sub_l, 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_s)], disp_sub_s, 'gray')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(u"Displacement", x=0.3, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((-0.1, 0.12))
    plt.subplot(3, 3, 7)

    plt.subplot(3, 3, 9)
    plt.plot(time_dt_max, disp_max_l[0:len(time_dt_max)], 'k')
    plt.plot(time_dt_max, disp_max_s[0:len(time_dt_max)], 'gray')
    plt.axvline(time_dt_sub[1119], color='gray', linestyle='--', linewidth=1)
    plt.title(u"Displacement", x=0.3, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((-0.1, 0.12))
    plt.xlim((0.08, 0.16))

    fig.text(0.03, 0.67, 'Cat CF (Fast)\nNormalized force', va='center', rotation='vertical',
             weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Errors + R²
    # -------------------------------------------------------------------------
    print("\n=== Benchmark: fast_M_dyn (errors) ===\n")

    cond_labels = [
        "20 Hz (0.95 L0)",
        "40 Hz (0.80 L0)",
        "60 Hz (0.95 L0)",
        "40 Hz (1.10 L0)",
        "120 Hz (0.95 L0)",
    ]

    # SHORTENING
    short_exp = [
        exp_dyn_20_095_l,
        exp_dyn_40_08_l,
        exp_dyn_60_095_l,
        exp_dyn_40_11_l,
        exp_dyn_120_095_l,
    ]
    short_sim = [
        sim_dyn_20_095_l,
        sim_dyn_40_08_l,
        sim_dyn_60_095_l,
        sim_dyn_40_11_l,
        sim_dyn_120_095_l,
    ]

    # LENGTHENING
    length_exp = [
        exp_dyn_20_095_s,
        exp_dyn_40_08_s,
        exp_dyn_60_095_s,
        exp_dyn_40_11_s,
        exp_dyn_120_095_s,
    ]
    length_sim = [
        sim_dyn_20_095_s,
        sim_dyn_40_08_s,
        sim_dyn_60_095_s,
        sim_dyn_40_11_s,
        sim_dyn_120_095_s,
    ]

    idx_sub = 1119
    idx_120_short = 1083
    idx_120_len = 1090

    def compute_err_r2_list(exp_list, sim_list, is_short=True):
        mean_list, max_list, std_list, r2_list = [], [], [], []

        for i, (e, s) in enumerate(zip(exp_list, sim_list)):
            if i < 4:
                start = idx_sub
            else:
                start = idx_120_short if is_short else idx_120_len

            e_seg = np.asarray(e[start:], dtype=float)
            s_seg = np.asarray(s[start:], dtype=float)

            abs_err = np.abs(s_seg - e_seg) * 100.0
            mean_list.append(np.mean(abs_err))
            max_list.append(np.max(abs_err))
            std_list.append(np.std(abs_err))
            r2_list.append(compute_r2(e_seg, s_seg))

        return (
            np.array(mean_list),
            np.array(max_list),
            np.array(std_list),
            np.array(r2_list),
        )

    short_mean, short_max, short_std, short_r2 = compute_err_r2_list(short_exp, short_sim, is_short=True)
    length_mean, length_max, length_std, length_r2 = compute_err_r2_list(length_exp, length_sim, is_short=False)

    print("Shortening:")
    for lab, mae, mxe, stde, r2 in zip(cond_labels, short_mean, short_max, short_std, short_r2):
        print(f"  {lab:18s}  mAE = {mae:6.2f}%  (std = {stde:6.2f})   MAE = {mxe:6.2f}%   R² = {r2:.4f}")

    print(
        f"{'Shortening mean':>20s}  "
        f"mAE = {np.nanmean(short_mean):6.2f}%  "
        f"(std = {np.nanstd(short_mean):6.2f})   "
        f"MAE = {np.nanmean(short_max):6.2f}%   "
        f"R² = {np.nanmean(short_r2):.4f}  (std = {np.nanstd(short_r2):.4f})"
    )

    print("\nLengthening:")
    for lab, mae, mxe, stde, r2 in zip(cond_labels, length_mean, length_max, length_std, length_r2):
        print(f"  {lab:18s}  mAE = {mae:6.2f}%  (std = {stde:6.2f})   MAE = {mxe:6.2f}%   R² = {r2:.4f}")

    print(
        f"{'Lengthening mean':>20s}  "
        f"mAE = {np.nanmean(length_mean):6.2f}%  "
        f"(std = {np.nanstd(length_mean):6.2f})   "
        f"MAE = {np.nanmean(length_max):6.2f}%   "
        f"R² = {np.nanmean(length_r2):.4f}  (std = {np.nanstd(length_r2):.4f})"
    )
    print()

    all_mae = list(short_mean) + list(length_mean)
    all_maxe = list(short_max) + list(length_max)
    all_r2 = list(short_r2) + list(length_r2)

    summarize_trials(
        all_mae,
        all_maxe,
        label="— fast_M_dyn summary: (all conditions) —",
        unit="%"
    )
    summarize_metric_list(all_r2, "— fast_M_dyn summary: R² (all conditions) —", "")

    # -------------------------------------------------------------------------
    # PLOT errors (unchanged)
    # -------------------------------------------------------------------------
    x = np.arange(1, 5 + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    lower_s = np.maximum(short_mean - short_std, 0)
    upper_s = short_mean + short_std
    ax1.plot(x, short_mean, '-o', color='k', label='mAE ± SD')
    ax1.fill_between(x, lower_s, upper_s, color='k', alpha=0.2)
    ax1.plot(x, short_max, '--*', color='k', label='MAE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels, rotation=20, ha='right')
    ax1.set_title('Shortening', fontweight='bold')
    ax1.set_ylabel('Error [%]', fontweight='bold')
    ax1.set_xlabel('Condition', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left')

    lower_l = np.maximum(length_mean - length_std, 0)
    upper_l = length_mean + length_std
    ax2.plot(x, length_mean, '-o', color='k', label='mAE ± SD')
    ax2.fill_between(x, lower_l, upper_l, color='k', alpha=0.2)
    ax2.plot(x, length_max, '--*', color='k', label='MAE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels, rotation=20, ha='right')
    ax2.set_xlabel('Condition', fontweight='bold')
    ax2.set_title('Lengthening', fontweight='bold')
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()

