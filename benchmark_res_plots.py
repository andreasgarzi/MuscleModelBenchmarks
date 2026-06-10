"""
Author: Andrea Sgarzi
Email: a.sgarzi@ad.unsw.edu.au
Affiliation: University of New South Wales (UNSW), Graduate School of Biomedical Engineering (GSBE)

Autonomous benchmark result and plotting script.
Use the command-line argument to choose one benchmark, or use 'all'.
"""

import argparse
from pathlib import Path
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

save_figures = False # option to save figures in benchmark_Figures
figures_path = Path() / "benchmark_Figures" # figures folder

# Numerical and plotting helpers

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


# Benchmarks

def run_slow_dyn2():
    
    base_path = Path() / 'benchmark_Results'
    dt = 1e-4

    sim_path = base_path / 'Muscle' / 'slow_dyn2' / 'sim'
    exp_path = base_path / 'Muscle' / 'slow_dyn2' / 'exp'

    exp1 = np.load(exp_path / 'rat_SOL_0.05mm_force.npy')
    exp2 = np.load(exp_path / 'rat_SOL_0.10mm_force.npy')
    exp3 = np.load(exp_path / 'rat_SOL_0.25mm_force.npy')
    exp4 = np.load(exp_path / 'rat_SOL_0.50mm_force.npy')
    exp5 = np.load(exp_path / 'rat_SOL_1.00mm_force.npy')
    exp6 = np.load(exp_path / 'rat_SOL_2.00mm_force.npy')

    sim1 = np.load(sim_path / 'rat_SOL_0.05mm_force.npy')
    sim2 = np.load(sim_path / 'rat_SOL_0.10mm_force.npy')
    sim3 = np.load(sim_path / 'rat_SOL_0.25mm_force.npy')
    sim4 = np.load(sim_path / 'rat_SOL_0.50mm_force.npy')
    sim5 = np.load(sim_path / 'rat_SOL_1.00mm_force.npy')
    sim6 = np.load(sim_path / 'rat_SOL_2.00mm_force.npy')

    t_end = 2
    time_dt = np.arange(0, t_end, dt)
    MVC = 1.32

    fig = plt.figure(figsize=(9, 8))

    gs = GridSpec(8, 1, height_ratios=[1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 0.15, 3.0], hspace=0.55, figure=fig)
    axes = []
    for i in range(6):
        axes.append(fig.add_subplot(gs[i, 0]))

    ax_err = fig.add_subplot(gs[7, 0])

    # -------------------------------------------------------------------------
    # FORCE SUBPLOTS 
    # -------------------------------------------------------------------------
    data = [
        (exp1, sim1, u"\u00B1 0.05 mm"),
        (exp2, sim2, u"\u00B1 0.10 mm"),
        (exp3, sim3, u"\u00B1 0.25 mm"),
        (exp4, sim4, u"\u00B1 0.50 mm"),
        (exp5, sim5, u"\u00B1 1.00 mm"),
        (exp6, sim6, u"\u00B1 2.00 mm"),
    ]

    for i, (exp, sim, label) in enumerate(data):
        ax = axes[i]

        ax.plot(time_dt, exp, 'k')
        ax.plot(time_dt, sim, 'r')

        ax.set_title(label, x=0.1, y=0.97, weight='bold')
        ax.set_xlim((-0.03, 2.03))
        ax.set_ylim((0, 2))

        if i < 5:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Time [s]', weight='bold', fontsize=11)

    fig.text(
        0.077, 0.63,
        'Rat Soleus (Slow) Force [N]',
        va='center',
        rotation='vertical',
        weight='bold',
        fontsize=12
    )

    # -------------------------------------------------------------------------
    # ERRORS 
    # -------------------------------------------------------------------------
    exp_all = [exp1, exp2, exp3, exp4, exp5, exp6]
    sim_all = [sim1, sim2, sim3, sim4, sim5, sim6]
    displacements = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]

    mean_err_list = []
    max_err_list = []
    std_err_list = []

    for exp, sim in zip(exp_all, sim_all):
        mae, maxae, stde = pct_errors(exp, sim, MVC)
        mean_err_list.append(mae)
        max_err_list.append(maxae)
        std_err_list.append(stde)

    x = np.arange(1, len(displacements) + 1)

    mean_err_arr = np.array(mean_err_list)
    max_err_arr = np.array(max_err_list)
    std_err_arr = np.array(std_err_list)

    # -------------------------------------------------------------------------
    # ERROR PLOT 
    # -------------------------------------------------------------------------
    ax_err.plot(x, mean_err_arr, '-o', color='k', label='mAE ± SD')
    ax_err.fill_between(
        x,
        mean_err_arr - std_err_arr,
        mean_err_arr + std_err_arr,
        color='k',
        alpha=0.2
    )

    ax_err.plot(x, max_err_arr, '--*', color='k', label='MAE')

    ax_err.set_xticks(x)
    ax_err.set_xticklabels([f'{d:.2f}' for d in displacements])
    ax_err.set_xlabel('Max. length variation amplitude [mm]', fontweight='bold')
    ax_err.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold', fontsize=11)
    ax_err.set_ylim(bottom=0)
    ax_err.legend()

    fig.text(0.1, 0.93, 'A', fontsize=15, fontweight='bold', ha='left', va='top')
    fig.text(0.1, 0.3, 'B', fontsize=15, fontweight='bold', ha='left', va='top')

    legend_handles = [
        Line2D([0], [0], color='k', lw=1.5, label='Experimental'),
        Line2D([0], [0], color='r', lw=1.5, label='Simulated')
        ]

    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.88, 0.97), fontsize=11)

    plt.tight_layout()
    if save_figures == True:
        plt.savefig(figures_path / 'slow_M_max_summary.png', dpi=300, bbox_inches='tight')
        
    plt.show()

def run_slow_isof_dyn1():

    # -------------------------------------------------------------------------
    # Load isometric trials
    # -------------------------------------------------------------------------

    base_path = Path() / 'benchmark_Results'
    dt = 1e-4

    sim_path_isof = base_path / 'Muscle' / 'slow_isof' / 'sim'
    exp_path_isof = base_path / 'Muscle' / 'slow_isof' / 'exp'

    sim_iso_c_10 = np.load(sim_path_isof / 'cat_SOL_10Hz_c_force.npy')
    sim_iso_c_20 = np.load(sim_path_isof / 'cat_SOL_20Hz_c_force.npy')
    sim_iso_c_30 = np.load(sim_path_isof / 'cat_SOL_30Hz_c_force.npy')
    sim_iso_v_10 = np.load(sim_path_isof / 'cat_SOL_10Hz_v_force.npy')
    sim_iso_v_20 = np.load(sim_path_isof / 'cat_SOL_20Hz_v_force.npy')
    sim_iso_v_30 = np.load(sim_path_isof / 'cat_SOL_30Hz_v_force.npy')

    exp_iso_c_10 = np.load(exp_path_isof / 'cat_SOL_10Hz_c_force.npy')
    exp_iso_c_20 = np.load(exp_path_isof / 'cat_SOL_20Hz_c_force.npy')
    exp_iso_c_30 = np.load(exp_path_isof / 'cat_SOL_30Hz_c_force.npy')
    exp_iso_v_10 = np.load(exp_path_isof / 'cat_SOL_10Hz_v_force.npy')
    exp_iso_v_20 = np.load(exp_path_isof / 'cat_SOL_20Hz_v_force.npy')
    exp_iso_v_30 = np.load(exp_path_isof / 'cat_SOL_30Hz_v_force.npy')

    t_end = 2
    time_dt = np.arange(0, t_end, dt)
    MVC = 26.13

    # -------------------------------------------------------------------------
    # Load dynamic constant trials
    # -------------------------------------------------------------------------

    sim_path_dyn1 = base_path / 'Muscle' / 'slow_dyn1' / 'sim'
    exp_path_dyn1 = base_path / 'Muscle' / 'slow_dyn1' / 'exp'

    sim_dyn_c_10_1 = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_c_1mm_force.npy') # with yielding
    sim_dyn_c_20_1 = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_c_1mm_force.npy')
    sim_dyn_c_30_1 = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_c_1mm_force.npy')
    sim_dyn_c_10_8 = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_c_8mm_force.npy')
    sim_dyn_c_20_8 = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_c_8mm_force.npy')
    sim_dyn_c_30_8 = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_c_8mm_force.npy')

    sim_dyn_c_10_1_noy = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_c_1mm_noy_force.npy') # without yielding
    sim_dyn_c_20_1_noy = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_c_1mm_noy_force.npy')
    sim_dyn_c_30_1_noy = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_c_1mm_noy_force.npy')
    sim_dyn_c_10_8_noy = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_c_8mm_noy_force.npy')
    sim_dyn_c_20_8_noy = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_c_8mm_noy_force.npy')
    sim_dyn_c_30_8_noy = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_c_8mm_noy_force.npy')

    exp_dyn_c_10_1 = np.load(exp_path_dyn1 / 'cat_SOL_10Hz_c_1mm_force.npy')
    exp_dyn_c_20_1 = np.load(exp_path_dyn1 / 'cat_SOL_20Hz_c_1mm_force.npy')
    exp_dyn_c_30_1 = np.load(exp_path_dyn1 / 'cat_SOL_30Hz_c_1mm_force.npy')
    exp_dyn_c_10_8 = np.load(exp_path_dyn1 / 'cat_SOL_10Hz_c_8mm_force.npy')
    exp_dyn_c_20_8 = np.load(exp_path_dyn1 / 'cat_SOL_20Hz_c_8mm_force.npy')
    exp_dyn_c_30_8 = np.load(exp_path_dyn1 / 'cat_SOL_30Hz_c_8mm_force.npy')

    # -------------------------------------------------------------------------
    # Load dynamic random trials
    # -------------------------------------------------------------------------

    sim_dyn_v_10_1 = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_v_1mm_force.npy')
    sim_dyn_v_20_1 = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_v_1mm_force.npy')
    sim_dyn_v_30_1 = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_v_1mm_force.npy')
    sim_dyn_v_10_8 = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_v_8mm_force.npy')
    sim_dyn_v_20_8 = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_v_8mm_force.npy')
    sim_dyn_v_30_8 = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_v_8mm_force.npy')

    sim_dyn_v_10_1_noy = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_v_1mm_noy_force.npy')
    sim_dyn_v_20_1_noy = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_v_1mm_noy_force.npy')
    sim_dyn_v_30_1_noy = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_v_1mm_noy_force.npy')
    sim_dyn_v_10_8_noy = np.load(sim_path_dyn1 / 'cat_SOL_10Hz_v_8mm_noy_force.npy')
    sim_dyn_v_20_8_noy = np.load(sim_path_dyn1 / 'cat_SOL_20Hz_v_8mm_noy_force.npy')
    sim_dyn_v_30_8_noy = np.load(sim_path_dyn1 / 'cat_SOL_30Hz_v_8mm_noy_force.npy')

    exp_dyn_v_10_1 = np.load(exp_path_dyn1 / 'cat_SOL_10Hz_v_1mm_force.npy')
    exp_dyn_v_20_1 = np.load(exp_path_dyn1 / 'cat_SOL_20Hz_v_1mm_force.npy')
    exp_dyn_v_30_1 = np.load(exp_path_dyn1 / 'cat_SOL_30Hz_v_1mm_force.npy')
    exp_dyn_v_10_8 = np.load(exp_path_dyn1 / 'cat_SOL_10Hz_v_8mm_force.npy')
    exp_dyn_v_20_8 = np.load(exp_path_dyn1 / 'cat_SOL_20Hz_v_8mm_force.npy')
    exp_dyn_v_30_8 = np.load(exp_path_dyn1 / 'cat_SOL_30Hz_v_8mm_force.npy')

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
      
    ####################################################################################
    # Plots
    ####################################################################################

    def build_err(trials, MVC):
        mean_list, max_list, std_list = [], [], []
        for _, exp, sim in trials:
            mae, maxae, stde = pct_errors(exp, sim, MVC)
            mean_list.append(mae)
            max_list.append(maxae)
            std_list.append(stde)
        return np.array(mean_list), np.array(max_list), np.array(std_list)

    iso_const_plot = [
        ("Const 10 Hz", exp_iso_c_10, sim_iso_c_10),
        ("Const 20 Hz", exp_iso_c_20, sim_iso_c_20),
        ("Const 30 Hz", exp_iso_c_30, sim_iso_c_30),
    ]
    mean_iso_const, max_iso_const, std_iso_const = build_err(iso_const_plot, MVC)

    iso_rand_plot = [
        ("Rand 10 Hz", exp_iso_v_10, sim_iso_v_10),
        ("Rand 20 Hz", exp_iso_v_20, sim_iso_v_20),
        ("Rand 30 Hz", exp_iso_v_30, sim_iso_v_30),
    ]
    mean_iso_rand, max_iso_rand, std_iso_rand = build_err(iso_rand_plot, MVC)

    dyn_const_trials = [
        ("Const 10 Hz, 1mm", exp_dyn_c_10_1, sim_dyn_c_10_1),
        ("Const 20 Hz, 1mm", exp_dyn_c_20_1, sim_dyn_c_20_1),
        ("Const 30 Hz, 1mm", exp_dyn_c_30_1, sim_dyn_c_30_1),
        ("Const 10 Hz, 8mm", exp_dyn_c_10_8, sim_dyn_c_10_8),
        ("Const 20 Hz, 8mm", exp_dyn_c_20_8, sim_dyn_c_20_8),
        ("Const 30 Hz, 8mm", exp_dyn_c_30_8, sim_dyn_c_30_8),
    ]
    mean_const, max_const, std_const = build_err(dyn_const_trials, MVC)

    dyn_rand_trials = [
        ("Rand 10 Hz, 1mm", exp_dyn_v_10_1, sim_dyn_v_10_1),
        ("Rand 20 Hz, 1mm", exp_dyn_v_20_1, sim_dyn_v_20_1),
        ("Rand 30 Hz, 1mm", exp_dyn_v_30_1, sim_dyn_v_30_1),
        ("Rand 10 Hz, 8mm", exp_dyn_v_10_8, sim_dyn_v_10_8),
        ("Rand 20 Hz, 8mm", exp_dyn_v_20_8, sim_dyn_v_20_8),
        ("Rand 30 Hz, 8mm", exp_dyn_v_30_8, sim_dyn_v_30_8),
    ]
    mean_rand, max_rand, std_rand = build_err(dyn_rand_trials, MVC)


    def plot_error_panel(ax, x, labels, mean_arr, max_arr, std_arr, xlabel, title, size_labels=11, size_title=11):
        lower = np.maximum(mean_arr - std_arr, 0)
        upper = mean_arr + std_arr

        ax.plot(x, mean_arr, '-o', color='k', label='mAE ± SD', markersize=3.5, linewidth=1)
        ax.fill_between(x, lower, upper, color='k', alpha=0.2)
        ax.plot(x, max_arr, '--*', color='k', label='MAE', markersize=3.5, linewidth=1)

        ax.set_title(title, fontweight='bold', fontsize=size_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_ylim([0, 45])
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=size_labels)
        ax.tick_params(axis='both', labelsize=8)


    def plot_force_panel(ax, time, exp, sim, ylim, label_text=None, sim_noy=None, label_inside=True):
        ax.plot(time, exp, 'k', lw=1, label='Experimental')
        ax.plot(time, sim, 'r', lw=1, label='Simulated')
        if sim_noy is not None:
            ax.plot(time, sim_noy, 'r--', lw=1, label='Simulated no yielding')

        if label_text is not None and label_inside:
            ax.text(
                0.96, 0.90, label_text,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=8,
                fontweight='bold'
            )

        ax.set_ylim(ylim)
        ax.tick_params(axis='both', labelsize=8)


    # ============================================================
    # ISOMETRIC FIGURE
    # ============================================================

    fig = plt.figure(figsize=(7, 8))

    gs = GridSpec(5, 2, height_ratios=[1, 1, 1, 0.12, 0.9], hspace=0.28, wspace=0.25, figure=fig)
    fig.text(0.09, 0.92, "A", fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.09, 0.26, "B", fontsize=14, fontweight="bold", ha="left", va="bottom")

    iso_force_data = [
        (exp_iso_c_10, sim_iso_c_10, "10 Hz"),
        (exp_iso_v_10, sim_iso_v_10, "10 Hz"),
        (exp_iso_c_20, sim_iso_c_20, "20 Hz"),
        (exp_iso_v_20, sim_iso_v_20, "20 Hz"),
        (exp_iso_c_30, sim_iso_c_30, "30 Hz"),
        (exp_iso_v_30, sim_iso_v_30, "30 Hz"),
    ]

    iso_axes = []

    for i, (exp, sim, freq_label) in enumerate(iso_force_data):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c])
        iso_axes.append(ax)

        plot_force_panel(ax=ax, time=time_dt, exp=exp, sim=sim, ylim=(0, 30), label_text=freq_label, label_inside=True)

        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time [s]", fontweight="bold", fontsize=8)

    iso_axes[0].set_title("Constant frequency", fontweight="bold", fontsize=10)
    iso_axes[1].set_title("Random frequency", fontweight="bold", fontsize=10)

    iso_axes[0].legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    fig.text(
        0.06, 0.61,
        "Cat Soleus (Slow) Force [N]",
        va='center',
        rotation='vertical',
        fontweight='bold',
        fontsize=10
    )

    x_iso = np.arange(1, 4)
    iso_labels = ["10", "20", "30"]

    ax_err1 = fig.add_subplot(gs[4, 0])
    ax_err2 = fig.add_subplot(gs[4, 1], sharey=ax_err1)

    ax_err1.set_anchor('N')
    ax_err2.set_anchor('N')

    plot_error_panel(
        ax_err1, x_iso, iso_labels,
        mean_iso_const, max_iso_const, std_iso_const,
        xlabel="Stimulation frequency [Hz]", title="Constant frequency", size_labels=9, size_title=10
    )
    plot_error_panel(
        ax_err2, x_iso, iso_labels,
        mean_iso_rand, max_iso_rand, std_iso_rand,
        xlabel="Stimulation frequency [Hz]", title="Random frequency", size_labels=9, size_title=10
    )

    ax_err1.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    ax_err1.legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    if save_figures == True:
        fig.savefig(figures_path / "slow_M_sub_isometric_summary.png", dpi=300, bbox_inches="tight")

    plt.show()


    # ============================================================
    # DYNAMIC FIGURE
    # ============================================================

    fig = plt.figure(figsize=(14.5, 8.2))
    gs = GridSpec(5, 4, height_ratios=[1, 1, 1, 0.12, 0.9], hspace=0.32, wspace=0.25, figure=fig)
    fig.text(0.099, 0.92, "A", fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.099, 0.26, "C", fontsize=15, fontweight="bold", ha="left", va="bottom")
    fig.text(0.52, 0.92, "B", fontsize=15, fontweight="bold", ha="center", va="top")
    fig.text(0.52, 0.26, "D", fontsize=15, fontweight="bold", ha="center", va="bottom")

    dyn_const_force_data = [
        (exp_dyn_c_10_1, sim_dyn_c_10_1, sim_dyn_c_10_1_noy, "10 Hz, ±1 mm"),
        (exp_dyn_c_10_8, sim_dyn_c_10_8, sim_dyn_c_10_8_noy, "10 Hz, ±8 mm"),
        (exp_dyn_c_20_1, sim_dyn_c_20_1, sim_dyn_c_20_1_noy, "20 Hz, ±1 mm"),
        (exp_dyn_c_20_8, sim_dyn_c_20_8, sim_dyn_c_20_8_noy, "20 Hz, ±8 mm"),
        (exp_dyn_c_30_1, sim_dyn_c_30_1, sim_dyn_c_30_1_noy, "30 Hz, ±1 mm"),
        (exp_dyn_c_30_8, sim_dyn_c_30_8, sim_dyn_c_30_8_noy, "30 Hz, ±8 mm"),
    ]

    dyn_rand_force_data = [
        (exp_dyn_v_10_1, sim_dyn_v_10_1, sim_dyn_v_10_1_noy, "10 Hz, ±1 mm"),
        (exp_dyn_v_10_8, sim_dyn_v_10_8, sim_dyn_v_10_8_noy, "10 Hz, ±8 mm"),
        (exp_dyn_v_20_1, sim_dyn_v_20_1, sim_dyn_v_20_1_noy, "20 Hz, ±1 mm"),
        (exp_dyn_v_20_8, sim_dyn_v_20_8, sim_dyn_v_20_8_noy, "20 Hz, ±8 mm"),
        (exp_dyn_v_30_1, sim_dyn_v_30_1, sim_dyn_v_30_1_noy, "30 Hz, ±1 mm"),
        (exp_dyn_v_30_8, sim_dyn_v_30_8, sim_dyn_v_30_8_noy, "30 Hz, ±8 mm"),
    ]

    dyn_axes = []

    for i, (exp, sim, sim_noy, label_text) in enumerate(dyn_const_force_data):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c])
        dyn_axes.append(ax)

        plot_force_panel(ax=ax, time=time_dt, exp=exp, sim=sim, ylim=(0, 37), label_text=None, sim_noy=sim_noy, label_inside=False)

        ax.set_title(label_text, fontweight='bold', fontsize=9, pad=3)

        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time [s]", fontweight="bold", fontsize=9)

    for i, (exp, sim, sim_noy, label_text) in enumerate(dyn_rand_force_data):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c + 2])
        dyn_axes.append(ax)

        plot_force_panel(ax=ax, time=time_dt, exp=exp, sim=sim, ylim=(0, 37), label_text=None, sim_noy=sim_noy, label_inside=False)

        ax.set_title(label_text, fontweight='bold', fontsize=9, pad=3)

        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time [s]", fontweight="bold", fontsize=9)

    fig.text(
        0.097, 0.61,
        "Cat Soleus (slow) Force [N]",
        va='center',
        rotation='vertical',
        fontweight='bold',
        fontsize=10
    )

    x_dyn = np.arange(1, 7)
    dyn_labels = [
        "10 Hz, ±1 mm", "20 Hz, ±1 mm", "30 Hz, ±1 mm",
        "10 Hz, ±8 mm", "20 Hz, ±8 mm", "30 Hz, ±8 mm"
    ]

    ax_err_c = fig.add_subplot(gs[4, 0:2])
    ax_err_v = fig.add_subplot(gs[4, 2:4], sharey=ax_err_c)

    plot_error_panel(
        ax_err_c, x_dyn, dyn_labels,
        mean_const, max_const, std_const, xlabel="Trial", title="Constant frequency"
    )
    plot_error_panel(
        ax_err_v, x_dyn, dyn_labels,
        mean_rand, max_rand, std_rand, xlabel="Trial", title="Random frequency"
    )

    ax_err_c.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')

    ax_err_c.legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )
    ax_err_v.legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    fig.text(
        0.3, 0.915,
        "Constant frequency",
        ha="center",
        fontweight="bold",
        fontsize=11
    )
    fig.text(
        0.72, 0.915,
        "Random frequency",
        ha="center",
        fontweight="bold",
        fontsize=11
    )

    dyn_axes[0].legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    dyn_axes[6].legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    if save_figures == True:
        fig.savefig(figures_path / "slow_M_sub_dynamic_summary.png", dpi=300, bbox_inches="tight")
    plt.show()

def run_slow_isol():
    
    base_path = Path() / 'benchmark_Results'
    dt = 1e-4

    sim_path = base_path / 'Muscle' / 'slow_isol' / 'sim'
    exp_path = base_path / 'Muscle' / 'slow_isol' / 'exp'

    exp_twitch_0 = np.load(exp_path / 'cat_SOL_1Hz_0mm_force.npy')
    exp_iso_0_10 = np.load(exp_path / 'cat_SOL_10Hz_0mm_force.npy')
    exp_iso_0_20 = np.load(exp_path / 'cat_SOL_20Hz_0mm_force.npy')
    exp_iso_0_40 = np.load(exp_path / 'cat_SOL_40Hz_0mm_force.npy')
    exp_twitch_8 = np.load(exp_path / 'cat_SOL_1Hz_8mm_force.npy')
    exp_iso_8_10 = np.load(exp_path / 'cat_SOL_10Hz_8mm_force.npy')
    exp_iso_8_20 = np.load(exp_path / 'cat_SOL_20Hz_8mm_force.npy')
    exp_iso_8_40 = np.load(exp_path / 'cat_SOL_40Hz_8mm_force.npy')
    exp_twitch_16 = np.load(exp_path / 'cat_SOL_1Hz_16mm_force.npy')
    exp_iso_16_10 = np.load(exp_path / 'cat_SOL_10Hz_16mm_force.npy')
    exp_iso_16_20 = np.load(exp_path / 'cat_SOL_20Hz_16mm_force.npy')
    exp_iso_16_40 = np.load(exp_path / 'cat_SOL_40Hz_16mm_force.npy')

    sim_twitch_0 = np.load(sim_path / 'cat_SOL_1Hz_0mm_force.npy')
    sim_iso_0_10 = np.load(sim_path / 'cat_SOL_10Hz_0mm_force.npy')
    sim_iso_0_20 = np.load(sim_path / 'cat_SOL_20Hz_0mm_force.npy')
    sim_iso_0_40 = np.load(sim_path / 'cat_SOL_40Hz_0mm_force.npy')
    sim_twitch_8 = np.load(sim_path / 'cat_SOL_1Hz_8mm_force.npy')
    sim_iso_8_10 = np.load(sim_path / 'cat_SOL_10Hz_8mm_force.npy')
    sim_iso_8_20 = np.load(sim_path / 'cat_SOL_20Hz_8mm_force.npy')
    sim_iso_8_40 = np.load(sim_path / 'cat_SOL_40Hz_8mm_force.npy')
    sim_twitch_16 = np.load(sim_path / 'cat_SOL_1Hz_16mm_force.npy')
    sim_iso_16_10 = np.load(sim_path / 'cat_SOL_10Hz_16mm_force.npy')
    sim_iso_16_20 = np.load(sim_path / 'cat_SOL_20Hz_16mm_force.npy')
    sim_iso_16_40 = np.load(sim_path / 'cat_SOL_40Hz_16mm_force.npy')

    t_end = 1.4
    time_dt = np.arange(0, t_end, dt)
    MVC = 30.25

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
    # Error panels 
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

    ######################################################################
    # Plots
    #######################################################################

    def plot_len_force_panel(ax, time, exp, sim, ylim, label_text=None):
        ax.plot(time, exp, 'k', lw=1, label='Experimental')
        ax.plot(time, sim, 'r', lw=1, label='Simulated')

        if label_text is not None:
            ax.text(
                0.96, 0.90,
                label_text,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=8,
                fontweight='bold'
            )

        ax.set_ylim(ylim)
        ax.tick_params(axis='both', labelsize=8)

    def plot_len_error_panel(ax, x, labels, mean_arr, max_arr, std_arr, title):
        lower = np.maximum(mean_arr - std_arr, 0)
        upper = mean_arr + std_arr

        ax.plot(x, mean_arr, '-o', color='k', label='mAE ± SD', markersize=3.5, linewidth=1)
        ax.fill_between(x, lower, upper, color='k', alpha=0.2)
        ax.plot(x, max_arr, '--*', color='k', label='MAE', markersize=3.5, linewidth=1)

        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_xlabel('Stimulation frequency [Hz]', fontweight='bold', fontsize=9)
        ax.set_ylim([0, 45])
        ax.tick_params(axis='both', labelsize=8)

    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(6, 3, height_ratios=[1, 1, 1, 1, 0.18, 1], hspace=0.24, wspace=0.25, figure=fig)

    fig.text(0.09, 0.92, "A", fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.09, 0.25, "B", fontsize=14, fontweight="bold", ha="left", va="bottom")

    force_data = [
        [
            (exp_twitch_0,  sim_twitch_0,  (0, 10), "1 Hz"),
            (exp_twitch_8,  sim_twitch_8,  (0, 10), "1 Hz"),
            (exp_twitch_16, sim_twitch_16, (0, 10), "1 Hz"),
        ],
        [
            (exp_iso_0_10,  sim_iso_0_10,  (0, 30), "10 Hz"),
            (exp_iso_8_10,  sim_iso_8_10,  (0, 30), "10 Hz"),
            (exp_iso_16_10, sim_iso_16_10, (0, 30), "10 Hz"),
        ],
        [
            (exp_iso_0_20,  sim_iso_0_20,  (0, 30), "20 Hz"),
            (exp_iso_8_20,  sim_iso_8_20,  (0, 30), "20 Hz"),
            (exp_iso_16_20, sim_iso_16_20, (0, 30), "20 Hz"),
        ],
        [
            (exp_iso_0_40,  sim_iso_0_40,  (0, 35), "40 Hz"),
            (exp_iso_8_40,  sim_iso_8_40,  (0, 35), "40 Hz"),
            (exp_iso_16_40, sim_iso_16_40, (0, 35), "40 Hz"),
        ],
    ]

    col_titles = [
        r"$\boldsymbol{\Delta}\mathbf{L}$ = 0 mm",
        r"$\boldsymbol{\Delta}\mathbf{L}$ = -8 mm",
        r"$\boldsymbol{\Delta}\mathbf{L}$ = -16 mm",
    ]

    force_axes = []

    for r in range(4):
        for c in range(3):
            exp, sim, ylim, freq_label = force_data[r][c]
            ax = fig.add_subplot(gs[r, c])
            force_axes.append(ax)

            plot_len_force_panel(ax, time_dt, exp, sim, ylim, label_text=freq_label)

            if r == 0:
                ax.set_title(col_titles[c], fontweight='bold', fontsize=10, pad=4)

            if r < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time [s]", fontweight="bold", fontsize=9)


    force_axes[1].legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    fig.text(
        0.07, 0.60,
        "Cat Soleus (Slow) Force [N]",
        va='center',
        rotation='vertical',
        fontweight='bold',
        fontsize=10
    )

    x = np.arange(1, 5)
    x_labels = ["1", "10", "20", "40"]

    ax_err_0 = fig.add_subplot(gs[5, 0])
    ax_err_8 = fig.add_subplot(gs[5, 1], sharey=ax_err_0)
    ax_err_16 = fig.add_subplot(gs[5, 2], sharey=ax_err_0)

    plot_len_error_panel(
        ax_err_0, x, x_labels,
        mean_0, max_0, std_0,
        r"$\boldsymbol{\Delta}\mathbf{L}$ = 0 mm"
    )
    plot_len_error_panel(
        ax_err_8, x, x_labels,
        mean_8, max_8, std_8,
        r"$\boldsymbol{\Delta}\mathbf{L}$ = -8 mm"
    )
    plot_len_error_panel(
        ax_err_16, x, x_labels,
        mean_16, max_16, std_16,
        r"$\boldsymbol{\Delta}\mathbf{L}$ = -16 mm"
    )

    ax_err_0.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    ax_err_8.legend(
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    if save_figures == True:
        fig.savefig(figures_path / "slow_M_len_summary.png", dpi=300, bbox_inches="tight")

    plt.show()

def run_MU():

    base_path = Path() / 'benchmark_Results'
    dt = 1e-4

    sim_path_S = base_path / 'MU' / 'MU_S' / 'sim'
    exp_path_S = base_path / 'MU' / 'MU_S' / 'exp'
    sim_path_FR = base_path / 'MU' / 'MU_FR' / 'sim'
    exp_path_FR = base_path / 'MU' / 'MU_FR' / 'exp'

    # -------------------------------------------------------------------------
    # Load experimental traces
    # -------------------------------------------------------------------------
    exp_S_twitch = np.load(exp_path_S / 'cat_LG_1Hz_force.npy')
    exp_S_unfused = np.load(exp_path_S / 'cat_LG_12.5Hz_force.npy')
    exp_S_fused = np.load(exp_path_S / 'cat_LG_40Hz_force.npy')

    exp_F_twitch = np.load(exp_path_FR / 'cat_MG_1Hz_force.npy')
    exp_F_unfused = np.load(exp_path_FR / 'cat_MG_20Hz_force.npy')
    exp_F_fused = np.load(exp_path_FR / 'cat_MG_40Hz_force.npy')

    exp_F_25 = np.load(exp_path_FR / 'rat_MG_25Hz_force.npy')
    exp_F_30 = np.load(exp_path_FR / 'rat_MG_30Hz_force.npy')
    exp_F_35 = np.load(exp_path_FR / 'rat_MG_35Hz_force.npy')
    exp_F_40 = np.load(exp_path_FR / 'rat_MG_40Hz_force.npy')
    exp_F_150 = np.load(exp_path_FR / 'rat_MG_150Hz_force.npy')

    # -------------------------------------------------------------------------
    # Load simulated traces
    # -------------------------------------------------------------------------
    sim_S_twitch = np.load(sim_path_S / 'cat_LG_1Hz_force.npy')
    sim_S_unfused = np.load(sim_path_S / 'cat_LG_12.5Hz_force.npy')
    sim_S_fused = np.load(sim_path_S / 'cat_LG_40Hz_force.npy')

    sim_F_twitch = np.load(sim_path_FR / 'cat_MG_1Hz_force.npy')  # with sag
    sim_F_twitch_nosag = np.load(sim_path_FR / 'cat_MG_1Hz_force.npy') # without sag
    sim_F_unfused = np.load(sim_path_FR / 'cat_MG_20Hz_force.npy')
    sim_F_unfused_nosag = np.load(sim_path_FR / 'cat_MG_20Hz_force_nosag.npy')
    sim_F_fused = np.load(sim_path_FR / 'cat_MG_40Hz_force.npy')

    sim_F_25 = np.load(sim_path_FR / 'rat_MG_25Hz_force.npy')
    sim_F_30 = np.load(sim_path_FR / 'rat_MG_30Hz_force.npy')
    sim_F_35 = np.load(sim_path_FR / 'rat_MG_35Hz_force.npy')
    sim_F_40 = np.load(sim_path_FR / 'rat_MG_40Hz_force.npy')
    sim_F_150 = np.load(sim_path_FR / 'rat_MG_150Hz_force.npy')

    sim_F_25_nosag = np.load(sim_path_FR / 'rat_MG_25Hz_force_nosag.npy')
    sim_F_30_nosag = np.load(sim_path_FR / 'rat_MG_30Hz_force_nosag.npy')
    sim_F_35_nosag = np.load(sim_path_FR / 'rat_MG_35Hz_force_nosag.npy')
    sim_F_40_nosag = np.load(sim_path_FR / 'rat_MG_40Hz_force_nosag.npy')

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
    # Force traces - combined figure
    # -------------------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color='k', lw=2, label='Experimental'),
        Line2D([0], [0], color='r', lw=2, label='Simulated (sag)'),
        Line2D([0], [0], color='r', lw=2, ls='--', label='Simulated (no sag)')
    ]

    fig = plt.figure(figsize=(14, 9))

    gs = GridSpec(
        3, 15,
        height_ratios=[1, 1, 1],
        hspace=0.35,
        wspace=0.35,
        figure=fig
    )

    # Top block: 2 x 3
    axs_top = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axs_top[r, c] = fig.add_subplot(gs[r, c*5:(c+1)*5])

    # Bottom block: 1 x 5
    axs_bot = []
    for c in range(5):
        axs_bot.append(fig.add_subplot(gs[2, c*3:(c+1)*3]))

    # -------------------------------------------------------------------------
    # Row 1: Cat LG (S)
    # -------------------------------------------------------------------------
    axs_top[0, 0].plot(time_dt_S, exp_S_twitch, 'k')
    axs_top[0, 0].plot(time_dt_S, sim_S_twitch, 'r')
    axs_top[0, 0].text(0.8, 0.95, '1 Hz', transform=axs_top[0, 0].transAxes,
                       ha='left', va='top', weight='bold')
    axs_top[0, 0].set_ylabel('Cat LG (Slow) \nMU Force [N]', weight='bold', fontsize=14)
    axs_top[0, 0].set_ylim((0, MVC_S + 0.01))

    axs_top[0, 1].plot(time_dt_S, exp_S_unfused, 'k')
    axs_top[0, 1].plot(time_dt_S, sim_S_unfused, 'r')
    axs_top[0, 1].text(0.8, 0.95, '12.5 Hz', transform=axs_top[0, 1].transAxes,
                       ha='left', va='top', weight='bold')
    axs_top[0, 1].tick_params(axis='y', which='both', labelleft=False)
    axs_top[0, 1].set_ylim((0, MVC_S + 0.01))

    axs_top[0, 2].plot(time_dt_S, exp_S_fused, 'k')
    axs_top[0, 2].plot(time_dt_S, sim_S_fused, 'r')
    axs_top[0, 2].plot(time_dt_S, sim_S_fused, 'r--')
    axs_top[0, 2].text(0.8, 0.95, '40 Hz', transform=axs_top[0, 2].transAxes,
                       ha='left', va='top', weight='bold')
    axs_top[0, 2].tick_params(axis='y', which='both', labelleft=False)
    axs_top[0, 2].set_ylim((0, MVC_S + 0.01))

    axs_top[0, 0].legend(
        handles=legend_handles,
        loc='upper left',
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor='black'
    )

    # -------------------------------------------------------------------------
    # Row 2: Cat MG (F)
    # -------------------------------------------------------------------------
    axs_top[1, 0].plot(time_dt_F1, exp_F_twitch, 'k')
    axs_top[1, 0].plot(time_dt_F1, sim_F_twitch, 'r')
    axs_top[1, 0].text(0.8, 0.95, '1 Hz', transform=axs_top[1, 0].transAxes,
                       ha='left', va='top', weight='bold')
    axs_top[1, 0].set_ylabel('Cat MG (Fast) \nMU Force [N]', weight='bold', fontsize=14)
    axs_top[1, 0].set_ylim((0, MVC_F1 + 0.04))

    axs_top[1, 1].plot(time_dt_F1, exp_F_unfused, 'k')
    axs_top[1, 1].plot(time_dt_F1, sim_F_unfused, 'r')
    axs_top[1, 1].plot(time_dt_F1, sim_F_unfused_nosag, 'r--')
    axs_top[1, 1].text(0.8, 0.95, '25 Hz', transform=axs_top[1, 1].transAxes,
                       ha='left', va='top', weight='bold')
    axs_top[1, 1].set_xlabel('Time [s]', weight='bold', fontsize=14)
    axs_top[1, 1].tick_params(axis='y', which='both', labelleft=False)
    axs_top[1, 1].set_ylim((0, MVC_F1 + 0.04))

    axs_top[1, 2].plot(time_dt_F1, exp_F_fused, 'k')
    axs_top[1, 2].plot(time_dt_F1, sim_F_fused, 'r')
    axs_top[1, 2].text(0.8, 0.95, '40 Hz', transform=axs_top[1, 2].transAxes,
                       ha='left', va='top', weight='bold')
    axs_top[1, 2].tick_params(axis='y', which='both', labelleft=False)
    axs_top[1, 2].set_ylim((0, MVC_F1 + 0.04))

    # -------------------------------------------------------------------------
    # Bottom row: Rat MG (F)
    # -------------------------------------------------------------------------
    bottom_data = [
        (exp_F_25,  sim_F_25,  sim_F_25_nosag,  '25 Hz'),
        (exp_F_30,  sim_F_30,  sim_F_30_nosag,  '30 Hz'),
        (exp_F_35,  sim_F_35,  sim_F_35_nosag,  '35 Hz'),
        (exp_F_40,  sim_F_40,  sim_F_40_nosag,  '40 Hz'),
        (exp_F_150, sim_F_150, None,             '150 Hz'),
    ]

    for i, (exp, sim, sim_nosag, label) in enumerate(bottom_data):
        axs_bot[i].plot(time_dt_F2, exp, 'k')
        axs_bot[i].plot(time_dt_F2, sim, 'r')

        if sim_nosag is not None:
            axs_bot[i].plot(time_dt_F2, sim_nosag, 'r--')

        axs_bot[i].text(0.72, 0.96, label, transform=axs_bot[i].transAxes,
                        ha='left', va='top', weight='bold')
        axs_bot[i].set_ylim((0, MVC_F2 + 0.012))

        if i == 0:
            axs_bot[i].set_ylabel('Rat MG (Fast) \nMU Force [N]', weight='bold', fontsize=14)
        else:
            axs_bot[i].tick_params(axis='y', which='both', labelleft=False)

        if i == 2:
            axs_bot[i].set_xlabel('Time [s]', weight='bold', fontsize=14)

    plt.tight_layout()
    if save_figures == True:
        fig.savefig(figures_path / "MU_summary.png", dpi=300, bbox_inches="tight")

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

def run_fast_iso():

    base_path = Path() / 'benchmark_Results'
    dt = 1e-4

    sim_path_isof = base_path / 'Muscle' / 'fast_isof' / 'sim'
    exp_path_isof = base_path / 'Muscle' / 'fast_isof' / 'exp'
    sim_path_isol = base_path / 'Muscle' / 'fast_isol' / 'sim'
    exp_path_isol = base_path / 'Muscle' / 'fast_isol' / 'exp'

    # -------------------------------------------------------------------------
    # Load FFR twitch (separate dataset)
    # -------------------------------------------------------------------------

    exp_FFR_1 = np.load(exp_path_isof / 'rat_EDL_isof_1Hz_force.npy')
    sim_FFR_1 = np.load(sim_path_isof / 'rat_EDL_isof_1Hz_force.npy')

    # -------------------------------------------------------------------------
    # Load FFR tetani
    # -------------------------------------------------------------------------

    exp_FFR_30  = np.load(exp_path_isof / 'rat_EDL_isof_30Hz_force.npy')
    exp_FFR_50  = np.load(exp_path_isof / 'rat_EDL_isof_50Hz_force.npy')
    exp_FFR_60  = np.load(exp_path_isof / 'rat_EDL_isof_60Hz_force.npy')
    exp_FFR_70  = np.load(exp_path_isof / 'rat_EDL_isof_70Hz_force.npy')
    exp_FFR_80  = np.load(exp_path_isof / 'rat_EDL_isof_80Hz_force.npy')
    exp_FFR_90  = np.load(exp_path_isof / 'rat_EDL_isof_90Hz_force.npy')
    exp_FFR_100 = np.load(exp_path_isof / 'rat_EDL_isof_100Hz_force.npy')
    exp_FFR_120 = np.load(exp_path_isof / 'rat_EDL_isof_120Hz_force.npy')

    sim_FFR_30  = np.load(sim_path_isof / 'rat_EDL_isof_30Hz_force.npy')
    sim_FFR_50  = np.load(sim_path_isof / 'rat_EDL_isof_50Hz_force.npy')
    sim_FFR_60  = np.load(sim_path_isof / 'rat_EDL_isof_60Hz_force.npy')
    sim_FFR_70  = np.load(sim_path_isof / 'rat_EDL_isof_70Hz_force.npy')
    sim_FFR_80  = np.load(sim_path_isof / 'rat_EDL_isof_80Hz_force.npy')
    sim_FFR_90  = np.load(sim_path_isof / 'rat_EDL_isof_90Hz_force.npy')
    sim_FFR_100 = np.load(sim_path_isof / 'rat_EDL_isof_100Hz_force.npy')
    sim_FFR_120 = np.load(sim_path_isof / 'rat_EDL_isof_120Hz_force.npy')

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
    exp_FLR_050 = np.load(exp_path_isol / 'rat_EDL_isol_0.50.npy')
    exp_FLR_100 = np.load(exp_path_isol / 'rat_EDL_isol_1.00.npy')
    exp_FLR_150 = np.load(exp_path_isol / 'rat_EDL_isol_1.50.npy')
    exp_FLR_200 = np.load(exp_path_isol / 'rat_EDL_isol_2.00.npy')
    exp_FLR_250 = np.load(exp_path_isol / 'rat_EDL_isol_2.50.npy')
    exp_FLR_300 = np.load(exp_path_isol / 'rat_EDL_isol_3.00.npy')
    exp_FLR_350 = np.load(exp_path_isol / 'rat_EDL_isol_3.50.npy')
    exp_FLR_400 = np.load(exp_path_isol / 'rat_EDL_isol_4.00.npy')

    sim_FLR_050 = np.load(sim_path_isol / 'rat_EDL_isol_0.50.npy')
    sim_FLR_100 = np.load(sim_path_isol / 'rat_EDL_isol_1.00.npy')
    sim_FLR_150 = np.load(sim_path_isol / 'rat_EDL_isol_1.50.npy')
    sim_FLR_200 = np.load(sim_path_isol / 'rat_EDL_isol_2.00.npy')
    sim_FLR_250 = np.load(sim_path_isol / 'rat_EDL_isol_2.50.npy')
    sim_FLR_300 = np.load(sim_path_isol / 'rat_EDL_isol_3.00.npy')
    sim_FLR_350 = np.load(sim_path_isol / 'rat_EDL_isol_3.50.npy')
    sim_FLR_400 = np.load(sim_path_isol / 'rat_EDL_isol_4.00.npy')

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
    # Error plots 
    # -------------------------------------------------------------------------
    ffr_x = [1] + freqs
    ffr_labels = ['1'] + [str(f) for f in freqs]

    ffr_mean_arr = np.array([twitch_eval["mae"]] + ffr_res["mae"])
    ffr_max_arr  = np.array([twitch_eval["maxae"]] + ffr_res["maxae"])
    ffr_std_arr  = np.array([twitch_eval["stde"]] + ffr_res["std"])

    flr_mean_arr = np.array(flr_res["mae"])
    flr_max_arr  = np.array(flr_res["maxae"])
    flr_std_arr  = np.array(flr_res["std"])

    # -------------------------------------------------------------------------
    # Combined plot: force traces + errors
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 7))

    gs = GridSpec(2, 5, width_ratios=[1, 1.2, 1.2, 0.05, 1.4], hspace=0.6, wspace=0.2, figure=fig)

    ax_twitch   = fig.add_subplot(gs[0, 0])
    ax_ffr_exp  = fig.add_subplot(gs[0, 1])
    ax_ffr_sim  = fig.add_subplot(gs[0, 2])
    ax_ffr_err  = fig.add_subplot(gs[0, 4])

    ax_blank    = fig.add_subplot(gs[1, 0])
    ax_flr_exp  = fig.add_subplot(gs[1, 1])
    ax_flr_sim  = fig.add_subplot(gs[1, 2])
    ax_flr_err  = fig.add_subplot(gs[1, 4])

    # -------------------------------------------------------------------------
    # Panel A: FFR
    # -------------------------------------------------------------------------
    ax_twitch.plot(time_dt_twitch, exp_FFR_1, 'k', label='Experimental')
    ax_twitch.plot(time_dt_twitch, sim_FFR_1, 'r', label='Simulated')
    ax_twitch.set_title("Twitch (1 Hz)", weight='bold', fontsize=10)
    ax_twitch.set_ylabel("Rat EDL (Fast)\nForce [N]", weight='bold')
    ax_twitch.set_xlabel("Time [s]", weight='bold')
    ax_twitch.legend(loc='upper right', fontsize=7)

    n_ffr = len(freqs)
    gray_levels = np.linspace(0.75, 0.15, n_ffr)
    red_levels = np.linspace(0.4, 1.0, n_ffr)

    for i, (f, y) in enumerate(zip(freqs, exp_FFR_series)):
        ax_ffr_exp.plot(time_dt, y, color=str(gray_levels[i]), label=f"{f} Hz")

    ax_ffr_exp.set_title("Experimental tetani", weight='bold', fontsize=10)
    ax_ffr_exp.set_ylim([-0.08, 1.7])
    ax_ffr_exp.set_xlabel("Time [s]", weight='bold')
    ax_ffr_exp.legend(loc='upper right', fontsize=7)

    for i, (f, y) in enumerate(zip(freqs, sim_FFR_series)):
        ax_ffr_sim.plot(time_dt, y, color=(red_levels[i], 0, 0), label=f"{f} Hz")

    ax_ffr_sim.set_title("Simulated tetani", weight='bold', fontsize=10)
    ax_ffr_sim.set_ylim([-0.08, 1.7])
    ax_ffr_sim.set_xlabel("Time [s]", weight='bold')
    ax_ffr_sim.legend(loc='upper right', fontsize=7)

    lower = np.maximum(ffr_mean_arr - ffr_std_arr, 0)
    upper = ffr_mean_arr + ffr_std_arr

    ax_ffr_err.plot(ffr_x, ffr_mean_arr, '-o', color='k', label='mAE ± SD')
    ax_ffr_err.fill_between(ffr_x, lower, upper, color='k', alpha=0.2)
    ax_ffr_err.plot(ffr_x, ffr_max_arr, '--*', color='k', label='MAE')
    ax_ffr_err.set_xticks(ffr_x)
    ax_ffr_err.set_xticklabels(ffr_labels)
    ax_ffr_err.set_xlabel('Stimulation frequency [Hz]', fontweight='bold')
    ax_ffr_err.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    ax_ffr_err.set_ylim([0, 50])
    ax_ffr_err.set_title("Experimental vs. Simulated", weight='bold', fontsize=10)
    ax_ffr_err.legend(loc='upper left', fontsize=7)

    # -------------------------------------------------------------------------
    # Panel B: FLR
    # -------------------------------------------------------------------------
    ax_blank.axis('off')

    n_flr = len(disp_mm)
    gray_levels_flr = np.linspace(0.15, 0.75, n_flr)
    red_levels_flr = np.linspace(0.4, 1.0, n_flr)

    for i, (d, y) in enumerate(zip(disp_mm, exp_FLR_series)):
        ax_flr_exp.plot(time_dt, y, color=str(gray_levels_flr[i]),
                        label=f"$\Delta L$ = + {d:.1f} mm")

    ax_flr_exp.set_title("Experimental tetani", weight='bold', fontsize=10)
    ax_flr_exp.set_ylabel("Rat EDL (Fast)\nForce [N]", weight='bold')
    ax_flr_exp.set_xlabel("Time [s]", weight='bold')
    ax_flr_exp.legend(loc='upper right', fontsize=7)

    for i, (d, y) in enumerate(zip(disp_mm, sim_FLR_series)):
        ax_flr_sim.plot(time_dt, y, color=(red_levels_flr[i], 0, 0),
                        label=f"$\Delta L$ = + {d:.1f} mm")

    ax_flr_sim.set_title("Simulated tetani", weight='bold', fontsize=10)
    ax_flr_sim.set_xlabel("Time [s]", weight='bold')
    ax_flr_sim.legend(loc='upper right', fontsize=7)

    lower_f = np.maximum(flr_mean_arr - flr_std_arr, 0)
    upper_f = flr_mean_arr + flr_std_arr

    ax_flr_err.plot(disp_mm, flr_mean_arr, '-o', color='k', label='mAE ± SD')
    ax_flr_err.fill_between(disp_mm, lower_f, upper_f, color='k', alpha=0.2)
    ax_flr_err.plot(disp_mm, flr_max_arr, '--*', color='k', label='MAE')
    ax_flr_err.set_xlabel(r"$\boldsymbol{\Delta}\mathbf{L}$ [mm]", fontweight='bold')
    ax_flr_err.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    ax_flr_err.set_ylim([0, 40])
    ax_flr_err.set_title("Experimental vs. Simulated", weight='bold', fontsize=10)
    ax_flr_err.legend(loc='upper left', fontsize=7)

    # Panel labels
    fig.text(0.09, 0.95, 'A', fontsize=15, fontweight='bold', ha='left', va='top')
    fig.text(0.26, 0.47, 'B', fontsize=15, fontweight='bold', ha='left', va='top')
    fig.text(0.68, 0.95, 'C', fontsize=15, fontweight='bold', ha='right', va='top')
    fig.text(0.68, 0.47, 'D', fontsize=15, fontweight='bold', ha='right', va='top')

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    if save_figures == True:
        plt.savefig(figures_path / 'fast_M_iso_summary.png', dpi=300, bbox_inches='tight')

    plt.show()

def run_fast_dyn():

    base_path = Path() / 'benchmark_Results'
    dt = 1e-4

    sim_path = base_path / 'Muscle' / 'fast_dyn' / 'sim'
    exp_path = base_path / 'Muscle' / 'fast_dyn' / 'exp'

    exp_dyn_120_095_s = np.load(exp_path / 'cat_CF_120Hz_0.95L0_length_force.npy')
    exp_dyn_120_095_l = np.load(exp_path / 'cat_CF_120Hz_0.95L0_short_force.npy')
    exp_dyn_20_095_s = np.load(exp_path / 'cat_CF_20Hz_0.95L0_length_force.npy')
    exp_dyn_20_095_l = np.load(exp_path / 'cat_CF_20Hz_0.95L0_short_force.npy')
    exp_dyn_40_08_s = np.load(exp_path / 'cat_CF_40Hz_0.8L0_length_force.npy')
    exp_dyn_40_08_l = np.load(exp_path / 'cat_CF_40Hz_0.8L0_short_force.npy')
    exp_dyn_40_11_s = np.load(exp_path / 'cat_CF_40Hz_1.1L0_length_force.npy')
    exp_dyn_40_11_l = np.load(exp_path / 'cat_CF_40Hz_1.1L0_short_force.npy')
    exp_dyn_60_095_s = np.load(exp_path / 'cat_CF_60Hz_0.95L0_length_force.npy')
    exp_dyn_60_095_l = np.load(exp_path / 'cat_CF_60Hz_0.95L0_short_force.npy')

    sim_dyn_120_095_s = np.load(sim_path / 'cat_CF_120Hz_0.95L0_length_force.npy')
    sim_dyn_120_095_l = np.load(sim_path / 'cat_CF_120Hz_0.95L0_short_force.npy')
    sim_dyn_20_095_s = np.load(sim_path / 'cat_CF_20Hz_0.95L0_length_force.npy')
    sim_dyn_20_095_l = np.load(sim_path / 'cat_CF_20Hz_0.95L0_short_force.npy')
    sim_dyn_40_08_s = np.load(sim_path / 'cat_CF_40Hz_0.8L0_length_force.npy')
    sim_dyn_40_08_l = np.load(sim_path / 'cat_CF_40Hz_0.8L0_short_force.npy')
    sim_dyn_40_11_s = np.load(sim_path / 'cat_CF_40Hz_1.1L0_length_force.npy')
    sim_dyn_40_11_l = np.load(sim_path / 'cat_CF_40Hz_1.1L0_short_force.npy')
    sim_dyn_60_095_s = np.load(sim_path / 'cat_CF_60Hz_0.95L0_length_force.npy')
    sim_dyn_60_095_l = np.load(sim_path / 'cat_CF_60Hz_0.95L0_short_force.npy')

    disp_sub_s = np.load(exp_path / 'cat_CF_120Hz_length_disp.npy')
    disp_sub_l = np.load(exp_path / 'cat_CF_120Hz_short_disp.npy')
    disp_max_s = np.load(exp_path / 'cat_CF_subfreq_length_interp_disp.npy')
    disp_max_l = np.load(exp_path / 'cat_CF_subfreq_short_interp_disp.npy')

    t_end_max = 0.16
    t_end_sub = 0.17
    time_dt_max = np.arange(0, t_end_max, dt)
    time_dt_sub = np.arange(0, t_end_sub, dt)
    MVC = 2.49

    # -------------------------------------------------------------------------
    # Plots 
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

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
    plt.plot(time_dt_max, exp_dyn_120_095_l, 'k')
    plt.plot(time_dt_max, exp_dyn_120_095_s, 'gray')
    plt.plot(time_dt_max, sim_dyn_120_095_l, 'r')
    plt.plot(time_dt_max, sim_dyn_120_095_s, 'orange')
    plt.axvline(time_dt_sub[1100], color='gray', linestyle='--', linewidth=1)
    plt.title(r"120 Hz, at 0.95$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
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

    fig.text(0.05, 0.65, 'Cat CF (Fast) Normalized force', va='center', rotation='vertical',
             weight='bold', fontsize=14)

    legend_handles = [
        Line2D([0], [0],color='k', lw=1.5, label='Experimental-Shortening'),
        Line2D([0], [0], color='gray', lw=1.5, label='Experimental-Lengthening'),
        Line2D([0], [0], color='r', lw=1.5, label='Simulated-Shortening'),
        Line2D([0], [0], color='orange', lw=1.5, label='Simulated-Lengthening')
    ]

    fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(0.9, 0.5), fontsize=11)

    if save_figures == True:
        plt.savefig(figures_path / 'fast_M_dynamic_summary.png', dpi=300, bbox_inches='tight')

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

def run_Ca_transients():

    base_path = Path() / 'benchmark_Results'
    dt = 1e-4

    sim_path = base_path / 'Ca_transients' / 'sim'
    exp_path = base_path / 'Ca_transients' / 'exp'

    exp_S_Ca = np.load(exp_path / 'slow_23_100Hz_Ca.npy')
    exp_F_Ca = np.load(exp_path / 'fast_35_125Hz_Ca.npy')

    sim_S_Ca = np.load(sim_path / 'slow_23_100Hz_Ca.npy')
    sim_F_Ca = np.load(sim_path / 'fast_35_125Hz_Ca.npy')

    time_dt = np.arange(0, 1.4, dt)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    # Panel A: slow
    ax[0].plot(time_dt, sim_S_Ca * 1e6, 'g', lw=1.5, label='Simulated')
    ax[0].plot((exp_S_Ca[:, 0] - exp_S_Ca[0, 0]) * 1e-3, exp_S_Ca[:, 1], 'k--', lw=1.2, label='Experimental')

    ax[0].text(0.03, 0.95, 'A', transform=ax[0].transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax[0].set_xlabel('Time [s]', fontweight='bold')
    ax[0].set_ylabel(r'[$\mathbf{Ca^{2+}}$] [$\mathbf{\mu}$M]', fontweight='bold')
    ax[0].set_xlim((0, 0.15))
    ax[0].set_ylim((0, 20))

    # Panel B: fast
    ax[1].plot(time_dt, sim_F_Ca * 1e6, 'g', lw=1.5, label='Simulated')
    ax[1].plot((exp_F_Ca[:, 0] - exp_F_Ca[0, 0]) * 1e-3, exp_F_Ca[:, 1], 'k--', lw=1.2, label='Experimental')

    ax[1].text(0.03, 0.95, 'B', transform=ax[1].transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax[1].set_xlabel('Time [s]', fontweight='bold')
    ax[1].set_xlim((0, 0.13))
    ax[1].tick_params(axis='y', labelleft=False)
    ax[1].legend(loc='upper right')

    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.tick_params(direction='out')
        a.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    plt.tight_layout()

    if save_figures == True:
        fig.savefig(figures_path / "Ca_transients.png", dpi=300, bbox_inches="tight")

    plt.show()


BENCHMARKS = {
    'slow_dyn2': run_slow_dyn2,
    'slow_isof_dyn1': run_slow_isof_dyn1,
    'slow_isol': run_slow_isol,
    'MU': run_MU,
    'fast_iso': run_fast_iso,
    'fast_dyn': run_fast_dyn,
    'Ca_transients': run_Ca_transients
}

def main(argv=None): # default runs fast_iso benchmark, otherwise select from command line
    parser = argparse.ArgumentParser(description="Plot muscle benchmark results from benchmark_Results.")
    parser.add_argument(
        "benchmark",
        nargs="?",
        default="fast_iso",
        choices=['slow_dyn2', 'slow_isof_dyn1', 'slow_isol', 'MU', 'fast_iso', 'fast_dyn', 'Ca_transients', "all"],
        help="Benchmark to run. Default: fast_iso.",
    )
    args = parser.parse_args(argv)

    selected = BENCHMARKS if args.benchmark == "all" else {args.benchmark: BENCHMARKS[args.benchmark]}
    for name, runner in selected.items():
        print(f"\n--- Running {name} ---")
        runner()


if __name__ == "__main__":
    main()
