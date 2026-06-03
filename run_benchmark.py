"""
Author: Andrea Sgarzi
Email: a.sgarzi@ad.unsw.edu.au
Affiliation: University of New South Wales (UNSW), Graduate School of Biomedical Engineering (GSBE)

Run, plot, save, or optimise one benchmark trial for the MU-driven muscle model.

The trial-specific information is stored in benchmark_trials.py. This driver mainly:
- loads the appropriate experimental data and stimulation input;
- constructs the model input dictionaries;
- runs the model;
- optionally runs the optimisation block associated with a trial;
- optionally saves simulated/experimental outputs and figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from benchmark_model import ModelConfig, benchmark_musclemodel
from benchmark_trials import all_trials


base_path = Path() / "benchmark_Data" # benchmark input data path

#______________________________________________________________________________________________
# Helper functions
#______________________________________________________________________________________________

def load_data(file: Path, data_type: str) -> np.ndarray:
    """
    Load experimental data from text-based files (.dat/.ddf) applying the correct
    number of header rows to skip depending on the dataset.

    Inputs:
    - file: Path
        Path to the data file.
    - data_type: str
        Identifier of the dataset (used to determine how many rows to skip).

    Outputs:
    - data: np.ndarray
        Loaded data array (typically columns = time, force, etc.).
    """

    skip_rows = {
        "rat_SOL_dyn2_disp": 16,
        "rat_SOL_dyn2_force": 15,        
        "cat_SOL_dyn1_disp": 18,
        "cat_SOL_dyn1_force": 18,
        "cat_SOL_isof_force": 19,
        "rat_EDL_isof": 30,
        "rat_EDL_isof_1Hz": 59,
        "rat_EDL_isol": 99,
        "rat_EDL_dyn": 190,
    }
    return np.loadtxt(file, delimiter="\t", skiprows=skip_rows[data_type]) 


def interp_xy(data: np.ndarray, t_new: np.ndarray, kind: str = "cubic") -> np.ndarray:
    """
    Interpolate experimental data onto a new time vector.

    Inputs:
    - data: np.ndarray
        Input array with columns [time, signal].
    - t_new: np.ndarray
        Target time vector for interpolation.
    - kind: str
        Interpolation type ("linear", "cubic", etc.).

    Outputs:
    - y_interp: np.ndarray
        Interpolated signal aligned to t_new.
    """

    return sp.interpolate.interp1d(data[:, 0], data[:, 1], kind=kind)(t_new) 


def extract_ratEDL_trial(data_path: Path, data_type: str, trial=None, f: int = 1000, dt: float = 1e-4) -> dict:
    """
    Extract and process a specific rat EDL trial from .ddf files.

    The function selects the correct segment (based on frequency or length),
    extracts force and spike data, and interpolates them.

    Inputs:
    - data_path: Path
        Path to the .ddf file.
    - data_type: str
        Type of dataset ("rat_EDL_isof", "rat_EDL_isol", etc.).
    - trial: int or float
        Identifier of the trial (frequency or length depending on dataset).
    - f: int
        Sampling frequency of the experimental data.
    - dt: float
        Desired time resolution for interpolation.

    Outputs:
    - dict containing:
        - "spike_times_sec": np.ndarray
        - "t_interp": np.ndarray
        - "force_interp": np.ndarray
    """

    data = load_data(data_path, data_type)
    
    if data_type == "rat_EDL_isof":
        freq_map = {30: 0, 50: 1, 60: 2, 70: 3, 80: 4, 90: 5, 100: 6, 120: 7} # freq. map
        seg_idx = freq_map[int(trial)]
        start = seg_idx * 20000 # 20000 in between trials
        N = 1000 # segment of interest is 1000 samples long (contains the force)
        seg_slice = slice(start, start + N) 

    elif data_type == "rat_EDL_isol":
        length_map = {
            0.25: 0, 0.5: 1, 0.75: 2, 1: 3, 1.25: 4, 1.5: 5,
            1.75: 6, 2: 7, 2.25: 8, 2.5: 9, 2.75: 10, 3: 11,
            3.25: 12, 3.5: 13, 3.75: 14, 4: 15, 4.25: 16,
            4.5: 17, 4.75: 18, 5: 19,
        } # length map
        seg_idx = length_map[float(trial)]
        start = seg_idx * 16000 + 1000 # 16000 samples in between trials (starting from 1000 samples offset)
        N = 1000 # segment of interest is 1000 samples long (contains the force)
        seg_slice = slice(start, start + N)

    elif data_type == "rat_EDL_isof_1Hz":
        start = 66025 # start samples
        N = 500 # segment is 500 samples long
        seg_slice = slice(start, start + N)

    else:
        raise ValueError(f"Unsupported segment type: {data_type}")

    t = np.arange(N, dtype=float) / f  # exp. time
    force = np.asarray(data[seg_slice, 2], dtype=float)
    spikes = np.asarray(data[seg_slice, 11], dtype=float)

    rising_edges = np.where((spikes[:-1] == 0) & (spikes[1:] == 1))[0] + 1 # discharges when 1 is preceeded by 0
    spike_times_sec = t[rising_edges]

    if data_type == "rat_EDL_isof_1Hz": # LPF for twitch force
        b, a = signal.butter(2, 40, btype="lowpass", fs=f)
        force = signal.filtfilt(b, a, force)

    force = force - force[0] # offset force
    t_interp = np.arange(0.0, t[-1] + 1e-12, dt) # interp time
    force_interp = sp.interpolate.interp1d(t, force, kind="cubic")(t_interp) # interp force
    force_interp[force_interp < 0] = 0 # inferior limit to 0

    return {"spike_times_sec": spike_times_sec, "t_interp": t_interp, "force_interp": force_interp} 


def build_model_config(config: dict) -> ModelConfig: 
    """
    Build ModelConfig object from trial configuration.

    Inputs:
    - config: dict
        Trial configuration dictionary (from benchmark_trials.py).

    Outputs:
    - model_config: ModelConfig
        Configuration of active model components (SE, PE, FL, FV, etc.).
    """

    return ModelConfig(
        use_SE=bool(config.get("use_SE", False)),
        use_PE=bool(config.get("use_PE", False)),
        use_FL=bool(config.get("use_FL", True)), 
        use_FV=bool(config.get("use_FV", False)),
        use_yielding=bool(config.get("use_yielding", False)),
        use_sag=bool(config.get("use_sag", False)),
    )


#___________________________________________________________________________________________________
# Benchmark trial builder
#___________________________________________________________________________________________________

def build_case(name: str, config: dict) -> dict: 
    """
    Build a benchmark case by loading experimental data and constructing
    the required model inputs (time, l_MT, discharge times, etc.).

    The behaviour depends on the benchmark type (isof, isol, dyn, MU, Ca_transients).

    Inputs:
    - name: str
        Name of the trial.
    - config: dict
        Trial configuration from benchmark_trials.py.

    Outputs:
    - case: dict containing:
        - "time": np.ndarray
        - "l_MT": np.ndarray
        - "distimes": np.ndarray
        - "exp_force": np.ndarray (if applicable)
        - "exp_ca": np.ndarray (if applicable)
        - "output_force": bool
        - "normalise_dynamic_force": bool
        - "mvc_sample": int (if applicable)
    """

    scale = config["scale"] # muscle scale ("Muscle", "MU" or "Ca_transients")
    benchmark = config["benchmark"] # benchmark name (based on paper)
    muscle = config["muscle"] # muscle name ("species_muscle")
    t_end = config.get("t_end", None) # final simulation time
    fs = str(config["freq"]) # stimulation frequency
    dt = 1e-4 # simulation time step
    time = None if t_end is None else np.arange(0, float(t_end), dt)# create time vector if final time is known

    if scale != "Ca_transients":
        results_path = Path() / "benchmark_Results" / scale / benchmark
    else:
        results_path = Path() / "benchmark_Results" / benchmark

    exp_force = None
    exp_ca = None
    distimes = None
    l_MT = None
    output_force = scale != "Ca_transients"
    normalise_dynamic_force = False
    mvc_sample = None

    if scale == "Muscle" and benchmark == "slow_dyn2" and muscle == "rat_SOL": # SLOW MUSCLE dynamic benchmark (Krylow, Sandercock 1997)

        path = base_path / scale / benchmark

        amp = config["amplitude_mm"] # displacement amplitude
        disp = load_data(path / f"{muscle}_disp.dat", "rat_SOL_dyn2_disp") # load displacement
        disp = interp_xy(disp, np.arange(0, t_end + dt, dt), kind="cubic") # interp displacement
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 2.0 # initial musculo-tendon length
        l_MT = l_MT_0 + disp * float(amp) # musculo-tendon length

        force_file = path / f"{muscle}_{amp}mm_force.dat"
        exp_data = load_data(force_file, "rat_SOL_dyn2_force") # load exp force
        exp_force = interp_xy(exp_data, time, kind="quadratic") # interp exp force
        distimes = np.arange(0, t_end, 1/float(fs)) # create discharge times

    elif scale == "Muscle" and benchmark == "slow_isof" and muscle == "cat_SOL": # SLOW MUSCLE iso-f benchmark (Perreault et al. 2003)

        path = base_path / scale / benchmark

        stim = config["stim"] # stimulation type (constant: "c" vs. random/variable: "v")
        force_file = path / f"{muscle}_{fs}Hz_{stim}_force.dat"
        exp_data = load_data(force_file, "cat_SOL_isof_force") # load exp force
        exp_force = interp_xy(exp_data, time, kind="cubic") # interp exp force
        distimes = np.load(path / f"{muscle}_{fs}Hz_{stim}_stim.npy")  # load discharge times
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 4.0 # initial musculo-tendon length
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length

    elif scale == "Muscle" and benchmark == "slow_isol" and muscle == "cat_SOL": # SLOW MUSCLE iso-l benchmark (Perreault et al. 2003, Kim et al. 2015)

        path = base_path / scale / benchmark

        length = config["length"] # delta L (to select length offset, based on ref.)
        offset = {0: 8.0, 8: 0.0, 16: -8.0}[int(length)]
        l_MT_0 = (float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 4.0) + offset # initial musculo-tendon length offset
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length
        exp_force = np.load(path / f"{muscle}_{fs}Hz_{length}mm_interp_force.npy") # load exp force
        distimes = np.load(path / f"{muscle}_{fs}Hz_{length}mm_stim.npy") # load discharge times
        if int(fs) == 1: # twitch case
            distimes = np.round(distimes, 3)

    elif scale == "Muscle" and benchmark == "slow_dyn1" and muscle == "cat_SOL": # SLOW MUSCLE dyn1 benchmark (Perreault et al. 2003)

        path = base_path / scale / benchmark

        stim = config["stim"] # stimulation type (constant: "c" vs. random/variable: "v")
        d = config["displacement_mm"] # extract discplacement amplitude (to select disp file, based on ref.)
        disp = load_data(path / f"{muscle}_{d}mm_disp.dat", "cat_SOL_dyn1_disp") # load displacement
        disp = interp_xy(disp, np.arange(0, t_end + dt, dt), kind="cubic") # interp displacement
    
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 4.0 # initial musculo-tendon length
        l_MT = l_MT_0 + (disp + 8.0) # musculo-tendon length
        force_file = path / f"{muscle}_{fs}Hz_{stim}_{d}mm_force.dat"
        exp_data = load_data(force_file, "cat_SOL_dyn1_force") # load exp force
        exp_force = interp_xy(exp_data, time, kind="cubic") # interp exp force
        distimes = np.load(path / f"{muscle}_{fs}Hz_{stim}_stim.npy")  # load discharge times

    elif scale == "Muscle" and benchmark == "fast_isof" and muscle == "rat_EDL":

        path = base_path / scale / benchmark

        if int(fs) == 1:
            part = extract_ratEDL_trial(path / f"{muscle}_isof_1Hz.ddf", "rat_EDL_isof_1Hz", dt=dt) # extract isof trial with 1Hz frequency (twitch)
        else:
            part = extract_ratEDL_trial(path / f"{muscle}_isof.ddf", "rat_EDL_isof", trial=int(fs), dt=dt) # extract isof trial with specified frequency
        time = part["t_interp"] # load exp time
        exp_force = part["force_interp"] # load exp force
        distimes = part["spike_times_sec"] # load discharge times
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) # initial musculo-tendon length
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length

    elif scale == "Muscle" and benchmark == "fast_isol" and muscle == "rat_EDL":

        path = base_path / scale / benchmark

        l = float(config["length_mm"]) # length offset (to select length trial, based on ref.)
        part = extract_ratEDL_trial(path / f"{muscle}_isol.ddf", "rat_EDL_isol", trial=l, dt=dt) # extract isof trial with specified frequency
        time = part["t_interp"] # load exp time
        exp_force = part["force_interp"] # load exp force
        distimes = part["spike_times_sec"] # load discharge times
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) + l # initial musculo-tendon length
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length

    elif scale == "Muscle" and benchmark == "fast_dyn" and muscle == "cat_CF":

        path = base_path / scale / benchmark

        trial = config["trial"] # shortening ("short") vs. lengthening ("length") trial
        l_M_0_scale = config["l_M_0_scale"] # scale factor for l_M_0 (to select trial, based on ref.)
        force_file = path / f"{muscle}_{fs}Hz_{l_M_0_scale}L0_{trial}_force.npy"
        exp_force = np.load(force_file)[: len(time)] # load exp force
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) # initial musculo-tendon length
        
        if fs == "120":
            disp_file = path / f"{muscle}_{fs}Hz_{trial}_disp.npy" 
            distimes = np.arange(0, t_end, 1.0/float(fs)) # create discharge times
            mvc_sample = 1083 if trial == "short" else 1090 # sample index corresponding to MVC (for normalisation)
        else:
            disp_file = path / f"{muscle}_subfreq_{trial}_interp_disp.npy"
            distimes = np.arange(0, 0.15, 1.0/float(fs)) # create discharge times for sub-freq trials (only first 150ms, to avoid including the second burst in the lengthening trial)
            mvc_sample = 1119 # sample index corresponding to MVC (for normalisation) for sub-freq trials (only first 150ms, to avoid including the second burst in the lengthening trial)

        disp = np.load(disp_file) # load displacement (shortening or lengthening)
        l_MT = np.empty(len(time) + 1, dtype=float)
        l_MT[:-1] = l_MT_0 + (disp[: len(time)] * float(config["l_M_opt"])) * np.cos(float(config["alpha_0"])) # musculo-tendon length (scaled by l_M_opt to match the reference trial)
        l_MT[-1] = l_MT[-2] # ensure the last value of l_MT is consistent with the second to last (to avoid issues with the force-velocity relationship in the last time step)
        normalise_dynamic_force = True

    elif config["scale"] == "MU":

        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) # initial musculo-tendon length
        l_MT = np.full(len(time), float(l_MT_0), dtype=float) # musculo-tendon length
        
        if muscle == "cat_LG":

            path = base_path / scale / benchmark
            exp_force = np.load(path / f"{muscle}_{fs}Hz_force.npy")

            if fs == "1":
                distimes = np.round([0], 3)
            elif fs == "12.5":
                distimes = np.arange(0, 17*0.0813, 0.0813) 
            elif fs == "40":
                distimes = np.arange(0, 13*0.025, 0.025)

        elif muscle == "cat_MG" or muscle == "rat_MG":

            path = base_path / scale / benchmark
            #path = base_path / scale / "MU_FF"
            exp_force = np.load(path / f"{muscle}_{fs}Hz_force.npy")

            if fs == "1" and muscle == "cat_MG":
                distimes = np.round([0],3) # twitch (Burke 1974)
            elif fs == "20" and muscle == "cat_MG":
                distimes = np.arange(0, 17*0.05, 0.05) # 20 Hz (Burke 1974)
            elif fs == "40" and muscle == "cat_MG":
                distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz (Burke 1974)
            elif muscle == "rat_MG":
                distimes = np.load(path / f"{muscle}_{fs}Hz_stim.npy") # (Chelichowski 1999)
        
        if muscle != 'rat_MG':

            time_exp = np.arange(0, exp_force[-1,0], dt) # for rat GM, exp. force was interpolated to time_dt
            exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='linear')(time_exp)
            target_len = len(time)
            if len(exp_force) < target_len:
                n_pad = target_len - len(exp_force)
                exp_force = np.concatenate([exp_force, np.zeros(n_pad)])

    elif config["scale"] == "Ca_transients":

        exp_ca = pd.read_csv(base_path / scale / f"{benchmark}.csv", delimiter=' ').to_numpy()
        
        if muscle == "rat_SOL":
            l_MT = np.full(len(time), 1.0, dtype=float) # full MT length array
            distimes = np.arange(0, 0.04, 1/float(fs)) # adjusted from paper
        elif muscle == "rat_EDL":
            l_MT = np.full(len(time), 1.6, dtype=float) # full MT length array
            distimes = np.arange(0, 0.08, 1/float(fs)) # adjusted from paper
    else:
        raise ValueError(f"No loader implemented for Muscle benchmark '{benchmark}' and muscle '{muscle}'.")
    
    if time is None or l_MT is None or distimes is None:
        raise RuntimeError(f"Incomplete case construction for trial '{name}'.")

    parameters = {
        "time": np.asarray(time, dtype=float),
        "dt": dt,
        "muscle": muscle,
        "scale": scale,
        "MVC": float(config["MVC"]),
        "vmax": float(config.get("vmax", 10.0)), # 10 if not specified
        "alpha_0": float(config["alpha_0"]),
        "l_MT": np.asarray(l_MT, dtype=float).ravel(),
        "l_M_opt": float(config["l_M_opt"]),
        "l_T_slack": float(config["l_T_slack"]),
    }

    states = {
        "MUAP_0": 0.0,
        "Ca_0": 0.0,
        "act_0": 1e-9,
        "l_M_0": float(config["l_M_0"]),
        "y_0": 1.0,
        "s_0": 1.0,
    }

    return {
        "name": name,
        "config": config,
        "parameters": parameters,
        "states": states,
        "distimes": np.asarray(distimes, dtype=float).ravel(),
        "model_config": build_model_config(config),
        "time": parameters["time"],
        "exp_force": exp_force,
        "exp_ca": exp_ca,
        "output_force": output_force,
        "normalise_dynamic_force": normalise_dynamic_force,
        "mvc_sample": mvc_sample,
        "results_path": results_path,
    }


# _____________________________________________________________________
# Running and optimization
# _____________________________________________________________________

def run_case(case: dict) -> dict:
    """
    Run the muscle model for a given benchmark case.

    Inputs:
    - case: dict
        Fully constructed case containing parameters, states, and inputs.

    Outputs:
    - out: dict
        Model outputs (force, Ca, activation, etc.).
    """
    model = benchmark_musclemodel(case["parameters"], case["states"], case["distimes"], case["model_config"])
    return model.run(output_force=case["output_force"])


def apply_values(case: dict, names: list[str], values: np.ndarray, opt_config: dict) -> dict:
    """
    Apply new parameter/state values to a case (used during optimisation).

    Inputs:
    - case: dict
        Original case.
    - names: list[str]
        Names of parameters or states to modify.
    - values: np.ndarray
        New values.
    - opt_config: dict
        Optimisation configuration.

    Outputs:
    - new_case: dict
        Updated case with modified parameters/states.
    """

    new_case = case.copy() 
    new_case["parameters"] = case["parameters"].copy() 
    new_case["states"] = case["states"].copy()

    for name, value in zip(names, values):
        value = float(value)
        if name in new_case["states"]:
            new_case["states"][name] = value
        else:
            new_case["parameters"][name] = value

    if opt_config.get("rebuild_par", False) or opt_config.get("rebuild_lMT_from_lM0", False): # if specified in opt_config, rebuild l_MT from l_M_0 (to ensure consistency when l_M_0 is optimised)
        l_M_0 = new_case["states"].get("l_M_0", case["states"]["l_M_0"])
        l_T_slack = new_case["parameters"]["l_T_slack"]
        alpha_0 = new_case["parameters"]["alpha_0"]
        l_MT_0 = l_T_slack + l_M_0 * np.cos(alpha_0)
        n = len(new_case["time"]) + 1 if new_case["model_config"].use_SE else len(new_case["time"])
        new_case["parameters"]["l_MT"] = np.full(n, float(l_MT_0), dtype=float)

    return new_case


def residual_vector(case: dict, out: dict, opt_config: dict) -> np.ndarray:
    """
    Compute residuals between simulated and experimental data.

    Target ref.erences for optimisation can be specified in opt_config["target"] and include:
    - force
    - dynamic force ratio
    - calcium transients

    Inputs:
    - case: dict
    - out: dict
        Model output.
    - opt_config: dict
        Optimisation settings.

    Outputs:
    - residuals: np.ndarray
    """

    target = opt_config.get("target", "force") # target reference for optimization (default: force)

    if target == "calcium":
        exp_data = case["exp_ca"]
        t_exp = (exp_data[:, 0] - exp_data[0, 0]) * 1e-3 # convert exp time from ms to s and align to model time
        idx = np.isin(np.round(case["time"], 4), np.round(t_exp, 4)).nonzero()[0] # find indices in model time that correspond to exp time (rounded to 4 decimals to avoid floating point issues)
        sim = out["Ca"][idx] * 1e6 # convert simulated Ca from M to uM for comparison with exp data
        exp = exp_data[:, 1] # exp Ca in uM
        n = min(len(sim), len(exp)) # ensure same length for residuals
        return sim[:n] - exp[:n] # residuals for calcium transient optimization

    sim = out["force"]
    exp = case["exp_force"]

    if target == "force_dynamic_ratio":
        i = int(case["mvc_sample"]) # sample index corresponding to MVC (for normalisation)
        sim = sim[i:] / sim[i] # normalise simulated force to MVC
        exp = exp[i:]
        n = min(len(sim), len(exp)) 
        return sim[:n] - exp[:n]

    n = min(len(sim), len(exp))
    return sim[:n] - exp[:n]


def objective(x: np.ndarray, case: dict, opt_config: dict) -> float:
    """
    Objective function for optimisation (sum of squared residuals).

    Inputs:
    - x: np.ndarray
        Current parameter values.
    - case: dict
    - opt_config: dict

    Outputs:
    - error: float
        Sum of squared residuals.
    """

    trial_case = apply_values(case, opt_config["parameters"], x, opt_config) # apply current parameter values to case
    out = run_case(trial_case)
    residuals = residual_vector(trial_case, out, opt_config) # compute residuals
    return float(np.sum(residuals ** 2))


def optimise_case(case: dict, maxiter: int | None = None) -> tuple[dict, dict, Any]:
    """
    Run parameter optimisation for a given case.

    Inputs:
    - case: dict
    - maxiter: int or None
        Maximum number of iterations.

    Outputs:
    - opt_case: dict
        Case with optimised parameters.
    - out: dict
        Model output using optimised parameters.
    - res: OptimizeResult
        Optimisation result object.
    """

    opt_config = case["config"].get("optimization") # get optimiziation config from case
    if opt_config is None:
        raise ValueError(f"No optimization block found for trial '{case['name']}'.")

    x0 = np.asarray(opt_config["x0"], dtype=float) # initial parameter values
    bounds = opt_config.get("bounds") # parameter bounds (optional)
    method = opt_config.get("method", "Nelder-Mead") # optimization method (default: Nelder-Mead)

    options = {"disp": True} # display optimization progress
    if maxiter is not None: # set maxiter if provided as argument, otherwise use value from opt_config if available
        options["maxiter"] = int(maxiter)
    elif opt_config.get("maxiter") is not None:
        options["maxiter"] = int(opt_config["maxiter"])

    print("\nOptimisation:", opt_config.get("label", case["name"]))
    print("Parameters:", ", ".join(opt_config["parameters"]))
    
    res = minimize(
        lambda x: objective(x, case, opt_config),
        x0,
        method=method,
        bounds=bounds,
        options=options,
    ) # run optimization

    opt_case = apply_values(case, opt_config["parameters"], res.x, opt_config) # apply optimised values to case
    out = run_case(opt_case) # run model with optimised case to get output for plotting/saving

    print("\nOptimized parameters:")
    for pname, pvalue in zip(opt_config["parameters"], res.x):
        print(f"  {pname} = {pvalue:.8g}")
    print(f"Objective = {res.fun:.8g}")

    return opt_case, out, res


# _____________________________________________________________________
# Plot and save
# _____________________________________________________________________

def plot_case(case: dict, out: dict, show: bool = True):
    """
    Plot simulated vs experimental results.

    Supports:
    - Force (Muscle / MU)
    - Calcium transients

    Inputs:
    - case: dict
    - out: dict
    - show: bool
        Whether to display the plot.
    """

    scale = case["config"]["scale"]
    time = case["time"]

    if scale in {"Muscle", "MU"}:
        force_sim = out["force"].copy()
        exp_force = np.asarray(case["exp_force"], dtype=float).copy()

        if case["normalise_dynamic_force"]:
            i = int(case["mvc_sample"]) # sample index corresponding to MVC (for normalisation)
            force_sim = force_sim / force_sim[i] # normalise simulated force to MVC
            if not np.isclose(float(case["config"]["freq"]), 120.0): # for the 120Hz dynamic trials, exp force is already normalised to MVC, so only normalise if not 120Hz
                exp_force = exp_force / exp_force[i]

        plt.figure(figsize=(8, 4))
        plt.plot(time, force_sim, label="Simulated Force", linewidth=2)
        plt.plot(time, exp_force, "k", label="Experimental Force", linewidth=1.5)
        plt.ylabel("Force [N]", fontsize=12)
        plt.xlabel("Time [s]", fontsize=12)
        plt.title(case["name"], weight="bold")
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()

    elif scale == "Ca_transients":
        exp_ca = case["exp_ca"]
        plt.figure(figsize=(5, 3))
        plt.plot(time, out["Ca"] * 1e6, label="Simulated")
        plt.plot((exp_ca[:, 0] - exp_ca[0, 0]) * 1e-3, exp_ca[:, 1], "k--", label="Experimental")
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel(r"[$Ca^{2+}$] [$\mu$M]", fontsize=12)
        plt.title(case["name"], weight="bold")
        plt.xlim((0, 0.15))
        plt.ylim((0, 20))
        plt.legend(loc="upper right")
        plt.grid()
        plt.tight_layout()

    if show:
        plt.show()


def save_case(case: dict, out: dict, results_path, opt_result=None):
    """
    Save simulation results, experimental data, figures, and optimisation results.

    Outputs are saved in:
    - results/simulated/
    - results/experimental/
    - results/figures/
    - results/optimisation/

    Inputs:
    - case: dict
    - out: dict
    - opt_result: optional optimisation result
    """

    name = case["name"]
    results_path.mkdir(exist_ok=True)
    (results_path / "sim").mkdir(exist_ok=True)
    (results_path / "exp").mkdir(exist_ok=True)
    (results_path / "optimisation").mkdir(exist_ok=True)

    if case["config"]["scale"] in {"Muscle", "MU"}:
        np.save(results_path / "sim" / f"{name}_force.npy", out["force"])
        np.save(results_path / "exp" / f"{name}_force.npy", case["exp_force"])
    else:
        np.save(results_path / "sim" / f"{name}.npy", out["Ca"])
        np.save(results_path / "exp" / f"{name}.npy", case["exp_ca"])

    if opt_result is not None: # if optimization was performed, save the optimized parameter values and objective value in a text file
        opt_config = case["config"].get("optimization", {}) # get optimization config for label and parameter names
        with open(results_path / "optimisation" / f"{name}_optimised.txt", "w", encoding="utf-8") as f:
            f.write(opt_config.get("label", name) + "\n")
            f.write(f"Objective = {opt_result.fun}\n")
            for pname, pval in zip(opt_config.get("parameters", []), opt_result.x):
                f.write(f"{pname} = {pval}\n")

    plot_case(case, out, show=False)
    # plt.savefig(results_path / "figures" / f"{name}.png", dpi=500)
    plt.close()


# _____________________________________________________________________
# Command line interface
# _____________________________________________________________________

def main():
    """
    Command-line interface for running and optimising benchmark trials.

    Available options:
    - --list: list all trials
    - --list-opt: list trials with optimisation
    - --optimize: run optimisation
    - --save: save outputs
    - --no-plot: disable plotting
    """

    parser = argparse.ArgumentParser(description="Run or optimise one muscle-model benchmark trial.") 
    parser.add_argument("trial", nargs="?", help="Trial name from benchmark_trials.py")
    parser.add_argument("--save", action="store_true", help="Save simulated/experimental arrays and figure.")
    parser.add_argument("--no-plot", action="store_true", help="Run without showing the plot.")
    parser.add_argument("--list", action="store_true", help="List available trials.")
    parser.add_argument("--list-opt", action="store_true", help="List trials that include an optimisation block.")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark filter for --list, e.g. isof, isol, dyn, dyn1, dyn2.")
    parser.add_argument("--scale", default=None, help="Optional scale filter for --list, e.g. Muscle, MU, Ca_transients.")
    parser.add_argument("--optimize", action="store_true", help="Run the optimisation block associated with this trial.")
    parser.add_argument("--maxiter", type=int, default=None, help="Override optimisation maxiter.")
    args = parser.parse_args()

    if args.list:
        print("Available benchmark trials:")
        for key, cfg in all_trials.items():
            if args.benchmark is not None and cfg.get("benchmark") != args.benchmark:
                continue
            if args.scale is not None and cfg.get("scale") != args.scale:
                continue
            print(f"  {key}")
        return

    if args.list_opt:
        print("Trials with optimisation blocks:")
        for key, cfg in all_trials.items():
            if "optimization" in cfg:
                label = cfg["optimization"].get("label", "")
                print(f"  {key}: {label}")
        return

    if args.trial is None:
        raise SystemExit("Please provide a trial name, or use --list / --list-opt.")

    if args.trial not in all_trials:
        raise SystemExit(f"Unknown trial '{args.trial}'. Use --list to see available trials.")

    config = all_trials[args.trial].copy()
    case = build_case(args.trial, config) # build benchmark trial case

    if args.optimize:
        case, out, opt_result = optimise_case(case, maxiter=args.maxiter) # optimize parameters in trial case
    else:
        out = run_case(case) # otherwise just run the simulation
        opt_result = None

    if not args.no_plot:
        plot_case(case, out, show=True) # plot results

    if args.save or config.get("save", False):
        save_case(case, out, case['results_path'], opt_result=opt_result) # save in results folder
        print(f"Saved results for {args.trial} in '{case['results_path']}'.")


if __name__ == "__main__":
    main()