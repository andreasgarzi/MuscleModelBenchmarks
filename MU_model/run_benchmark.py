"""
Author: Andrea Sgarzi
Email: a.sgarzi@ad.unsw.edu.au
Affiliation: University of New South Wales (UNSW), Graduate School of Biomedical Engineering (GSBE)

This script implements a collection of animal-based experimental benchmarks
used to validate and test a motor-unit (MU) driven muscle model.

The benchmarks include:
- Slow and fast fibre types
- Isometric and dynamic contractions at various frequencies
- Muscle-scale, motor-unit-scale, and calcium-transient-scale simulations

Experimental datasets are derived from the literature and are used to:
- Configure model parameters and inputs (stimulation, length, force)
- Select active model components (tendon, force-length, force-velocity, yielding, sag)
- Compare simulated outputs against experimental force or calcium traces
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from MU_model import MU_model, ModelConfig
from benchmark_trials import BENCHMARK_TRIALS


BASE_PATH = Path("benchmarkData")
#RESULTS_PATH = Path("results")


#______________________________________________________________________________________________
# Helper functions
#______________________________________________________________________________________________

def load_data(file: Path, data_type: str) -> np.ndarray:
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
    return np.loadtxt(file, delimiter="\t", skiprows=skip_rows[data_type]) # load data skipping sepcified n. of rows


def interp_xy(data: np.ndarray, t_new: np.ndarray, kind: str = "cubic") -> np.ndarray:
    return sp.interpolate.interp1d(data[:, 0], data[:, 1], kind=kind)(t_new) # cubic interp


def extract_ratEDL_trial(data_path: Path, data_type: str, trial=None, f: int = 1000, dt: float = 1e-4) -> dict:
    data = load_data(data_path, data_type)

    if data_type == "rat_EDL_isof":
        freq_map = {30: 0, 50: 1, 60: 2, 70: 3, 80: 4, 90: 5, 100: 6, 120: 7} # freq. map
        seg_idx = freq_map[int(trial)]
        start = seg_idx * 20000 # 20000 in between trials
        N = 1000 # segment of interest is 1000 samples long (contains the force)
        seg_slice = slice(start, start + N) 

    elif data_type == "rat_EDL_isol":
        length_map = {
            0.25: 0, 0.5: 1, 0.75: 2, 1.0: 3, 1.25: 4, 1.5: 5,
            1.75: 6, 2.0: 7, 2.25: 8, 2.5: 9, 2.75: 10, 3.0: 11,
            3.25: 12, 3.5: 13, 3.75: 14, 4.0: 15, 4.25: 16,
            4.5: 17, 4.75: 18, 5.0: 19,
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

    return {"spike_times_sec": spike_times_sec, "t_hi": t_interp, "force_hi": force_interp}

def fibre_type(m: str) -> str:  
        if m in {"rat_SOL", "cat_SOL", "cat_LG"}:  # slow muscles
            return "slow"  
        if m in {"rat_EDL", "cat_MG", "cat_CF", "rat_MG"}:  # fast muscles
            return "fast"  

def build_model_config(config: dict) -> ModelConfig:
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

    benchmark = config["benchmark"] # benchmark name (based on paper)
    muscle = config["muscle"] # muscle name ("species_muscle")
    t_end = config.get("t_end", None) # final simulation time
    fs = str(config["freq"]) # stimulation frequency
    dt = 1e-4 # simulation time step
    if t_end != None: time = np.arange(0, t_end, dt) # create time vector if final time is known

    if benchmark == "dyn2" and muscle == "rat_SOL": # SLOW MUSCLE dynamic benchmark (Krylow, Sandercock 1997)

        path = BASE_PATH / config["scale"] / f"{fibre_type(muscle)}_{benchmark}"

        amp = str(config["amplitude_mm"]) # displacement amplitude
        disp = load_data(path / f"{muscle}_disp.dat", "rat_SOL_dyn2_disp") # load displacement
        disp = interp_xy(disp, np.arange(0, t_end + dt, dt), kind="cubic") # interp displacement
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 2.0 # initial musculo-tendon length
        l_MT = l_MT_0 + disp * float(amp) # musculo-tendon length

        force_file = path / f"{muscle}_{amp}mm_force.dat"
        exp_data = load_data(force_file, "rat_SOL_dyn2_force") # load exp force
        exp_force = interp_xy(exp_data, time, kind="quadratic") # interp exp force
        distimes = np.arange(0, t_end, 1/float(fs)) # create discharge times

    elif benchmark == "isof" and muscle == "cat_SOL": # SLOW MUSCLE iso-f benchmark (Perreault et al. 2003)

        path = BASE_PATH / config["scale"] / f"{fibre_type(muscle)}_{benchmark}"

        stim = config["stim"] # stimulation type (constant: "c" vs. random/variable: "v")
        force_file = path / f"{muscle}_{fs}Hz_{stim}_force.dat"
        exp_data = load_data(force_file, "cat_SOL_isof_force") # load exp force
        exp_force = interp_xy(exp_data, time, kind="cubic") # interp exp force
        distimes = np.load(path / f"{muscle}_{fs}Hz_{stim}_stim.npy")  # load discharge times
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 4.0 # initial musculo-tendon length
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length

    elif benchmark == "isol" and muscle == "cat_SOL": # SLOW MUSCLE iso-l benchmark (Perreault et al. 2003, Kim et al. 2015)

        path = BASE_PATH / config["scale"] / f"{fibre_type(muscle)}_{benchmark}"

        length = int(config["length"]) # delta L (to select length offset, based on ref.)
        offset = {0: 8.0, 8: 0.0, 16: -8.0}[length]
        l_MT_0 = (float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 4.0) + offset # initial musculo-tendon length offset
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length
        exp_force = np.load(path / f"{muscle}_{fs}Hz_{length}mm_interp_force.npy") # load exp force
        distimes = np.load(path / f"{muscle}_{fs}Hz_{length}mm_stim.npy") # load discharge times
        if int(fs) == 1: # twitch case
            distimes = np.round(distimes, 3)

    elif benchmark == "dyn1" and muscle == "cat_SOL":

        path = BASE_PATH / config["scale"] / f"{fibre_type(muscle)}_{benchmark}"

        stim = config["stim"] # stimulation type (constant: "c" vs. random/variable: "v")
        d = int(config["displacement_mm"]) # extract discplacement amplitude (to select disp file, based on ref.)
        disp_file = path / f"cat_SOL_{d}mm_disp.dat"
        disp = load_data(disp_file, "cat_SOL_dyn1_disp") # load displacement
        disp = interp_xy(disp, np.arange(0, t_end + dt, dt), kind="cubic") # interp displacement
    
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) - 4.0 # initial musculo-tendon length
        l_MT = l_MT_0 + (disp + 8.0) # musculo-tendon length
        force_file = path / f"{muscle}_{fs}Hz_{stim}_{d}mm_force.dat"
        exp_data = load_data(force_file, "cat_SOL_dyn1_force") # load exp force
        exp_force = interp_xy(exp_data, time, kind="cubic") # interp exp force
        distimes = np.load(path / f"{muscle}_{fs}Hz_{stim}_stim.npy")  # load discharge times

    elif benchmark == "isof" and muscle == "rat_EDL":

        path = BASE_PATH / config["scale"] / f"{fibre_type(muscle)}_{benchmark}"

        if fs == 1:
            part = extract_ratEDL_trial(path / "rat_EDL_isof_1Hz.ddf", "rat_EDL_isof_1Hz") # extract twitch trial (1Hz) with LPF
        else:
            part = extract_ratEDL_trial(path / "rat_EDL_isof.ddf", "rat_EDL_isof", trial=int(fs)) # extract isof trial with specified frequency
        time = part["t_interp"] # load exp time
        exp_force = part["force_interp"] # load exp force
        distimes = part["spike_times_sec"] # load discharge times
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) # initial musculo-tendon length
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length

    elif benchmark == "isol" and muscle == "rat_EDL":

        path = BASE_PATH / config["scale"] / f"{fibre_type(muscle)}_{benchmark}"

        l = float(config["length_mm"]) # length offset (to select length trial, based on ref.)
        part = extract_ratEDL_trial(path / "rat_EDL_isof.ddf", "rat_EDL_isof", trial=l) # extract isof trial with specified frequency
        time = part["t_interp"] # load exp time
        exp_force = part["force_interp"] # load exp force
        distimes = part["spike_times_sec"] # load discharge times
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) + l # initial musculo-tendon length
        l_MT = np.full(len(time) + 1, float(l_MT_0), dtype=float) # musculo-tendon length

    elif benchmark == "dyn" and muscle == "cat_CF":

        path = BASE_PATH / config["scale"] / f"{fibre_type(muscle)}_{benchmark}"

        trial = config["trial"] # shortening ("short") vs. lengthening ("length") trial
        l_M_0_scale = float(config["l_M_0_scale"]) # scale factor for l_M_0 (to select trial, based on ref.)
        force_file = path / f"{muscle}_{fs}Hz_{l_M_0_scale}L0_{trial}_force.npy"
        exp_force = np.load(force_file)[: len(time)] # load exp force
        l_MT_0 = float(config["l_T_slack"]) + float(config["l_M_0"]) * np.cos(float(config["alpha_0"])) # initial musculo-tendon length
        
        if fs == 120:
            disp_file = path / f"{muscle}_{fs}Hz_{trial}_disp.npy" 
            distimes = np.arange(0, t_end, 1.0/fs) # create discharge times
            mvc_sample = 1083 if trial == "short" else 1090 # sample index corresponding to MVC (for normalisation)
        else:
            disp_file = path / f"{muscle}_subfreq_{trial}_interp_disp.npy"
            distimes = np.arange(0, 0.15, 1.0/fs) # create discharge times for sub-freq trials (only first 150ms, to avoid including the second burst in the lengthening trial)
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

            path = BASE_PATH / config["scale"] / "MU_S"
            exp_force = np.load(path / f"{muscle}_{fs}Hz_force.npy")

            if fs == 1:
                distimes = np.round([0], 3)
            elif fs == 12.5:
                distimes = np.arange(0, 17*0.0813, 0.0813) 
            elif fs == 40:
                distimes = np.arange(0, 13*0.025, 0.025)

        elif muscle == "cat_MG" or muscle == "rat_MG":

            path = BASE_PATH / config["scale"] / "MU_FR"
            exp_force = np.load(path / f"{muscle}_{fs}Hz_force.npy")

            if fs == 1 and muscle == "cat_MG":
                distimes = np.round([0],3) # twitch (Burke 1974)
            elif fs == 20 and muscle == "cat_MG":
                distimes = np.arange(0, 17*0.05, 0.05) # 20 Hz (Burke 1974)
            elif fs == 40 and muscle == "cat_MG":
                distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz (Burke 1974)
            else: 
                distimes = np.load(path / f"{muscle}_{fs}Hz_stim.npy") # (Chelichowski 1999)
                time_exp = np.arange(0, exp_force[-1,0], dt) # for rat GM, exp. force was interpolated to time_dt
                exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='linear')(time_exp)
                target_len = len(time)
                if len(exp_force) < target_len:
                    n_pad = target_len - len(exp_force)
                    exp_force = np.concatenate([exp_force, np.zeros(n_pad)])

    elif config["scale"] == "Ca_transients":

        path = BASE_PATH / config["scale"] 
        
        if muscle == "cat_SOL":
            exp_ca = pd.read_csv(path / "Ca_slow_23_100Hz.csv", delimiter=' ').to_numpy()
            l_MT = np.full(len(time), 1, dtype=float) # full MT length array
            T = 1/fs # adjusted from paper
            distimes = np.arange(0, 0.04, T)

        elif muscle == "rat_EDL":
            exp_ca = pd.read_csv(path / "Ca_fast_35_125Hz.csv", delimiter=' ').to_numpy()
            l_MT = np.full(len(time), 1.6, dtype=float) # full MT length array
            T = 1/fs # adjusted from paper
            distimes = np.arange(0, 0.08, T)

    else:
        raise ValueError(f"No loader implemented for Muscle benchmark '{benchmark}' and muscle '{muscle}'.")

    return {
        "time": time,
        "l_MT": np.asarray(l_MT, dtype=float).ravel(),
        "distimes": np.asarray(distimes, dtype=float).ravel(),
        "exp_force": exp_force,
        "exp_ca": exp_ca,
        "output_force": True,
        "normalise_dynamic_force": normalise_dynamic_force,
        "mvc_sample": mvc_sample,
    }


# _____________________________________________________________________
# Running and optimization
# _____________________________________________________________________

def run_case(case: dict) -> dict:
    model = MU_model(case["parameters"], case["states"], case["distimes"], case["model_config"])
    return model.run(output_force=case["output_force"])


def apply_values(case: dict, names: list[str], values: np.ndarray, opt_config: dict) -> dict:
    # Copy enough of the case to avoid carrying parameter changes between objective calls.
    new_case = case.copy()
    new_case["parameters"] = case["parameters"].copy()
    new_case["states"] = case["states"].copy()

    for name, value in zip(names, values):
        value = float(value)
        if name in new_case["parameters"]:
            new_case["parameters"][name] = value
        elif name in new_case["states"]:
            new_case["states"][name] = value
        else:
            raise ValueError(f"Cannot optimise unknown parameter/state: {name}")

    # Special case used in the old script: l_M_0 changes the isometric MT length too.
    if opt_config.get("rebuild_lMT_from_lM0", False):
        l_M_0 = new_case["states"]["l_M_0"]
        l_T_slack = new_case["parameters"]["l_T_slack"]
        alpha_0 = new_case["parameters"]["alpha_0"]
        l_MT_0 = l_T_slack + l_M_0 * np.cos(alpha_0)
        new_case["parameters"]["l_MT"] = np.full(len(new_case["time"]) + 1, float(l_MT_0), dtype=float)

    return new_case


def residual_vector(case: dict, out: dict, opt_config: dict) -> np.ndarray:
    target = opt_config.get("target", "force")

    if target == "calcium":
        exp_data = case["exp_ca"]
        t_exp = (exp_data[:, 0] - exp_data[0, 0]) * 1e-3
        idx = np.isin(np.round(case["time"], 4), np.round(t_exp, 4)).nonzero()[0]
        sim = out["Ca"][idx] * 1e6
        exp = exp_data[:, 1]
        n = min(len(sim), len(exp))
        return sim[:n] - exp[:n]

    sim = out["force"]
    exp = case["exp_force"]

    if target == "force_dynamic_ratio":
        i = int(case["mvc_sample"])
        sim = sim[i:] / sim[i]
        exp = exp[i:]
        n = min(len(sim), len(exp))
        return sim[:n] - exp[:n]

    n = min(len(sim), len(exp))
    return sim[:n] - exp[:n]


def objective(x: np.ndarray, case: dict, opt_config: dict) -> float:
    trial_case = apply_values(case, opt_config["parameters"], x, opt_config)
    out = run_case(trial_case)
    residuals = residual_vector(trial_case, out, opt_config)
    return float(np.sum(residuals ** 2))


def optimise_case(case: dict, maxiter: int | None = None) -> tuple[dict, dict, object]:
    opt_config = case["config"].get("optimization")
    if opt_config is None:
        raise ValueError(f"No optimization block found for trial '{case['name']}'.")

    x0 = np.asarray(opt_config["x0"], dtype=float)
    bounds = opt_config.get("bounds")
    method = opt_config.get("method", "Nelder-Mead")

    options = {"disp": True}
    if maxiter is not None:
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
    )

    opt_case = apply_values(case, opt_config["parameters"], res.x, opt_config)
    out = run_case(opt_case)

    print("\nOptimized parameters:")
    for name, value in zip(opt_config["parameters"], res.x):
        print(f"  {name} = {value:.8g}")
    print(f"Objective = {res.fun:.8g}")

    return opt_case, out, res


# _____________________________________________________________________
# Plot and save
# _____________________________________________________________________

def plot_case(case: dict, out: dict, show: bool = True) -> None:
    scale = case["config"]["scale"]
    time_dt = case["time"]

    if scale in {"M", "MU"}:
        force_sim = out["force"].copy()
        exp_force = case["exp_force"].copy()

        if case["normalise_dynamic_force"]:
            i = case["mvc_sample"]
            force_sim = force_sim / force_sim[i]
            if case["config"].get("frequency") != "120":
                exp_force = exp_force / exp_force[i]

        plt.figure(figsize=(8, 4))
        plt.plot(time_dt, force_sim, label="Simulated Force", linewidth=2)
        plt.plot(time_dt[: len(exp_force)], exp_force[: len(time_dt)], "k", label="Experimental Force", linewidth=1.5)
        plt.ylabel("Force [N]", fontsize=12)
        plt.xlabel("Time [s]", fontsize=12)
        plt.title(case["name"], weight="bold")
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()

    elif scale == "Ca":
        Ca = out["Ca"]
        exp_ca = case["exp_ca"]
        fibre_type = case["config"]["fibre_type"]

        plt.figure(figsize=(5, 3), dpi=300)
        plt.plot(time_dt, Ca * 1e6, label="Simulated")
        plt.plot((exp_ca[:, 0] - exp_ca[0, 0]) * 1e-3, exp_ca[:, 1], "k--", label="Experimental")
        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel(r"[$Ca^{2+}$] [$\mu$M]", fontsize=12)
        plt.title(f"Ca transient - {fibre_type}", weight="bold")
        plt.xlim((0, 0.15))
        plt.ylim((0, 20))
        plt.legend(loc="upper right")
        plt.grid()
        plt.tight_layout()

    if show:
        plt.show()


def save_case(case: dict, out: dict, opt_result=None) -> None:
    name = case["name"]
    RESULTS_PATH.mkdir(exist_ok=True)
    (RESULTS_PATH / "simulated").mkdir(exist_ok=True)
    (RESULTS_PATH / "experimental").mkdir(exist_ok=True)
    (RESULTS_PATH / "figures").mkdir(exist_ok=True)
    (RESULTS_PATH / "optimisation").mkdir(exist_ok=True)

    if case["config"]["scale"] in {"M", "MU"}:
        np.save(RESULTS_PATH / "simulated" / f"{name}_force.npy", out["force"])
        np.save(RESULTS_PATH / "experimental" / f"{name}_force.npy", case["exp_force"])
    else:
        np.save(RESULTS_PATH / "simulated" / f"{name}_Ca.npy", out["Ca"])
        np.save(RESULTS_PATH / "experimental" / f"{name}_Ca.npy", case["exp_ca"])

    if opt_result is not None:
        opt_config = case["config"].get("optimization", {})
        with open(RESULTS_PATH / "optimisation" / f"{name}_optimised.txt", "w", encoding="utf-8") as f:
            f.write(opt_config.get("label", name) + "\n")
            f.write(f"Objective = {opt_result.fun}\n")
            for pname, pval in zip(opt_config.get("parameters", []), opt_result.x):
                f.write(f"{pname} = {pval}\n")

    plot_case(case, out, show=False)
    plt.savefig(RESULTS_PATH / "figures" / f"{name}.png", dpi=300)
    plt.close()


# _____________________________________________________________________
# Command line interface
# _____________________________________________________________________

def main() -> None:
    parser = argparse.ArgumentParser(description="Run or optimise one muscle-model benchmark trial.")
    parser.add_argument("trial", nargs="?", help="Trial name from benchmark_trials.py")
    parser.add_argument("--save", action="store_true", help="Save simulated/experimental arrays and figure.")
    parser.add_argument("--no-plot", action="store_true", help="Run without showing the plot.")
    parser.add_argument("--list", action="store_true", help="List available trials.")
    parser.add_argument("--list-opt", action="store_true", help="List trials that include an optimisation block.")
    parser.add_argument("--optimize", action="store_true", help="Run the optimisation block associated with this trial.")
    parser.add_argument("--maxiter", type=int, default=None, help="Override optimisation maxiter.")
    args = parser.parse_args()

    if args.list:
        print("Available benchmark trials:")
        for key in BENCHMARK_TRIALS:
            print(f"  {key}")
        return

    if args.list_opt:
        print("Trials with optimisation blocks:")
        for key, config in BENCHMARK_TRIALS.items():
            if "optimization" in config:
                label = config["optimization"].get("label", "")
                print(f"  {key}: {label}")
        return

    if args.trial is None:
        raise SystemExit("Please provide a trial name, or use --list / --list-opt.")

    if args.trial not in BENCHMARK_TRIALS:
        raise SystemExit(f"Unknown trial '{args.trial}'. Use --list to see available trials.")

    config = BENCHMARK_TRIALS[args.trial].copy()
    case = build_case(args.trial, config)

    if args.optimize:
        case, out, opt_result = optimise_case(case, maxiter=args.maxiter)
    else:
        out = run_case(case)
        opt_result = None

    if not args.no_plot:
        plot_case(case, out, show=True)

    if args.save or config.get("save", False):
        save_case(case, out, opt_result=opt_result)
        print(f"Saved results for {args.trial} in '{RESULTS_PATH}'.")


if __name__ == "__main__":
    main()