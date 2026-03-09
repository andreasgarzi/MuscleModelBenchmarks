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
import matplotlib.pyplot as plt

# Helper function to summarize errors across trials 
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



#############################################################################
" PLOT MU_model benchmark results and COMPUTE ERRORS & STATISTICS"
" Files were kept separated to avoid confusion in error and statistics computation"
#############################################################################

benchmark = 'MU' # specify benchmark among [MU, slow_M_max, slow_M_sub, slow_M_len, fast_M_iso, fast_M_dyn]
base_path = Path('..') / 'Results_benchmarks'
dt = 1e-4

if benchmark == 'slow_M_max': # Krylow & Sandercock 1997 experiments

    sim_path = base_path / 'slow_M' / 'sim' # experimental path
    exp_path = base_path / 'slow_M' / 'exp' # simulations path

    exp1 = np.load(exp_path / 'max_0.npy') # load experimental forces
    exp2 = np.load(exp_path / 'max_1.npy')
    exp3 = np.load(exp_path / 'max_2.npy')
    exp4 = np.load(exp_path / 'max_3.npy')
    exp5 = np.load(exp_path / 'max_4.npy')
    exp6 = np.load(exp_path / 'max_5.npy')

    sim1 = np.load(sim_path / 'max_0.npy') # load simulated forces
    sim2 = np.load(sim_path / 'max_1.npy')
    sim3 = np.load(sim_path / 'max_2.npy')
    sim4 = np.load(sim_path / 'max_3.npy')
    sim5 = np.load(sim_path / 'max_4.npy')
    sim6 = np.load(sim_path / 'max_5.npy')

    t_end = 2 # total seconds
    time_dt = np.arange(0, t_end, dt) # time for BB
    MVC = 1.32

    fig = plt.figure(figsize=(7, 8))

    plt.subplot(6, 1, 1)
    plt.plot(time_dt, exp1, 'k', label='Experimental')
    plt.plot(time_dt, sim1, 'r', label='Simulated')
    plt.title(u"\u00B1 0.05 mm", x=0.1, y=0.97, weight='bold')
    plt.legend(loc=(0.8, 1.1), fontsize=12)
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

    fig.text(0.01, 0.5, 'Force [N]', va='center', rotation='vertical', 
         weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    # Errors
    exp_all = [exp1, exp2, exp3, exp4, exp5, exp6]
    sim_all = [sim1, sim2, sim3, sim4, sim5, sim6]
    displacements = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]

    mean_err_list = []
    max_err_list = []
    std_err_list = []

    print("\n=== Benchmark: slow_M_max ===")
    print(f"MVC = {MVC:.2f} N\n")

    for exp, sim, disp in zip(exp_all, sim_all, displacements):

        # maschera: prendi solo punti dove almeno uno è ≠ 0
        mask = (exp != 0) | (sim != 0)

        if np.any(mask):  # evita errori se tutti 0
            abs_err = np.abs((sim[mask] / MVC) - (exp[mask] / MVC)) * 100.0

            mae  = np.mean(abs_err)
            maxae = np.max(abs_err)
            stde = np.std(abs_err)
        else:
            mae = maxae = stde = 0.0

        mean_err_list.append(mae)
        max_err_list.append(maxae)
        std_err_list.append(stde)

        print(f"Displacement ±{disp:.2f} mm:")
        print(f"  mAE = {mae:.2f}% MVC   (std = {stde:.2f})")
        print(f"  MAE = {maxae:.2f}% MVC\n")

    mmae  = np.mean(mean_err_list)
    mstde = np.std(mean_err_list)
    print(f"  Mean mAE = {mmae:.2f}% MVC   (std = {mstde:.2f})")

    summarize_trials(mean_err_list, max_err_list,
                 label="— slow_M_max summary (all displacements) —",
                 unit="% MVC")

    x = np.arange(1, len(displacements) + 1)

    mean_err_arr = np.array(mean_err_list)
    max_err_arr  = np.array(max_err_list)
    std_err_arr  = np.array(std_err_list)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # mean + std area
    ax.plot(x, mean_err_arr, '-o', color='k', label='mAE ± SD')
    ax.fill_between(
        x,
        mean_err_arr - std_err_arr,
        mean_err_arr + std_err_arr,
        color='k',
        alpha=0.2
    )

    # max
    ax.plot(x, max_err_arr, '--*', color='k', label='MAE')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.2f}' for d in displacements])
    ax.set_xlabel('Max. length variation amplitude [mm]', fontweight='bold')
    ax.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.tight_layout()
    plt.show()


elif benchmark == 'slow_M_sub': # Perreault 2003 experiments
    
    sim_path = base_path / 'slow_M' / 'sim' # experimental path
    exp_path = base_path / 'slow_M' / 'exp' # simulations path

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

    t_end = 2 # total seconds
    time_dt = np.arange(0, t_end, dt) # time for BB
    MVC = 26.13

    fig = plt.figure(figsize=(10, 9))

    plt.subplot(3, 2, 1)
    plt.plot(time_dt, exp_iso_c_10, 'k')
    plt.plot(time_dt, sim_iso_c_10, 'r')
    plt.title(u"Constant 10 Hz", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 2)
    plt.plot(time_dt, exp_iso_v_10, 'k', label='Experimental')
    plt.plot(time_dt, sim_iso_v_10, 'r', label='Simulated')
    plt.title(u"Random 10 Hz", x=0.16, y=0.99, weight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 3)
    plt.plot(time_dt, exp_iso_c_20, 'k')
    plt.plot(time_dt, sim_iso_c_20, 'r')
    plt.title(u"Constant 20 Hz", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel('Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 4)
    plt.plot(time_dt, exp_iso_v_20, 'k')
    plt.plot(time_dt, sim_iso_v_20, 'r')
    plt.title(u"Random 20 Hz", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 5)
    plt.plot(time_dt, exp_iso_c_30, 'k')
    plt.plot(time_dt, sim_iso_c_30, 'r')
    plt.title(u"Constant 30 Hz", x=0.16, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 30))

    plt.subplot(3, 2, 6)
    plt.plot(time_dt, exp_iso_v_30, 'k')
    plt.plot(time_dt, sim_iso_v_30, 'r')
    plt.title(u"Random 30 Hz", x=0.16, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 30))

    plt.tight_layout()
    plt.show()

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

    fig = plt.figure(figsize=(10, 9))
    
    plt.subplot(3, 2, 1)
    plt.plot(time_dt, exp_dyn_c_10_1, 'k')
    plt.plot(time_dt, sim_dyn_c_10_1, 'r')
    plt.plot(time_dt, sim_dyn_c_10_1_noy, 'r--')
    plt.title(u"Constant 10 Hz, \u00B1 1 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 2)
    plt.plot(time_dt, exp_dyn_c_10_8, 'k', label='Experimental')
    plt.plot(time_dt, sim_dyn_c_10_8, 'r', label='Simulated (yielding)')
    plt.plot(time_dt, sim_dyn_c_10_8_noy, 'r--', label='Simulated (no yielding)')
    plt.legend(loc='upper right', fontsize=12)
    plt.title(u"Constant 10 Hz, \u00B1 8 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 3)
    plt.plot(time_dt, exp_dyn_c_20_1, 'k')
    plt.plot(time_dt, sim_dyn_c_20_1, 'r')
    plt.plot(time_dt, sim_dyn_c_20_1_noy, 'r--')
    plt.title(u"Constant 20 Hz, \u00B1 1 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel('Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 4)
    plt.plot(time_dt, exp_dyn_c_20_8, 'k')
    plt.plot(time_dt, sim_dyn_c_20_8, 'r')
    plt.plot(time_dt, sim_dyn_c_20_8_noy, 'r--')
    plt.title(u"Constant 20 Hz, \u00B1 8 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 5)
    plt.plot(time_dt, exp_dyn_c_30_1, 'k')
    plt.plot(time_dt, sim_dyn_c_30_1, 'r')
    plt.plot(time_dt, sim_dyn_c_30_1_noy, 'r--')
    plt.title(u"Constant 30 Hz, \u00B1 1 mm", x=0.16, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 6)
    plt.plot(time_dt, exp_dyn_c_30_8, 'k')
    plt.plot(time_dt, sim_dyn_c_30_8, 'r')
    plt.plot(time_dt, sim_dyn_c_30_8_noy, 'r--')
    plt.title(u"Constant 30 Hz, \u00B1 8 mm", x=0.16, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.tight_layout()
    plt.show()

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

    fig = plt.figure(figsize=(10, 9))

    plt.subplot(3, 2, 1)
    plt.plot(time_dt, exp_dyn_v_10_1, 'k')
    plt.plot(time_dt, sim_dyn_v_10_1, 'r')
    #plt.plot(time_dt, sim_dyn_v_10_1_noy, 'r--')
    plt.title(u"Random 10 Hz, \u00B1 1 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 2)
    plt.plot(time_dt, exp_dyn_v_10_8, 'k', label='Experimental')
    plt.plot(time_dt, sim_dyn_v_10_8, 'r', label='Simulated')
    #plt.plot(time_dt, sim_dyn_v_10_8_noy, 'r--', label='Simulated (no yielding)')
    plt.legend(loc='upper right', fontsize=12)
    plt.title(u"Random 10 Hz, \u00B1 8 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 3)
    plt.plot(time_dt, exp_dyn_v_20_1, 'k')
    plt.plot(time_dt, sim_dyn_v_20_1, 'r')
    #plt.plot(time_dt, sim_dyn_v_20_1_noy, 'r--')
    plt.title(u"Random 20 Hz, \u00B1 1 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylabel('Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 4)
    plt.plot(time_dt, exp_dyn_v_20_8, 'k')
    plt.plot(time_dt, sim_dyn_v_20_8, 'r')
    #plt.plot(time_dt, sim_dyn_v_20_8_noy, 'r--')
    plt.title(u"Random 20 Hz, \u00B1 8 mm", x=0.16, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 5)
    plt.plot(time_dt, exp_dyn_v_30_1, 'k')
    plt.plot(time_dt, sim_dyn_v_30_1, 'r')
    #plt.plot(time_dt, sim_dyn_v_30_1_noy, 'r--')
    plt.title(u"Random 30 Hz, \u00B1 1 mm", x=0.16, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.subplot(3, 2, 6)
    plt.plot(time_dt, exp_dyn_v_30_8, 'k')
    plt.plot(time_dt, sim_dyn_v_30_8, 'r')
    #plt.plot(time_dt, sim_dyn_v_30_8_noy, 'r--')
    plt.title(u"Random 30 Hz, \u00B1 8 mm", x=0.16, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.ylim((0, 37))

    plt.tight_layout()
    plt.show()

    # =========================
    # Errors
    # =========================
    def pct_errors_full(exp: np.ndarray, sim: np.ndarray, MVC: float):

        mask = (exp != 0) | (sim != 0)
        if not np.any(mask):
           return 0.0, 0.0, 0.0

        exp_nz = exp[mask]
        sim_nz = sim[mask]

        abs_err_pct = np.abs((sim_nz / MVC) - (exp_nz / MVC)) * 100.0
        mae   = float(np.mean(abs_err_pct))
        maxae = float(np.max(abs_err_pct))
        stde  = float(np.std(abs_err_pct))
        return mae, maxae, stde

    print("\n=== Benchmark: slow_M_sub ===")
    print(f"MVC = {MVC:.2f} N\n")

    # -------- ISO (isometric: constant/random 10,20,30 Hz) --------
    iso_trials = [
        ("ISO Const 10 Hz", exp_iso_c_10, sim_iso_c_10),
        ("ISO Rand  10 Hz", exp_iso_v_10, sim_iso_v_10),
        ("ISO Const 20 Hz", exp_iso_c_20, sim_iso_c_20),
        ("ISO Rand  20 Hz", exp_iso_v_20, sim_iso_v_20),
        ("ISO Const 30 Hz", exp_iso_c_30, sim_iso_c_30),
        ("ISO Rand  30 Hz", exp_iso_v_30, sim_iso_v_30),
    ]

    iso_mae, iso_max, iso_std = [], [], []
    print("— ISO trials —")
    for name, exp, sim in iso_trials:
        mae, maxae, stde = pct_errors_full(exp, sim, MVC)
        iso_mae.append(mae); iso_max.append(maxae); iso_std.append(stde)
        print(f"{name:>16}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")

    print(f"{'ISO Average':>16}:  MAE = {np.mean(iso_mae):6.2f}% MVC  (std = {np.mean(iso_std):6.2f})   MaxAE = {np.mean(iso_max):6.2f}% MVC\n")

    # -------- DYN (constant displacement: 1 mm, 8 mm @ 10/20/30 Hz) --------
    dyn_c_trials = [
        ("DYN Const 10 Hz ±1 mm", exp_dyn_c_10_1, sim_dyn_c_10_1),
        ("DYN Const 10 Hz ±8 mm", exp_dyn_c_10_8, sim_dyn_c_10_8),
        ("DYN Const 20 Hz ±1 mm", exp_dyn_c_20_1, sim_dyn_c_20_1),
        ("DYN Const 20 Hz ±8 mm", exp_dyn_c_20_8, sim_dyn_c_20_8),
        ("DYN Const 30 Hz ±1 mm", exp_dyn_c_30_1, sim_dyn_c_30_1),
        ("DYN Const 30 Hz ±8 mm", exp_dyn_c_30_8, sim_dyn_c_30_8),
    ]

    dync_mae, dync_max, dync_std = [], [], []
    print("— Dynamic (constant) trials —")
    for name, exp, sim in dyn_c_trials:
        mae, maxae, stde = pct_errors_full(exp, sim, MVC)
        dync_mae.append(mae); dync_max.append(maxae); dync_std.append(stde)
        print(f"{name:>24}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")

    print(f"{'DYN Const Average':>24}:  MAE = {np.mean(dync_mae):6.2f}% MVC  (std = {np.mean(dync_std):6.2f})   MaxAE = {np.mean(dync_max):6.2f}% MVC\n")

    # -------- DYN (random/variable displacement) --------
    dyn_v_trials = [
        ("DYN Rand 10 Hz ±1 mm", exp_dyn_v_10_1, sim_dyn_v_10_1),
        ("DYN Rand 10 Hz ±8 mm", exp_dyn_v_10_8, sim_dyn_v_10_8),
        ("DYN Rand 20 Hz ±1 mm", exp_dyn_v_20_1, sim_dyn_v_20_1),
        ("DYN Rand 20 Hz ±8 mm", exp_dyn_v_20_8, sim_dyn_v_20_8),
        ("DYN Rand 30 Hz ±1 mm", exp_dyn_v_30_1, sim_dyn_v_30_1),
        ("DYN Rand 30 Hz ±8 mm", exp_dyn_v_30_8, sim_dyn_v_30_8),
    ]

    dynv_mae, dynv_max, dynv_std = [], [], []
    print("— Dynamic (random) trials —")
    for name, exp, sim in dyn_v_trials:
        mae, maxae, stde = pct_errors_full(exp, sim, MVC)
        dynv_mae.append(mae); dynv_max.append(maxae); dynv_std.append(stde)
        print(f"{name:>23}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")

    print(f"{'DYN Rand Average':>23}:  MAE = {np.mean(dynv_mae):6.2f}% MVC  (std = {np.mean(dynv_std):6.2f})   MaxAE = {np.mean(dynv_max):6.2f}% MVC\n")


    # -------- DYN (constant) — NO YIELDING --------
    dyn_c_trials_noy = [
        ("DYN Const 10 Hz ±1 mm (NOY)", exp_dyn_c_10_1, sim_dyn_c_10_1_noy),
        ("DYN Const 10 Hz ±8 mm (NOY)", exp_dyn_c_10_8, sim_dyn_c_10_8_noy),
        ("DYN Const 20 Hz ±1 mm (NOY)", exp_dyn_c_20_1, sim_dyn_c_20_1_noy),
        ("DYN Const 20 Hz ±8 mm (NOY)", exp_dyn_c_20_8, sim_dyn_c_20_8_noy),
        ("DYN Const 30 Hz ±1 mm (NOY)", exp_dyn_c_30_1, sim_dyn_c_30_1_noy),
        ("DYN Const 30 Hz ±8 mm (NOY)", exp_dyn_c_30_8, sim_dyn_c_30_8_noy),
    ]

    dync_noy_mae, dync_noy_max, dync_noy_std = [], [], []
    print("— Dynamic (constant) trials — NO YIELDING —")
    for name, exp, sim_noy in dyn_c_trials_noy:
        mae, maxae, stde = pct_errors_full(exp, sim_noy, MVC)
        dync_noy_mae.append(mae); dync_noy_max.append(maxae); dync_noy_std.append(stde)
        print(f"{name:>34}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")
    print(f"{'DYN Const Average (NOY)':>34}:  MAE = {np.mean(dync_noy_mae):6.2f}% MVC  "
          f"(std = {np.mean(dync_noy_std):6.2f})   MaxAE = {np.mean(dync_noy_max):6.2f}% MVC\n")

    # -------- DYN (random/variable) — NO YIELDING --------
    dyn_v_trials_noy = [
        ("DYN Rand 10 Hz ±1 mm (NOY)", exp_dyn_v_10_1, sim_dyn_v_10_1_noy),
        ("DYN Rand 10 Hz ±8 mm (NOY)", exp_dyn_v_10_8, sim_dyn_v_10_8_noy),
        ("DYN Rand 20 Hz ±1 mm (NOY)", exp_dyn_v_20_1, sim_dyn_v_20_1_noy),
        ("DYN Rand 20 Hz ±8 mm (NOY)", exp_dyn_v_20_8, sim_dyn_v_20_8_noy),
        ("DYN Rand 30 Hz ±1 mm (NOY)", exp_dyn_v_30_1, sim_dyn_v_30_1_noy),
        ("DYN Rand 30 Hz ±8 mm (NOY)", exp_dyn_v_30_8, sim_dyn_v_30_8_noy),
    ]

    dynv_noy_mae, dynv_noy_max, dynv_noy_std = [], [], []
    print("— Dynamic (random) trials — NO YIELDING —")
    for name, exp, sim_noy in dyn_v_trials_noy:
        mae, maxae, stde = pct_errors_full(exp, sim_noy, MVC)
        dynv_noy_mae.append(mae); dynv_noy_max.append(maxae); dynv_noy_std.append(stde)
        print(f"{name:>33}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")
    print(f"{'DYN Rand Average (NOY)':>33}:  MAE = {np.mean(dynv_noy_mae):6.2f}% MVC  "
          f"(std = {np.mean(dynv_noy_std):6.2f})   MaxAE = {np.mean(dynv_noy_max):6.2f}% MVC\n")

    summarize_trials(iso_mae,  iso_max,  label="— slow_M_sub summary: ISO (all trials) —", unit="% MVC")
    all_mae_dyn = list(dync_mae)  + list(dynv_mae)
    all_max_dyn = list(dync_max) + list(dynv_max)
    summarize_trials(all_mae_dyn, all_max_dyn, label="— slow_M_sub summary: DYN (all trials) —", unit="% MVC")

    all_mae_noy_dyn = list(dync_noy_mae)  + list(dynv_noy_mae)
    all_max_noy_dyn = list(dync_noy_max) + list(dynv_noy_max)
    summarize_trials(all_mae_noy_dyn, all_max_noy_dyn, label="— slow_M_sub summary: DYN NO YIELDING (all trials) —", unit="% MVC")

    # ---------------------------
    # helper per i plot
    # ---------------------------
    def build_err(trials, MVC):
        mean_list, max_list, std_list = [], [], []
        for _, exp, sim in trials:
            # filtro anche qui
            mask = (exp != 0) | (sim != 0)
            if np.any(mask):
                abs_err = np.abs((sim[mask] / MVC) - (exp[mask] / MVC)) * 100.0
                mean_list.append(np.mean(abs_err))
                max_list.append(np.max(abs_err))
                std_list.append(np.std(abs_err))
            else:
                mean_list.append(0.0)
                max_list.append(0.0)
                std_list.append(0.0)
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
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim([0, 45])

    #plot_panel(axes[0], x, mean_iso,  max_iso,  std_iso,  "Isometric")
    plot_panel(axes[0], x, mean_1mm,  max_1mm,  std_1mm,  "Dynamic ±1 mm")
    plot_panel(axes[1], x, mean_8mm,  max_8mm,  std_8mm,  "Dynamic ±8 mm")

    axes[0].set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Stimulation', fontweight='bold')
    axes[0].legend()

    plt.tight_layout()
    plt.show()


elif benchmark == 'slow_M_len': # Kim et al. 2015 from Perreault 2003 experiments

    sim_path = base_path / 'slow_M' / 'sim' # experimental path
    exp_path = base_path / 'slow_M' / 'exp' # simulations path

    exp_twitch_0 = np.load(exp_path / 'len_iso_twitch_0.npy') # load experimental forces
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

    sim_twitch_0 = np.load(sim_path / 'len_iso_twitch_0.npy') # load experimental forces
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

    t_end = 1.4 # total seconds
    time_dt = np.arange(0, t_end, dt) # time for BB
    MVC = 30.25

    fig = plt.figure(figsize=(9, 8))

    plt.subplot(4, 3, 1)
    plt.plot(time_dt, exp_twitch_0, 'k')
    plt.plot(time_dt, sim_twitch_0, 'r')
    plt.title(r"1 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm",  x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 10))

    plt.subplot(4, 3, 2)
    plt.plot(time_dt, exp_twitch_8, 'k', label='Experimental')
    plt.plot(time_dt, sim_twitch_8, 'r', label='Simulated')
    plt.title(r"1 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.3, y=0.99, weight='bold')
    plt.legend(loc=(0.2, 0.6), fontsize=12)
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 10))

    plt.subplot(4, 3, 3)
    plt.plot(time_dt, exp_twitch_16, 'k')
    plt.plot(time_dt, sim_twitch_16, 'r')
    plt.title(r"1 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm",x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 10))

    plt.subplot(4, 3, 4)
    plt.plot(time_dt, exp_iso_0_10, 'k')
    plt.plot(time_dt, sim_iso_0_10, 'r')
    plt.title(r"10 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm",  x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 5)
    plt.plot(time_dt, exp_iso_8_10, 'k')
    plt.plot(time_dt, sim_iso_8_10, 'r')
    plt.title(r"10 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 6)
    plt.plot(time_dt, exp_iso_16_10, 'k')
    plt.plot(time_dt, sim_iso_16_10, 'r')
    plt.title(r"10 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm",x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 7)
    plt.plot(time_dt, exp_iso_0_20, 'k')
    plt.plot(time_dt, sim_iso_0_20, 'r')
    plt.title(r"20 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm",  x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 8)
    plt.plot(time_dt, exp_iso_8_20, 'k')
    plt.plot(time_dt, sim_iso_8_20, 'r')
    plt.title(r"20 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 9)
    plt.plot(time_dt, exp_iso_16_20, 'k')
    plt.plot(time_dt, sim_iso_16_20, 'r')
    plt.title(r"20 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm",x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.ylim((0, 30))

    plt.subplot(4, 3, 10)
    plt.plot(time_dt, exp_iso_0_40, 'k')
    plt.plot(time_dt, sim_iso_0_40, 'r')
    plt.title(r"40 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = 0 mm",  x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 35))

    plt.subplot(4, 3, 11)
    plt.plot(time_dt, exp_iso_8_40, 'k')
    plt.plot(time_dt, sim_iso_8_40, 'r')
    plt.title(r"40 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 8 mm", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 35))
    plt.xlabel('Time [s]', weight='bold', fontsize=14)

    plt.subplot(4, 3, 12)
    plt.plot(time_dt, exp_iso_16_40, 'k')
    plt.plot(time_dt, sim_iso_16_40, 'r')
    plt.title(r"40 Hz, $\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm",x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 35))

    fig.text(0.01, 0.5, 'Force [N]', va='center', rotation='vertical', 
         weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    # Error
    def pct_errors(exp: np.ndarray, sim: np.ndarray, MVC: float):
        # considera solo i punti dove almeno uno è diverso da zero
        mask = (exp != 0) | (sim != 0)
        if not np.any(mask):
            return 0.0, 0.0, 0.0

        exp_nz = exp[mask]
        sim_nz = sim[mask]

        abs_err_pct = np.abs((sim_nz / MVC) - (exp_nz / MVC)) * 100.0
        mae   = float(np.mean(abs_err_pct))
        maxae = float(np.max(abs_err_pct))
        stde  = float(np.std(abs_err_pct))
        return mae, maxae, stde

    print("\n=== Benchmark: slow_M_len (Perreault/Kim) ===")
    print(f"MVC = {MVC:.2f} N\n")

    # Trials list
    trials = [
        ("Twitch 1 Hz, 0 mm",    exp_twitch_0,   sim_twitch_0),
        ("Twitch 1 Hz, -8 mm",   exp_twitch_8,   sim_twitch_8),
        ("Twitch 1 Hz, -16 mm",  exp_twitch_16,  sim_twitch_16),

        ("10 Hz,  -0 mm",        exp_iso_0_10,   sim_iso_0_10),
        ("10 Hz,  -8 mm",        exp_iso_8_10,   sim_iso_8_10),
        ("10 Hz,  -16 mm",       exp_iso_16_10,  sim_iso_16_10),

        ("20 Hz,  -0 mm",        exp_iso_0_20,   sim_iso_0_20),
        ("20 Hz,  -8 mm",        exp_iso_8_20,   sim_iso_8_20),
        ("20 Hz,  -16 mm",       exp_iso_16_20,  sim_iso_16_20),

        ("40 Hz,  -0 mm",        exp_iso_0_40,   sim_iso_0_40),
        ("40 Hz,  -8 mm",        exp_iso_8_40,   sim_iso_8_40),
        ("40 Hz,  -16 mm",       exp_iso_16_40,  sim_iso_16_40),
    ]

    all_mae, all_maxae, all_stde = [], [], []
    print("— Errori per trial —")
    for name, exp, sim in trials:
        mae, maxae, stde = pct_errors(exp, sim, MVC)
        all_mae.append(mae)
        all_maxae.append(maxae)
        all_stde.append(stde)
        print(f"{name:>18}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")

    print(
        f"\n{'Overall Average':>18}:  "
        f"MAE = {np.mean(all_mae):6.2f}% MVC  "
        f"(std = {np.mean(all_stde):6.2f})   "
        f"MaxAE = {np.mean(all_maxae):6.2f}% MVC\n"
    )

    summarize_trials(all_mae, all_maxae,
                 label="— slow_M_len summary (all trials) —",
                 unit="% MVC")

    # ------------------------------------------------
    # funzione per i 3 pannelli (0, -8, -16 mm)
    # ------------------------------------------------
    def build_len_err(exp_list, sim_list, MVC):
        mean_list, max_list, std_list = [], [], []
        for exp, sim in zip(exp_list, sim_list):
            # maschera non-zero
            mask = (exp != 0) | (sim != 0)
            if np.any(mask):
                exp_nz = exp[mask]
                sim_nz = sim[mask]
                abs_err = np.abs((sim_nz / MVC) - (exp_nz / MVC)) * 100.0
                mean_list.append(np.mean(abs_err))
                max_list.append(np.max(abs_err))
                std_list.append(np.std(abs_err))
            else:
                mean_list.append(0.0)
                max_list.append(0.0)
                std_list.append(0.0)
        return (
            np.array(mean_list),
            np.array(max_list),
            np.array(std_list),
        )

    # 0 mm
    mean_0, max_0, std_0 = build_len_err(
        [exp_twitch_0, exp_iso_0_10, exp_iso_0_20, exp_iso_0_40],
        [sim_twitch_0, sim_iso_0_10, sim_iso_0_20, sim_iso_0_40],
        MVC
    )
    # -8 mm
    mean_8, max_8, std_8 = build_len_err(
        [exp_twitch_8, exp_iso_8_10, exp_iso_8_20, exp_iso_8_40],
        [sim_twitch_8, sim_iso_8_10, sim_iso_8_20, sim_iso_8_40],
        MVC
    ) 
    # -16 mm
    mean_16, max_16, std_16 = build_len_err(
        [exp_twitch_16, exp_iso_16_10, exp_iso_16_20, exp_iso_16_40],
        [sim_twitch_16, sim_iso_16_10, sim_iso_16_20, sim_iso_16_40],
        MVC
    )

    x = np.arange(1, 4 + 1)
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
    plot_panel(axes[2], x, mean_16,  max_16,  std_16,  r"$\boldsymbol{\Delta}\mathbf{L}$ = - 16 mm")

    axes[0].set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Stimulation frequency', fontweight='bold')
    axes[0].legend()

    plt.tight_layout()
    plt.show()

    
elif benchmark == 'MU': # Burke 1974 experiments

    sim_path = base_path / 'MU' / 'sim' # experimental path
    exp_path = base_path / 'MU' / 'exp' # simulations path

    exp_S_twitch = np.load(exp_path / 'slow_twitch.npy') # load experimental forces
    exp_S_unfused = np.load(exp_path / 'slow_unfused.npy')
    exp_S_fused = np.load(exp_path / 'slow_fused.npy')
    exp_F_twitch = np.load(exp_path / 'fast_twitch.npy')
    exp_F_unfused = np.load(exp_path / 'fast_unfused.npy')
    exp_F_fused = np.load(exp_path / 'fast_fused.npy')

    sim_S_twitch = np.load(sim_path / 'slow_twitch.npy') # load simulated forces
    sim_S_unfused = np.load(sim_path / 'slow_unfused.npy')
    sim_S_fused = np.load(sim_path / 'slow_fused.npy')
    sim_F_twitch = np.load(sim_path / 'fast_twitch_nosag.npy') # sag is not actiove as tp is not detected during a twitch
    sim_F_twitch_nosag = np.load(sim_path / 'fast_twitch_nosag.npy')
    sim_F_unfused = np.load(sim_path / 'fast_unfused.npy')
    sim_F_unfused_nosag = np.load(sim_path / 'fast_unfused_nosag.npy')
    sim_F_fused = np.load(sim_path / 'fast_fused.npy')

    t_end_S = 1.8 # total seconds
    t_end_F = 1.2
    time_dt_S = np.arange(0, t_end_S, dt) # time for BB
    time_dt_F = np.arange(0, t_end_F, dt)
    MVC_S = 0.04
    MVC_F = 0.37

    fig = plt.figure(figsize=(12, 7))

    plt.subplot(2, 3, 1)
    plt.plot(time_dt_S, exp_S_twitch, 'k')
    plt.plot(time_dt_S, sim_S_twitch, 'r')
    plt.title(r"1 Hz at $\mathbf{L^{CE}_{0}}$ (S MU)", x=0.3, y=0.99, weight='bold')
    plt.ylabel('Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, MVC_S+0.01))

    plt.subplot(2, 3, 4)
    plt.plot(time_dt_F, exp_F_twitch, 'k')
    plt.plot(time_dt_F, sim_F_twitch, 'r')
    #plt.plot(time_dt_F, sim_F_twitch_nosag, 'r--')
    plt.title(r"1 Hz at $\mathbf{L^{CE}_{0}}$ (F MU)", x=0.3, y=0.99, weight='bold')
    plt.ylabel('Force [N]', weight='bold', fontsize=14)
    plt.ylim((0, MVC_F))

    plt.subplot(2, 3, 2)
    plt.plot(time_dt_S, exp_S_unfused, 'k')
    plt.plot(time_dt_S, sim_S_unfused, 'r')
    plt.title(r"12.5 Hz at $\mathbf{L^{CE}_{0}}$ (S MU)", x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='y', which='both', labelbottom=False)
    plt.ylim((0, MVC_S+0.01))

    plt.subplot(2, 3, 5)
    plt.plot(time_dt_F, exp_F_unfused, 'k')
    plt.plot(time_dt_F, sim_F_unfused, 'r')
    plt.plot(time_dt_F, sim_F_unfused_nosag, 'r--')
    plt.title(r"25 Hz at $\mathbf{L^{CE}_{0}}$ (F MU)", x=0.3, y=0.99, weight='bold')
    plt.xlabel('Time [s]', weight='bold', fontsize=14)
    plt.gca().tick_params(axis='y', which='both', labelbottom=False)
    plt.ylim((0, MVC_F))

    plt.subplot(2, 3, 3)
    plt.plot(time_dt_S, exp_S_fused, 'k', label='Experimental')
    plt.plot(time_dt_S, sim_S_fused, 'r', label='Simulated (sag)')
    plt.plot(time_dt_S, sim_S_fused, 'r--', label='Simulated (no sag)')
    plt.title(r"40 Hz at $\mathbf{L^{CE}_{0}}$ (S MU)", x=0.3, y=0.99, weight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.gca().tick_params(axis='y', which='both', labelbottom=False)
    plt.ylim((0, MVC_S+0.01))

    plt.subplot(2, 3, 6)
    plt.plot(time_dt_F, exp_F_fused, 'k')
    plt.plot(time_dt_F, sim_F_fused, 'r')
    plt.title(r"40 Hz at $\mathbf{L^{CE}_{0}}$ (F MU)", x=0.3, y=0.99, weight='bold')
    plt.gca().tick_params(axis='y', which='both', labelbottom=False)
    plt.ylim((0, MVC_F))

    plt.tight_layout()
    plt.show()

    def pct_errors(exp: np.ndarray, sim: np.ndarray, MVC: float):
        mask = (exp != 0) | (sim != 0)
        if not np.any(mask):
            return 0.0, 0.0, 0.0

        exp_nz = exp[mask]
        sim_nz = sim[mask]

        abs_err_pct = np.abs((sim_nz / MVC) - (exp_nz / MVC)) * 100.0
        mae   = float(np.mean(abs_err_pct))
        maxae = float(np.max(abs_err_pct))
        stde  = float(np.std(abs_err_pct))
        return mae, maxae, stde

    print("\n=== Benchmark: MU (Burke 1974) ===")
    print(f"MVC_S (slow) = {MVC_S:.4f} N   |   MVC_F (fast) = {MVC_F:.4f} N\n")

    # --------- Slow MU trials ---------
    slow_trials = [
        ("Slow Twitch   (1 Hz)",   exp_S_twitch,   sim_S_twitch,   MVC_S),
        ("Slow Unfused (12.5 Hz)", exp_S_unfused,  sim_S_unfused,  MVC_S),
        ("Slow Fused   (40 Hz)",   exp_S_fused,    sim_S_fused,    MVC_S),
    ]

    mae_s, maxae_s, std_s = [], [], []
    print("— Slow MU trials —")
    for name, exp, sim, mvc in slow_trials:
        mae, maxae, stde = pct_errors(exp, sim, mvc)
        mae_s.append(mae)
        maxae_s.append(maxae)
        std_s.append(stde)
        print(f"{name:>22}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")

    print(
        f"{'Slow Average':>22}:  "
        f"MAE = {np.mean(mae_s):6.2f}% MVC  "
        f"(std = {np.mean(std_s):6.2f})   "
        f"MaxAE = {np.mean(maxae_s):6.2f}% MVC\n"
    )

    # --------- Fast MU trials ---------
    fast_trials = [
        ("Fast Twitch  (1 Hz)",   exp_F_twitch,   sim_F_twitch,   MVC_F),
        ("Fast Unfused (25 Hz)",  exp_F_unfused,  sim_F_unfused,  MVC_F),
        ("Fast Fused   (40 Hz)",  exp_F_fused,    sim_F_fused,    MVC_F),
    ]

    mae_f, maxae_f, std_f = [], [], []
    print("— Fast MU trials —")
    for name, exp, sim, mvc in fast_trials:
        mae, maxae, stde = pct_errors(exp, sim, mvc)
        mae_f.append(mae)
        maxae_f.append(maxae)
        std_f.append(stde)
        print(f"{name:>22}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")

    print(
        f"{'Fast Average':>22}:  "
        f"MAE = {np.mean(mae_f):6.2f}% MVC  "
        f"(std = {np.mean(std_f):6.2f})   "
        f"MaxAE = {np.mean(maxae_f):6.2f}% MVC\n"
    )

        # --------- Fast MU trials — NO SAG ---------
    fast_trials_nosag = [
        ("Fast Twitch  (1 Hz, no sag)",   exp_F_twitch,  sim_F_twitch_nosag,   MVC_F),
        ("Fast Unfused (25 Hz, no sag)",  exp_F_unfused, sim_F_unfused_nosag,  MVC_F),
    ]

    mae_f_nosag, maxae_f_nosag, std_f_nosag = [], [], []
    print("— Fast MU trials — NO SAG —")
    for name, exp, sim_nosag, mvc in fast_trials_nosag:
        mae, maxae, stde = pct_errors(exp, sim_nosag, mvc)
        mae_f_nosag.append(mae)
        maxae_f_nosag.append(maxae)
        std_f_nosag.append(stde)
        print(f"{name:>30}:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")

    print(
        f"{'Fast Average (no sag)':>30}:  "
        f"MAE = {np.mean(mae_f_nosag):6.2f}% MVC  "
        f"(std = {np.mean(std_f_nosag):6.2f})   "
        f"MaxAE = {np.mean(maxae_f_nosag):6.2f}% MVC\n"
    )

    summarize_trials(mae_s, maxae_s, label="— MU summary: S MU (all trials) —", unit="% MVC")
    summarize_trials(mae_f, maxae_f, label="— MU summary: F MU (all trials) —", unit="% MVC")
    summarize_trials(mae_f_nosag, maxae_f_nosag, label="— MU summary: F MU NO SAG (all trials) —", unit="% MVC")


elif benchmark == 'fast_M_iso':  # Millard 2025 experiments

    sim_path = base_path / 'fast_M' / 'sim'
    exp_path = base_path / 'fast_M' / 'exp'

    # FFR 
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

    # FLR
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

    t_end = 0.9991
    time_dt = np.arange(0, t_end, dt)
    MVC = 2.49

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    ax_ffr_exp = axes[0, 0]
    ax_ffr_sim = axes[0, 1]
    ax_flr_exp = axes[1, 0]
    ax_flr_sim = axes[1, 1]

    # palette grigi per exp (nero- grigio chiaro)
    n_ffr = len(freqs)
    gray_levels = np.linspace(0.75, 0.15, n_ffr)
    red_levels = np.linspace(0.4, 1.0, n_ffr)  # per simulazioni

    # FFR experimental
    for i, (f, y) in enumerate(zip(freqs, exp_FFR_series)):
        ax_ffr_exp.plot(time_dt, y, color=str(gray_levels[i]), label=f"{f} Hz")
    ax_ffr_exp.set_title("Experimental", weight='bold')
    ax_ffr_exp.set_ylim([-0.08, 1.7])
    ax_ffr_exp.set_ylabel("Force [N]", weight='bold')
    ax_ffr_exp.set_xlabel("Time [s]", weight='bold')
    ax_ffr_exp.legend(loc='upper right', fontsize=7)

    # FFR simulated
    for i, (f, y) in enumerate(zip(freqs, sim_FFR_series)):
        ax_ffr_sim.plot(time_dt, y, color=(red_levels[i], 0, 0), label=f"{f} Hz")
    ax_ffr_sim.set_ylim([-0.08, 1.7])
    ax_ffr_sim.set_title("Simulated", weight='bold')
    ax_ffr_sim.set_xlabel("Time [s]", weight='bold')
    ax_ffr_sim.legend(loc='upper right', fontsize=7)

    # FLR experimental
    n_flr = len(disp_mm)
    gray_levels_flr = np.linspace(0.15, 0.75, n_flr)
    red_levels_flr = np.linspace(0.4, 1.0, n_flr)

    for i, (d, y) in enumerate(zip(disp_mm, exp_FLR_series)):
        ax_flr_exp.plot(time_dt, y, color=str(gray_levels_flr[i]),  label=f"$\Delta L$ = + {d:.1f} mm")
    ax_flr_exp.set_title("Experimental", weight='bold')
    ax_flr_exp.set_ylabel("Force [N]", weight='bold')
    ax_flr_exp.set_xlabel("Time [s]", weight='bold')
    ax_flr_exp.legend(loc='upper right', fontsize=7)

    # FLR simulated
    for i, (d, y) in enumerate(zip(disp_mm, sim_FLR_series)):
        ax_flr_sim.plot(time_dt, y, color=(red_levels_flr[i], 0, 0), label=f"$\Delta L$ = + {d:.1f} mm")
    ax_flr_sim.set_title("Simulated", weight='bold')
    ax_flr_sim.set_xlabel("Time [s]", weight='bold')
    ax_flr_sim.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.show()

    # =======================
    # Errors % MVC
    # =======================
    def pct_err_all(exp_arr: np.ndarray, sim_arr: np.ndarray, MVC: float):
        # usa solo i campioni dove almeno uno è diverso da 0
        mask = (exp_arr != 0) | (sim_arr != 0)
        if not np.any(mask):
            return 0.0, 0.0, 0.0

        exp_nz = exp_arr[mask]
        sim_nz = sim_arr[mask]

        abs_err_pct = np.abs((sim_nz / MVC) - (exp_nz / MVC)) * 100.0
        mae   = float(np.mean(abs_err_pct))
        maxae = float(np.max(abs_err_pct))
        stde  = float(np.std(abs_err_pct))
        return mae, maxae, stde

    print("\n=== Benchmark: fast_M_iso ===")
    print(f"MVC = {MVC:.2f} N\n")

    # FFR errors
    ffr_mean_list, ffr_max_list, ffr_std_list = [], [], []
    print("— FFR errors —")
    for f, e, s in zip(freqs, exp_FFR_series, sim_FFR_series):
        mae, maxae, stde = pct_err_all(e, s, MVC)
        ffr_mean_list.append(mae)
        ffr_max_list.append(maxae)
        ffr_std_list.append(stde)
        print(f"{f:>3} Hz:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")
    print()

    # FLR errors
    flr_mean_list, flr_max_list, flr_std_list = [], [], []
    print("— FLR errors —")
    for d, e, s in zip(disp_mm, exp_FLR_series, sim_FLR_series):
        mae, maxae, stde = pct_err_all(e, s, MVC)
        flr_mean_list.append(mae)
        flr_max_list.append(maxae)
        flr_std_list.append(stde)
        print(f"+{d:4.1f} mm:  MAE = {mae:6.2f}% MVC  (std = {stde:6.2f})   MaxAE = {maxae:6.2f}% MVC")
    print()

    ffr_mean_arr = np.array(ffr_mean_list)
    ffr_max_arr  = np.array(ffr_max_list)
    ffr_std_arr  = np.array(ffr_std_list)
    ffr_mmae = np.mean(ffr_mean_arr)
    ffr_mstd = np.std(ffr_mean_arr)
    print(f"average MAE = {ffr_mmae:6.2f}% MVC  (std = {ffr_mstd:6.2f})")

    flr_mean_arr = np.array(flr_mean_list)
    flr_max_arr  = np.array(flr_max_list)
    flr_std_arr  = np.array(flr_std_list)
    flr_mmae = np.mean(flr_mean_arr)
    flr_mstd = np.std(flr_mean_arr)
    print(f"average MAE = {flr_mmae:6.2f}% MVC  (std = {flr_mstd:6.2f})")

    summarize_trials(ffr_mean_list, ffr_max_list,
                 label="— fast_M_iso summary: FFR (all freqs) —",
                 unit="% MVC")

    summarize_trials(flr_mean_list, flr_max_list,
                 label="— fast_M_iso summary: FLR (all ΔL) —",
                 unit="% MVC")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # --- FFR ---
    lower = np.maximum(ffr_mean_arr - ffr_std_arr, 0)
    upper = ffr_mean_arr + ffr_std_arr
    ax1.plot(freqs, ffr_mean_arr, '-o', color='k', label='mAE ± SD')
    ax1.fill_between(freqs, lower, upper, color='k', alpha=0.2)
    ax1.plot(freqs, ffr_max_arr, '--*', color='k', label='MAE')
    ax1.set_ylim([0,60])
    ax1.set_xlabel('Stimulation frequency [Hz]', fontweight='bold')
    ax1.set_ylabel(r'Error [%$\mathbf{F_{0}}$]', fontweight='bold')
    #ax1.set_title('FFR', fontweight='bold')
    ax1.set_ylim([0, 70])
    ax1.legend(loc='upper left')

    # --- FLR ---
    lower_f = np.maximum(flr_mean_arr - flr_std_arr, 0)
    upper_f = flr_mean_arr + flr_std_arr
    ax2.plot(disp_mm, flr_mean_arr, '-o', color='k', label='mAE ± SD')
    ax2.fill_between(disp_mm, lower_f, upper_f, color='k', alpha=0.2)
    ax2.plot(disp_mm, flr_max_arr, '--*', color='k', label='MAE')
    ax2.set_ylim([0,60])
    ax2.set_xlabel(r"$\boldsymbol{\Delta}\mathbf{L}$ [mm]", fontweight='bold')
    #ax2.set_title('FLR', fontweight='bold')
    ax2.set_ylim([0, 70])

    plt.tight_layout()
    plt.show()


elif benchmark == 'fast_M_dyn': # Brown 1999 experiments

    sim_path = base_path / 'fast_M' / 'sim' # experimental path
    exp_path = base_path / 'fast_M' / 'exp' # simulations path

    exp_dyn_120_095_s = np.load(exp_path / 'dyn_120_0.95_length.npy') # load experimental forces
    exp_dyn_120_095_l = np.load(exp_path / 'dyn_120_0.95_short.npy')
    exp_dyn_20_095_s = np.load(exp_path / 'dyn_20_0.95_length.npy')
    exp_dyn_20_095_l = np.load(exp_path / 'dyn_20_0.95_short.npy')
    exp_dyn_40_08_s = np.load(exp_path / 'dyn_40_0.8_length.npy')
    exp_dyn_40_08_l = np.load(exp_path / 'dyn_40_0.8_short.npy')
    exp_dyn_40_11_s = np.load(exp_path / 'dyn_40_1.1_length.npy') 
    exp_dyn_40_11_l = np.load(exp_path / 'dyn_40_1.1_short.npy')
    exp_dyn_60_095_s = np.load(exp_path / 'dyn_60_0.95_length.npy') 
    exp_dyn_60_095_l = np.load(exp_path / 'dyn_60_0.95_short.npy')

    sim_dyn_120_095_s = np.load(sim_path / 'dyn_120_0.95_length.npy') # load simulated forces
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

    fig = plt.figure(figsize=(12, 7))

    plt.subplot(3, 3, 1)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_20_095_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_20_095_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_20_095_l[0:len(disp_sub_l)], 'red')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_20_095_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"20 Hz, 0.95$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    plt.subplot(3, 3, 2)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_08_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_08_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_08_l[0:len(disp_sub_l)], 'red')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_08_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"40 Hz, 0.8$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    plt.subplot(3, 3, 3)
    plt.plot(time_dt_max, exp_dyn_120_095_l, 'k', label='Experimental-Shortening')
    plt.plot(time_dt_max, exp_dyn_120_095_s, 'gray', label='Experimental-Lengthening')
    plt.plot(time_dt_max, sim_dyn_120_095_l, 'r', label='Simulated-Shortening')
    plt.plot(time_dt_max, sim_dyn_120_095_s, 'orange', label='Simulated-Lengthening')
    plt.axvline(time_dt_sub[1100], color='gray', linestyle='--', linewidth=1)
    plt.title(r"120 Hz, 0.95$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.legend(loc=(0.2, 1.3), fontsize=12)
    plt.ylim((0, 1.7))
    plt.xlim((0.08, 0.16))

    plt.subplot(3, 3, 4)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_60_095_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_60_095_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_60_095_l[0:len(disp_sub_l)], 'r')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_60_095_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"60 Hz, 0.95$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    plt.subplot(3, 3, 5)
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_11_l[0:len(disp_sub_l)], 'k')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], exp_dyn_40_11_s[0:len(disp_sub_l)], 'gray')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_11_l[0:len(disp_sub_l)], 'r')
    plt.plot(time_dt_sub[0:len(disp_sub_l)], sim_dyn_40_11_s[0:len(disp_sub_l)], 'orange')
    plt.axvline(time_dt_sub[1122], color='gray', linestyle='--', linewidth=1)
    plt.title(r"40 Hz, 1.1$\mathbf{L^{CE}_{0}}$", x=0.3, y=0.99, weight='bold')
    plt.ylim((0, 1.9))

    # Displacement
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

    fig.text(0.03, 0.67, 'Normalized force', va='center', rotation='vertical', 
        weight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()

    # ==========================
    # ERRORs fast_M_dyn
    # ==========================

    cond_labels = [
        "20 Hz (0.95 L0)",
        "40 Hz (0.80 L0)",
        "60 Hz (0.95 L0)",
        "40 Hz (1.10 L0)",
        "120 Hz (0.95 L0)",
    ]

    # SHORTENING (l)
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

    # LENGTHENING (s)
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

    # start samples
    idx_sub = 1119
    idx_120_short = 1083
    idx_120_len   = 1090

    def compute_err_list(exp_list, sim_list, is_short=True):
        mean_list, max_list, std_list = [], [], []
        for i, (e, s) in enumerate(zip(exp_list, sim_list)):
            if i < 4:  # submaximal conditions (< 120Hz)
                start = idx_sub
            else:      # maximal frequency (120 hz)
                start = idx_120_short if is_short else idx_120_len

            # % errors (normalized data)
            abs_err = np.abs(s[start:] - e[start:]) * 100.0
            mean_list.append(np.mean(abs_err))
            max_list.append(np.max(abs_err))
            std_list.append(np.std(abs_err))
        return (
            np.array(mean_list),
            np.array(max_list),
            np.array(std_list),
        )

    short_mean, short_max, short_std = compute_err_list(short_exp, short_sim, is_short=True)
    length_mean, length_max, length_std = compute_err_list(length_exp, length_sim, is_short=False)

        # sprint
    print("\n=== Benchmark: fast_M_dyn (errors) ===\n")

    print("Shortening:")
    for lab, mae, mxe, stde in zip(cond_labels, short_mean, short_max, short_std):
        print(f"  {lab:18s}  MAE = {mae:6.2f}%  (std = {stde:6.2f})   MaxAE = {mxe:6.2f}%")

    print("\nLengthening:")
    for lab, mae, mxe, stde in zip(cond_labels, length_mean, length_max, length_std):
        print(f"  {lab:18s}  MAE = {mae:6.2f}%  (std = {stde:6.2f})   MaxAE = {mxe:6.2f}%")
    print()

    all_mae  = list(short_mean)  + list(length_mean)  
    all_maxe = list(short_max)   + list(length_max)

    summarize_trials(all_mae, all_maxe,
                 label="— fast_M_dyn summary: (all conditions) —",
                 unit="%")

    # ==========================
    # PLOT errors (2 subplot)
    # ==========================
    x = np.arange(1, 5 + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    # --- SHORTENING ---
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

    # --- LENGTHENING ---
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

