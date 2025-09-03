"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Animal benchmarks of slow & fast muscle isometric & dynamic contrations for testing MU model.

"""


from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

from MU_model2 import MU_model  


# =====================================================================
# Utils
# =====================================================================

def read_data(file, data_type):

    if data_type == 'displacement_M':  # maximal benchmark
      skip = 16
    elif data_type == 'displacement_SM': # submaximal benchmark
      skip = 18
    elif data_type == 'force_M': # maximal benchmark
      skip = 15
    elif data_type == 'force_SM': # submaximal benchmark (dynamic trials)
      skip = 18
    elif data_type == 'force_force_SM': # submaximal benchmark (isometric trials)
      skip = 19
    elif data_type == 'Millard_EDL_iso':
      skip = 30
    elif data_type == 'Millard_EDL_dyn':
      skip = 24

    data = np.loadtxt(file, delimiter='\t', skiprows=skip)

    return data


def prepare_segment_iso(data, fs=1000, target_freq=30, dt=1e-4):

    freq_map = {30:0, 50:1, 60:2, 70:3, 80:4, 90:5, 100:6, 120:7} # frequency map
    seg_idx = freq_map[target_freq]

    seg_len = 20000 # Each segment is 20000 samples long
    start = seg_idx * seg_len

    N = 1000 # Each part starts from 0 with 1000 samples
    seg_slice = slice(start, start + N)

    length = np.asarray(data[seg_slice, 1], dtype=float) # Colonne (0-based): lunghezza=1, forza=2, spikes=11
    force  = np.asarray(data[seg_slice, 2], dtype=float)
    spikes = np.asarray(data[seg_slice, 11], dtype=float)

    t = np.arange(N, dtype=float) / fs # otriginal time
    prev = spikes[:-1] # 0-1 transitions
    curr = spikes[1:]
    rising_edges = np.where((prev == 0) & (curr == 1))[0] + 1  # discharges when 1 is preceeded by 0
    spike_times_sec = t[rising_edges]

    t_end = t[-1]
    t_hi = np.arange(0.0, t_end + 1e-12, dt) # interpolated time

    kind = 'cubic'
    length_hi = sp.interpolate.interp1d(t, length, kind=kind)(t_hi)
    force_hi = sp.interpolate.interp1d(t, force,  kind=kind)(t_hi)

    return {
        'freq': target_freq,
        't': t,
        'force': force,
        'length': length,
        'spike_times_sec': spike_times_sec,
        't_hi': t_hi,
        'force_hi': force_hi,
        'length_hi': length_hi,
    }


# ====================================================================
# Benchmark config
# ====================================================================

dt = 1e-4  # s
base_path = Path('..') / 'benchmarkData'
save = False

root = tk.Tk() # Initialise input window
root.withdraw()  # Hide the root window
benchmark = simpledialog.askstring("Input", "Select benchmark ('max', 'sub', 'len', 'fast', 'Ca'):") 

# Variables to be populated
muscle = None
yielding = 0
sag = 0
MVC = None
l_T_slack = None
l_M_opt = None
l_M_0 = None
alpha_0 = None
time_dt = None
l_MT = None
exp_force = None
Distimes = None


if benchmark == 'max': # MAXIMAL benchmark - SLOW MUSCLE (Sandercock 1997) """
    
    scale = simpledialog.askstring("Input", "Amplitude displacement scale (e.g., 0.05, 0.1, 0.25, 0.5, 1, 2):")
    scales = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0], dtype=float)
    idx_arr = np.where(np.isclose(scales, float(scale), rtol=0, atol=1e-9))[0]
    idx = int(idx_arr[0]) 

    path = base_path / 'slowMuscle_maximalActivation'

    t_end = 2
    fs = '70'
    muscle = 'rat_SOL' # dorsi/plantar
    time_dt = np.arange(0, t_end, dt) # time for BB
    Distimes = np.arange(0, t_end, 1/float(fs)) # create array of dischare times at 70 Hz
    disp = read_data(path / 'displacement.dat', 'displacement_M')
    disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # load displacement
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.36, 17.1, 17.1, 17.1, 6*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) - 2 # Musculo-tendon length (mm)
    l_MT = l_MT_0 + disp*float(scale) # scaled MT length  

    exp_force_path = os.path.join(path, f"force_trial{idx+1}.dat")
    exp_force = read_data(exp_force_path, 'force_M')
    exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='quadratic')(np.arange(0,2,dt)) # interpolate to have equal n of points

    yielding = 0 # yielding not included
    sag = 0
    
    
elif benchmark == 'sub': #SUB-MAXIMAL benchmark - SLOW MUSCLE (Perreault 2003) 
    
    trial = simpledialog.askstring("Input", "Trial type? ('iso' or 'dyn'):")
    stim = simpledialog.askstring("Input", "Stimulation type? ('c' or 'v'):")
    fs = simpledialog.askstring("Input", "Stimulation frequency (Hz):")

    muscle = 'rat_SOL' # dorsi/plantar
    yielding = 1 # yielding included
    sag = 0
    
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [26.9, 65, 30, 30, 7.5*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) - 4  # Musculo-tendon length (mm)
    
    t_end = 2
    time_dt = np.arange(0, t_end, dt) # time for BB

    path_disp = base_path / 'slowMuscle_submaximalActivation'
    path = path_disp / f"{stim}_freq"

    Distimes = np.load(path / f"{fs}_{stim}_times.npy")  # sec
    
    if trial == 'iso':
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # constant MT length
        exp_force = read_data(path / f"force_isometric_{stim}{fs}.dat", 'force_force_SM')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))
    
    elif trial == 'dyn':
        
        d = simpledialog.askstring("Input", "Displacement amplitude (1 or 8mm):")
        
        disp = read_data(path_disp / f"displacement_{d}.dat", 'displacement_SM') # read displacement
        disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate displacement
        l_MT = l_MT_0 + (disp + 8) # scaled MT length
        exp_force = read_data(path / f"force_{stim}{fs}_{d}.dat", 'force_SM')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))


elif benchmark == 'len': # TWITCH, SUB-TETANIC, TETANIC at different lengths (Kim 2015)
    
    l = simpledialog.askstring("Input", "Length? ('0', '8', or '16'):")
    fs = simpledialog.askstring("Input", '"Stimulation frequency (1, 10, 20, 40 Hz):"')
    yielding = 0
    sag = 0
    
    path = base_path / 'slowMuscle_length'
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [30, 65, 30, 30, 7.5*np.pi/180] # MVC and M/T lengths
    t_end = 1.4
    muscle = 'rat_SOL' # dorsi/plantar
    time_dt = np.arange(0, t_end, dt) # time for BB
    
    if l == '0':
        d = 8
    elif l == '8':
        d = 0 
    elif l == '16':
        d = -8
        
    l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) - 4    # Musculo-tendon length (mm)
    l_MT_0 = l_MT_0 + d  # apply displacement
    l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array

    exp_force = np.load(path / f"{l}_{fs}_interp.npy")
    Distimes = np.load(path / f"{l}_{fs}_times.npy") # exp. discharge times
    if fs == '1': # twitch
        Distimes = np.round(Distimes, 3)
         

elif benchmark == 'fast':  # Test fast muscle (Millard exp data, rat EDL & Chelikowski 1999 rat GM)

    type = simpledialog.askstring("Input", "muscle or MU: ")
    
    if type == 'muscle':
        
        trial = simpledialog.askstring("Input", "Trial type? ('iso_f', 'iso_l', 'dyn'):")
        yielding = 0
        sag = 0

        if trial == 'iso_f': # isometric frequency variation

            path = base_path / 'fastMuscle_Millard'
            data = read_data(path / 'FFR.ddf', 'Millard_EDL_iso')

            fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (30, 50, 60, 70, 80, 90, 100, 120):")

            part = prepare_segment_iso(data, f=1000, target_freq = float(fs), dt=1e-4)
            Distimes = part['spike_times_sec']
            time_dt = part['t_hi']
            exp_force = part['force_hi']
            muscle = 'rat_EDL'
            
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.5, 9, 13.7, 13.7, 10*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
    
    elif type == 'MU': # fast MU benchmark Chelichowski 1999
        
        path = base_path / 'fastMU'
        yielding = 0
        sag = 1
        
        fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (25, 30, 35, 40, 150):")
        exp_force = np.load(path / f"iso_{fs}_interp.npy")
        Distimes = np.load(path / f"iso_{fs}_times.npy")
        t_end = 0.7
        muscle = 'rat_GM' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.078, 24.8, 13.5, 13.5, 0] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
        
    
elif benchmark == 'Ca': # Test Ca dynamics (Hollingworth, Rincon exp. data)
    
    fibre = simpledialog.askstring("Input", "Fibre type? ('slow', 'fast'):")
    yielding = 0
    sag = 0
    
    path = base_path / 'Ca_transients'
    t_end = 1.4
    time_dt = np.arange(0, t_end, dt)
    
    if fibre == 'slow':

        Ca_slow_23 = pd.read_csv(path / "Ca_slow_23_100Hz.csv", delimiter=' ').to_numpy()

        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1, 0, 30, 30, 0] # at optimal sarcomere length (assumption)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array
        fs = '102' # adjusted from paper
        T = 1/float(fs) 
        Distimes = np.arange(0, 0.04, T)
        muscle = 'rat_SOL' # slow fibre muscle

    elif fibre == 'fast':

        Ca_fast_35 = pd.read_csv(path / "Ca_fast_35_125Hz.csv", delimiter=' ').to_numpy()

        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1.6, 0, 30, 1.6*30, 0] # at longer sarcomere length (see article)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array
        fs = '125'
        T = 1/float(fs) 
        Distimes = np.arange(0, 0.08, T)
        muscle = 'rat_EDL' # fast fibre muscle
    

# =====================================================================
# Dict input to the model
# =====================================================================

parameters = {
    'muscle': muscle,
    'time': time_dt,
    'dt': dt,
    'MVC': MVC,
    'yielding': yielding,
    'sag': sag,
    'l_T_slack': l_T_slack,
    'l_M_opt': l_M_opt,
    'alpha_0': alpha_0,
    'l_MT': l_MT,
    'vmax': 10.5428 * l_M_opt,
    'stim_freq': float(fs)
    # optional: 'stim_freq', 'Ca_max_slow', 'Ca_max_fast', 'k1','k2','c1_*','c3_*'
}

states = {'MUAP_0': 0.0, 'Ca_0': 0.0, 'act_0': 1e-9, 'l_M_0': l_M_0, 'y_0': 1.0, 's_0': 1.0,}


###############################################################################
""" PARAMETERS OPTIMIZATION (Nelder-Mead-lstsq)"""
# Decomment section and comment the "Running simulations and plotting solutions" block
###############################################################################

""" 1) MAXIMAL benchmark [MVC, Vmax] estimation """

# def obj(x, parameters, states, Distimes, exp_force): 
    
#     parameters['MVC'] =  x[0]
#     parameters['vmax'] = x[1]
    
#     model = MU_model(parameters, states, Distimes) # Create an model class instance
#     force_sim, _, _, _, _, _, _ = model.run_MT_simulation() # Run the simulation
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [1.2, 10]
# bnds = [(1, 1.5), (8, 12)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("MVC =", res.x[0])
# print("vmax =", res.x[1])

"""  2) SUB-MAXIMAL benchmark [MVC, Ca_max] on 30Hz force and 40Hz for the length trials """
"""  3) Fast muscle benchmark [MVC] estimation on 150 Hz trial """
"""  4) Fast muscle benchmark [Ca_max] estimation on 30 Hz trial"""

# def obj(x, parameters, states, Distimes, exp_force): 
    
#     #parameters['MVC'] =  x[0]
#     parameters['Ca_max'] = x[0]

    
#     model = MU_model(parameters, states, Distimes) # Create an model class instance
#     force_sim, _, _, _, _, _, _ = model.run_MT_simulation() # Run the simulation
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [5e5]
# bnds = [(1e5, 5e6)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# #print("MVC =", res.x[0])
# print("Ca_max =", res.x[0])

"""  5) SUB-MAXIMAL benchmark [k1, k2] on 40Hz length trial """

# def obj(x, parameters, states, Distimes, exp_force): 
    
#     parameters['k1'] =  x[0]
#     parameters['k2'] = x[1]
#     parameters['Ca_max'] = x[2]
    
#     model = MU_model(parameters, states, Distimes) # Create an model class instance
#     force_sim, _, _, _, _, _, _ = model.run_MT_simulation() # Run the simulation
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [11, 17, 188000]
# bnds = [(10, 19), (10, 19), (5e4, 1e6)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("k1 = ", res.x[0])
# print("k2 = ", res.x[1])
# print("Ca_max = ", res.x[2])

""" 6) Ca transient ODE [c1, c2, c3] parameters estimation """

# def obj(x, parameters, states, Distimes, exp_data): 
    
#     parameters['c_1'] = x[0]
#     parameters['c_3'] = x[1]
    
#     model = MU_model(parameters, states, Distimes) # Create an model class instance
#     _, _, Ca, _, _ = model.run_M_simulation() # Run the simulation

#     idx = np.isin(np.round(time_dt,4), np.round((exp_data[:,0]-exp_data[0,0])*1e-3, 4)).nonzero()[0]

#     residuals = Ca[0,idx]*10**6 - exp_data[:,1]  
#     return np.sum(residuals**2)  

# if fibre == 'slow':
#     exp_data = Ca_slow_23
#     x0 = [8*10.**3, 0.6]
#     bnds = [(1e2, 1e5), (0.1, 2)]
# elif fibre == 'fast':
#     exp_data = Ca_fast_35
#     x0 = [2.8*10.**3, 0.7]
#     bnds = [(1e2, 1e4), (0.1, 2)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_data), 
#                 x0, method='Nelder-Mead', bounds=bnds, options={'disp': True, 'maxiter': 500})

# print("Optimized parameters:")
# print("c_1 =", res.x[0])
# print("c_3 =", res.x[1])


# =====================================================================
# Simulation exec & plots
# =====================================================================

model = MU_model(parameters, states, Distimes)

if benchmark in {'max', 'sub', 'len', 'fast'}:
    force_sim, _, Ca, a, l_M, _, _ = model.run_MT_simulation()

    plt.rcParams['figure.dpi'] = 110
    plt.figure(figsize=(8, 4))
    plt.plot(time_dt, force_sim, label='Simulated Force', linewidth=2)
    if exp_force is not None:
        plt.plot(time_dt, exp_force, 'k', label='Exp. Force', linewidth=1.5)
    plt.ylabel('Force [N]', weight='bold', fontsize=12)
    plt.xlabel('Time [s]', weight='bold', fontsize=12)
    plt.title(f'Reconstructed {muscle} force', weight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

elif benchmark == 'ca':
    _, _, Ca, _, _ = model.run_M_simulation()

    plt.rcParams['figure.dpi'] = 110
    plt.figure(figsize=(10, 3))
    if fibre == 'slow':
        plt.plot(time_dt, Ca[0, :] * 1e6, 'g', label='Sim (23°C)')
        plt.plot((Ca_slow_23[:, 0] - Ca_slow_23[0, 0]) * 1e-3, Ca_slow_23[:, 1], 'k--',
                 label='Rincon 2021 (23°C)')
        plt.title(r'Free [$Ca^{2+}$] slow fibres', weight='bold', fontsize=14)
    else:
        plt.plot(time_dt, Ca[0, :] * 1e6, 'g', label='Sim (35°C)')
        plt.plot((Ca_fast_35[:, 0] - Ca_fast_35[0, 0]) * 1e-3, Ca_fast_35[:, 1], 'k--',
                 label='Hollingworth 1996 (35°C)')
        plt.title(r'Free [$Ca^{2+}$] fast fibres', weight='bold', fontsize=14)

    plt.ylabel(r'[$\mu$M]', weight='bold', fontsize=12)
    plt.xlabel('Time [s]', weight='bold', fontsize=12)
    plt.xlim((0, 0.12))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show() 



# # Error metrics (%mAE, %MAE)
# mean_abs_error = np.mean((np.abs(force_sim - exp_force)/MVC)*100)
# max_abs_error = np.max((np.abs(force_sim - exp_force)/MVC)*100)


# Save results

# if save == 1:
#     print("Saving results...")
#     np.save('a_30', a, allow_pickle=True)

