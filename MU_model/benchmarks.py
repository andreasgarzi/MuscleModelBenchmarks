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
from scipy import signal
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog
from scipy.optimize import minimize

from MU_model import MU_model  

# =====================================================================
# Utils
# =====================================================================

def read_data(file, data_type):

    if data_type == 'disp_M':  # maximal benchmark
      skip = 16
    elif data_type == 'disp_SM': # submaximal benchmark
      skip = 18
    elif data_type == 'f_M': # maximal benchmark
      skip = 15
    elif data_type == 'f_SM': # submaximal benchmark (dynamic trials)
      skip = 18
    elif data_type == 'f_iso_SM': # submaximal benchmark (isometric trials)
      skip = 19
    elif data_type == 'Millard_EDL_FFR':
      skip = 30
    elif data_type == 'Millard_EDL_FLR':
      skip = 99
    elif data_type == 'Millard_EDL_dyn':
      skip = 190

    data = np.loadtxt(file, delimiter='\t', skiprows=skip)

    return data


def prepare_segment_iso(data_path, data_type, trial, f=1000, dt=1e-4):

    data = read_data(data_path, data_type)

    if data_type == 'Millard_EDL_FFR':
       
       freq_map = {30:0, 50:1, 60:2, 70:3, 80:4, 90:5, 100:6, 120:7} # frequency map
       seg_idx = freq_map[trial]

       seg_len = 20000 # Each segment is 20000 samples long
       start = seg_idx * seg_len

       N = 1000 # Each part starts from 0 with 1000 samples
       seg_slice = slice(start, start + N)

    
    elif data_type  == 'Millard_EDL_FLR':
       
       l_map = {
          0.25:0, 0.5:1, 0.75:2, 1:3, 1.25:4, 1.5:5, 1.75:6, 2:7,
          2.25:8, 2.5:9, 2.75:10, 3:11, 3.25:12, 3.5:13, 3.75:14, 4:15,
          4.25:16, 4.5:17, 4.75:18, 5:19, 5.25:20, 5.5:21, 5.75:22, 6:23,
          6.25:24, 6.5:25, 6.75:26, 7:27, 7.25:28, 7.5:29, 7.75:30, 8:31,
          8.25:32, 8.5:33, 8.75:34, 9:35, 9.25:36,
       } # length map
       
       seg_idx = l_map[trial]

       seg_len = 16000 # Each segment is 16000 samples long
       start = seg_idx * seg_len + 1000 # starting from 1000 samples

       N = 1000 # Each part starts from 0 with 1000 samples
       seg_slice = slice(start, start + N)

    t = np.arange(N, dtype=float) / f # original time

    length = np.asarray(data[seg_slice, 1], dtype=float) 
    force  = np.asarray(data[seg_slice, 4], dtype=float)
    spikes = np.asarray(data[seg_slice, 11], dtype=float)
    prev = spikes[:-1] # 0-1 transitions
    curr = spikes[1:]
    rising_edges = np.where((prev == 0) & (curr == 1))[0] + 1  # discharges when 1 is preceeded by 0
    spike_times_sec = t[rising_edges]

    t_end = t[-1]
    t_hi = np.arange(0.0, t_end + 1e-12, dt) # interpolated time

    kind = 'cubic' # interp
    fs, cutoff, order = 1000, 10, 4  # LPF setup
    b, a = signal.butter(order, cutoff, btype='lowpass', fs=fs) # LPF

    length = length - length[0] # offset length

    length_filt = signal.filtfilt(b, a, length) # filt length
    length_hi = sp.interpolate.interp1d(t, length_filt, kind=kind)(np.arange(0,t_end+dt,dt)) # interp length with one more point

    force = force - force[0] # offset force

    force_filt = signal.filtfilt(b, a, force) # filt force
    force_hi = sp.interpolate.interp1d(t, force_filt,  kind=kind)(t_hi) # interp force

    return {
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
scale = simpledialog.askstring("Input", "Select muscle scale ('M'- muscle, 'MU' - motor unit, 'Ca' - calcium transients):") # muscle scale selection

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


if scale == 'M': # muscle benchmarks

    scale_path = base_path / 'Muscle'
    type = simpledialog.askstring("Input", "Select fibre type ('S'- slow, 'F' - fast):") # fibre type selection

    if type == 'S':
       
        benchmark = simpledialog.askstring("Input", "Select benchmark ('max'- max. activation, 'sub'- submax. activation, 'len'- length dependency):") # benchmark selection

        if benchmark == 'max':  # MAXIMAL benchmark - SLOW MUSCLE (Sandercock 1997)
           
            path = scale_path / 'slowMuscle_maximalActivation'
            trial = simpledialog.askstring("Input", "Amplitude displacement [mm] (e.g., 0.05, 0.1, 0.25, 0.5, 1, 2):")
            amp = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0], dtype=float)
            idx_arr = np.where(np.isclose(amp, float(trial), rtol=0, atol=1e-9))[0]
            idx = int(idx_arr[0]) 

            t_end = 2
            muscle = 'rat_SOL' # rat soleus
            time_dt = np.arange(0, t_end, dt) # time for BB
            Distimes = np.arange(0, t_end, 1/70) # create array of dischare times at 70 Hz
            disp = read_data(path / 'displacement.dat', 'disp_M')
            disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # load displacement
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.36, 17.1, 17.1, 17.1, 6*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) - 2 # Musculo-tendon length (mm)
            l_MT = l_MT_0 + disp*float(trial) # scaled MT length  

            exp_force_path = os.path.join(path, f"force_trial{idx+1}.dat")
            exp_force = read_data(exp_force_path, 'f_M')
            exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='quadratic')(np.arange(0,2,dt)) # interpolate to have equal n of points

            yielding = 0 # yielding not included
            sag = 0
    
    
        elif benchmark == 'sub': # SUB-MAXIMAL benchmark - SLOW MUSCLE (Perreault 2003) 
    
            path_disp = scale_path / 'slowMuscle_submaximalActivation'
            trial = simpledialog.askstring("Input", "Trial type? ('iso' or 'dyn'):")
            stim = simpledialog.askstring("Input", "Stimulation type? ('c'- constant or 'v'- variable):")
            fs = simpledialog.askstring("Input", "Stimulation frequency (10, 20, 30 Hz):")

            muscle = 'cat_SOL' # 
            yielding = 1 # yielding included
            sag = 0
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [26.9, 65, 30, 30, 7.5*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) - 4  # Musculo-tendon length (mm)
    
            t_end = 2
            time_dt = np.arange(0, t_end, dt) # time for BB

            path = path_disp / f"{stim}_freq"

            Distimes = np.load(path / f"{fs}_{stim}_times.npy")  # sec
    
            if trial == 'iso':
              
                l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # constant MT length
                exp_force = read_data(path / f"force_isometric_{stim}{fs}.dat", 'f_iso_SM')
                exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))
    
            elif trial == 'dyn':
        
                d = simpledialog.askstring("Input", "Displacement amplitude ('1' or '8' mm):")
        
                disp = read_data(path_disp / f"displacement_{d}.dat", 'disp_SM') # read displacement
                disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate displacement
                l_MT = l_MT_0 + (disp + 8) # scaled MT length
                exp_force = read_data(path / f"force_{stim}{fs}_{d}.dat", 'f_SM') # read exp force
                exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt)) # interpolate exp force


        elif benchmark == 'len': # TWITCH, SUB-TETANIC, TETANIC - SLOW MUSCLE at different lengths (Kim 2015)
    
            path = base_path / 'slowMuscle_length'
            l = simpledialog.askstring("Input", "Length? ('0', '8', or '16'):")
            fs = simpledialog.askstring("Input", '"Stimulation frequency (1, 10, 20, 40 Hz):"')
            yielding = 0
            sag = 0
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [30, 65, 30, 30, 7.5*np.pi/180] # MVC and M/T lengths
            t_end = 1.4
            muscle = 'cat_SOL' # dorsi/plantar
            time_dt = np.arange(0, t_end, dt) # time for BB
    
            if l == '0': # displacement cases (centered at -8 mm from physiological length, see paper)
                d = 8
            elif l == '8':
                d = 0 
            elif l == '16':
                d = -8
        
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) - 4 # Musculo-tendon length (mm)
            l_MT_0 = l_MT_0 + d  # apply displacement
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array

            exp_force = np.load(path / f"{l}_{fs}_interp.npy") # exp. force
            Distimes = np.load(path / f"{l}_{fs}_times.npy") # exp. discharge times

            if fs == '1': # twitch
                Distimes = np.round(Distimes, 3)
         

    elif type == 'F': # Test fast muscle (Millard exp data)

        path = base_path / 'fastMuscle_Millard'
        benchmark = simpledialog.askstring("Input", "Select benchmark ('FFR'- force-freq. isometric, 'FLR'- force-fre. isometric at different lengths, 'short'- dynamic shortening, 'len'- dynamic lengthening):") # benchmark selection

        if benchmark == 'FFR': # isometric frequency variation

            data_path = path / 'FFR.ddf'
            fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (30, 50, 60, 70, 80, 90, 100, 120):")

            part = prepare_segment_iso(data_path, 'Millard_EDL_FFR', trial=float(fs), f=1000, dt=1e-4) 
            Distimes = part['spike_times_sec']
            time_dt = part['t_hi']
            exp_force = part['force_hi']
            muscle = 'rat_EDL'
            
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.5, 9, 13.7, 13.7, 10*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
        
        elif benchmark == 'FLR': # isometric length variation
        
            data_path = path / 'FLR.ddf'
            l = simpledialog.askstring("Input", "Muscle length [mm] (0.25:0.25:9.25):")

            part = prepare_segment_iso(data_path, 'Millard_EDL_FLR', trial = float(l), f=1000, dt=1e-4) 
            Distimes = part['spike_times_sec']
            time_dt = part['t_hi']
            exp_force = part['force_hi']
            length = part['length_hi'] # has one more point
            muscle = 'rat_EDL'
            
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.5, 9, 13.7, 13.7, 10*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)
            l_MT = l_MT_0 + length # full MT length array (n+1 points)

        # elif benchmark == 'short': # dynamic shortening
           


        # elif benchmark == 'len': # dynamic lengthening
           
  

if scale == 'MU': # motor-unit benchmarks

    scale_path = base_path / 'Motorunit'
    type = simpledialog.askstring("Input", "Select fibre type ('S'- slow, 'FF'- fast fatiguable, 'FR'- fast fatigue resistent):") # fibre type selection

    if type == 'S': # slow MU

        path = scale_path / 'MU_S'
        yielding = 0
        sag = 0
        
        fs = simpledialog.askstring("Input", "Stimulation type ('twitch', 'unfused', 'tetanus'):")
        exp_force = np.load(path / f"MU_S_{fs}.npy")
        
        if fs == 'twitch':
            Distimes = np.round([0], 3)
        elif fs == 'unfused':
            Distimes = np.arange(0, 19*0.08, 0.08) # 12 Hz ish
        elif fs == 'tetanus':
            Distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz
        
        t_end = 2
        muscle = 'cat_SOL' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.048, 65, 20, 20, 20*np.pi/180] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array

    elif type == 'FF': # fast fatiguable MU

        path = scale_path / 'MU_FF'
        yielding = 0
        sag = 1
        
        fs = simpledialog.askstring("Input", "Stimulation type ('twitch', 'unfused', 'tetanus'):")
        exp_force = np.load(path / f"MU_FF_{fs}.npy")
        
        if fs == 'twitch':
            Distimes = np.round([0],3)
        elif fs == 'unfused':
            Distimes = np.arange(0, 21*0.04, 0.04) # 25 Hz
        elif fs == 'tetanus':
            Distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz

        t_end = 1
        muscle = 'cat_GM' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.756, 65, 20, 20, 20*np.pi/180] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
 
    elif type == 'FR': # fast fatigue resistent MU

        path = scale_path / 'MU_FR'
        yielding = 0
        sag = 1
        
        fs = simpledialog.askstring("Input", "Stimulation type ('twitch', 'unfused', 'tetanus'):")
        exp_force = np.load(path / f"MU_FR_{fs}.npy")
        
        if fs == 'twitch':
            Distimes = np.round([0],3)
        elif fs == 'unfused':
            Distimes = np.arange(0, 17*0.05, 0.05) # 25 Hz
        elif fs == 'tetanus':
            Distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz 

        t_end = 1
        muscle = 'cat_GM' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.214, 65, 20, 20, 20*np.pi/180] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array


elif scale == 'Ca': # Test Ca dynamics (Hollingworth, Rincon exp. data)
    
    type = simpledialog.askstring("Input", "Fibre type? ('slow', 'fast'):")
    yielding = 0
    sag = 0
    
    path = base_path / 'Ca_transients'
    t_end = 1.4
    time_dt = np.arange(0, t_end, dt)
    
    if type == 'slow':

        Ca_slow_23 = pd.read_csv(path / "Ca_slow_23_100Hz.csv", delimiter=' ').to_numpy()

        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1, 0, 30, 30, 0] # at optimal sarcomere length (assumption)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array 
        T = 1/102 # adjusted from paper
        Distimes = np.arange(0, 0.04, T)
        muscle = 'rat_SOL' # slow fibre muscle

    elif type == 'fast':

        Ca_fast_35 = pd.read_csv(path / "Ca_fast_35_125Hz.csv", delimiter=' ').to_numpy()

        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1.6, 0, 30, 1.6*30, 0] # at longer sarcomere length (see article)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array
        T = 1/125 # from paper
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
    
#     parameters['c1_fast'] = x[0]
#     parameters['c2_fast'] = x[1]
#     parameters['c3_fast'] = x[2]
    
#     model = MU_model(parameters, states, Distimes) # Create an model class instance
#     _, _, Ca, _, _ = model.run_M_simulation() # Run the simulation

#     idx = np.isin(np.round(time_dt,4), np.round((exp_data[:,0]-exp_data[0,0])*1e-3, 4)).nonzero()[0]

#     residuals = Ca[idx]*10**6 - exp_data[:,1]  
#     return np.sum(residuals**2)  

# if fibre == 'slow':
#     exp_data = Ca_slow_23
#     x0 = [6.029e3, 1.8e5, 0.54]
#     bnds = [(1e3, 1e5), (1e5, 1e6), (0.1, 2)]
# elif fibre == 'fast':
#     exp_data = Ca_fast_35
#     x0 = [2.4e3, 4.3e5, 0.6]
#     bnds = [(1e3, 1e5), (1e5, 1e6), (0.1, 2)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_data), 
#                 x0, method='Nelder-Mead', bounds=bnds, options={'disp': True, 'maxiter': 500})

# print("Optimized parameters:")
# print("c1 =", res.x[0])
# parameters['c1_fast'] = res.x[0]
# print("c2 =", res.x[1])
# parameters['c2_fast'] = res.x[1]
# print("c3 =", res.x[2])
# parameters['c3_fast'] = res.x[2]


# =====================================================================
# Simulation & plots
# =====================================================================

model = MU_model(parameters, states, Distimes)

if scale in {'M', 'MU'}:

    force_sim, _, Ca, a, l_M, _, _ = model.run_FLV_simulation()

    plt.figure(figsize=(8, 4))
    plt.plot(time_dt, force_sim, label='Simulated Force', linewidth=2)
    if scale == 'M':
        plt.plot(time_dt, exp_force, 'k', label='Exp. Force', linewidth=1.5)
    else:
        plt.plot(exp_force[:,0], exp_force[:,1], 'k', label='Exp. Force', linewidth=1.5)
    plt.ylabel('Force [N]', fontsize=12)
    plt.xlabel('Time [s]', fontsize=12)
    plt.title(f'Reconstructed {muscle} force', weight='bold')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

elif scale == 'Ca':

    _, _, Ca, _, _ = model.run_FL_simulation()

    plt.figure(figsize=(5, 3))
    if type == 'slow':
        plt.plot(time_dt, Ca * 1e6, 'g', label='Sim (23°C)')
        plt.plot((Ca_slow_23[:, 0] - Ca_slow_23[0, 0]) * 1e-3, Ca_slow_23[:, 1], 'k--',
                 label='Rincon 2021 (23°C)')
        #plt.title(r'Free [$Ca^{2+}$] slow fibres', weight='bold', fontsize=14)
        plt.ylim([0,20])
    else:
        plt.plot(time_dt, Ca * 1e6, 'g', label='Sim (35°C)')
        plt.plot((Ca_fast_35[:, 0] - Ca_fast_35[0, 0]) * 1e-3, Ca_fast_35[:, 1], 'k--',
                 label='Hollingworth 1996 (35°C)')
        #plt.title(r'Free [$Ca^{2+}$] fast fibres', weight='bold', fontsize=14)
        plt.ylim([0,20])

    #plt.ylabel(r'[$Ca^{2+}$] [$\mu$M]', fontsize=12)
    plt.xlabel('Time [s]', fontsize=12)
    plt.xlim((0, 0.12))
    plt.grid()
    #plt.legend()
    plt.tight_layout()
    plt.show() 



# # Error metrics (%mAE, %MAE)
# mean_abs_error = np.mean((np.abs(force_sim - exp_force)/MVC)*100)
# max_abs_error = np.max((np.abs(force_sim - exp_force)/MVC)*100)


# Save results

# if save == 1:
#     print("Saving results...")
#     np.save('a_30', a, allow_pickle=True)

