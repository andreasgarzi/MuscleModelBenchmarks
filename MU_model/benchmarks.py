"""
Author: Andrea Sgarzi
Email: a.sgarzi@ad.unsw.edu.au
Affiliation: University of New South Wales (UNSW), Graduate School of Biomedical Engineering (GSBE)

Description:
This script implements a collection of animal-based experimental benchmarks
used to validate and test a motor-unit (MU) driven muscle model.

The benchmarks include:
- Slow and fast muscle contractions
- Isometric and dynamic conditions
- Muscle-scale, motor-unit-scale, and calcium-transient-scale simulations

Experimental datasets are derived from the literature and are used to:
- Configure model parameters and inputs (stimulation, length, force)
- Select active model components (tendon, force-length, force-velocity, yielding, sag)
- Compare simulated outputs against experimental force or calcium traces

This script acts as an interface for reproducing and evaluating
benchmark simulations across different physiological conditions.
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

from MU_model import MU_model, ModelConfig 

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
    elif data_type == 'EDL_FFR':
      skip = 30
    elif data_type == 'EDL_FLR':
      skip = 99
    elif data_type == 'EDL_dyn':
      skip = 190

    data = np.loadtxt(file, delimiter='\t', skiprows=skip)

    return data


def prepare_segment_iso(data_path, data_type, trial, f=1000, dt=1e-4):

    data = read_data(data_path, data_type)

    if data_type == 'EDL_FFR':
       
       freq_map = {30:0, 50:1, 60:2, 70:3, 80:4, 90:5, 100:6, 120:7} # frequency map
       seg_idx = freq_map[trial]

       seg_len = 20000 # Each segment is 20000 samples long
       start = seg_idx * seg_len

       N = 1000 # Each part starts from 0 with 1000 samples
       seg_slice = slice(start, start + N)

    
    elif data_type  == 'EDL_FLR':
       
       l_map = {
          0.25:0, 0.5:1, 0.75:2, 1:3, 1.25:4, 1.5:5, 1.75:6, 2:7,
          2.25:8, 2.5:9, 2.75:10, 3:11, 3.25:12, 3.5:13, 3.75:14, 4:15,
          4.25:16, 4.5:17, 4.75:18, 5:19,
       } # length map
       
       seg_idx = l_map[trial]

       seg_len = 16000 # Each segment is 16000 samples long
       start = seg_idx * seg_len + 1000 # starting from 1000 samples

       N = 1000 # Each part starts from 0 with 1000 samples
       seg_slice = slice(start, start + N)

    t = np.arange(N, dtype=float) / f # original time

    force  = np.asarray(data[seg_slice, 2], dtype=float)
    spikes = np.asarray(data[seg_slice, 11], dtype=float)
    prev = spikes[:-1] # 0-1 transitions
    curr = spikes[1:]
    rising_edges = np.where((prev == 0) & (curr == 1))[0] + 1  # discharges when 1 is preceeded by 0
    spike_times_sec = t[rising_edges]

    t_end = t[-1]
    t_hi = np.arange(0.0, t_end + 1e-12, dt) # interpolated time

    kind = 'cubic' # interp
    # fs, cutoff, order = 1000, 30, 4  # LPF setup
    # b, a = signal.butter(order, cutoff, btype='lowpass', fs=fs) # LPF

    force = force - force[0] # offset force

    # force_filt = signal.filtfilt(b, a, force) # filt force
    force_hi = sp.interpolate.interp1d(t, force, kind=kind)(t_hi) # interp force
    force_hi[force_hi < 0] = 0 # set inferior limit to 0

    return {
        'spike_times_sec': spike_times_sec,
        't_hi': t_hi,
        'force_hi': force_hi,
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
use_yielding = None
use_sag = None
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
    use_FV = True
    type = simpledialog.askstring("Input", "Select fibre type ('S'- slow, 'F' - fast):") # fibre type selection

    if type == 'S':
       
        use_PE = True
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
            Distimes = np.arange(0, t_end, 1/70) # create array of discharge times at 70 Hz
            disp = read_data(path / 'displacement.dat', 'disp_M')
            disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # load displacement
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.32, 17.1, 17.1, 17.1, 6*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) - 2 # Musculo-tendon length (mm)
            l_MT = l_MT_0 + disp*float(trial) # scaled MT length  

            exp_force_path = os.path.join(path, f"force_trial{idx+1}.dat")
            exp_force = read_data(exp_force_path, 'f_M')
            exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='quadratic')(np.arange(0,2,dt)) # interpolate to have equal n of points

            use_yielding = False # yielding not included
            use_sag = False
    
    
        elif benchmark == 'sub': # SUB-MAXIMAL benchmark - SLOW MUSCLE (Perreault 2003) 
    
            path_disp = scale_path / 'slowMuscle_submaximalActivation'
            trial = simpledialog.askstring("Input", "Trial type? ('iso' or 'dyn'):")
            stim = simpledialog.askstring("Input", "Stimulation type? ('c'- constant or 'v'- variable):")
            fs = simpledialog.askstring("Input", "Stimulation frequency (10, 20, 30 Hz):")

            muscle = 'cat_SOL' # 
            use_yielding = True # yielding included
            use_sag = False
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [26.13, 65, 30, 30, 7.5*np.pi/180] # MVC and M/T lengths
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
    
            path = scale_path / 'slowMuscle_length'
            l = simpledialog.askstring("Input", "Length? ('0', '8', or '16'):")
            fs = simpledialog.askstring("Input", '"Stimulation frequency (1, 10, 20, 40 Hz):"')
            use_yielding = False
            use_sag = False
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [30.25, 65, 30, 30, 7.5*np.pi/180] # MVC and M/T lengths
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

        use_PE = False
        path = scale_path / 'fastMuscle'
        benchmark = simpledialog.askstring("Input", "Select benchmark ('FFR'- force-freq. isometric, 'FLR'- force-fre. isometric at different lengths, 'dyn'- dynamic):") # benchmark selection

        if benchmark == 'FFR': # isometric frequency variation

            data_path = path / 'FFR.ddf'
            fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (30, 50, 60, 70, 80, 90, 100, 120):")

            part = prepare_segment_iso(data_path, 'EDL_FFR', trial=float(fs), f=1000, dt=1e-4) 
            Distimes = part['spike_times_sec']
            time_dt = part['t_hi']
            exp_force = part['force_hi']
            
            muscle = 'rat_EDL'
            
            use_yielding = False
            use_sag = False
            
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [2.49, 5, 6.6, 4.73, 10*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) 
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
        
        elif benchmark == 'FLR': # isometric length variation
        
            data_path = path / 'FLR.ddf'
            l = simpledialog.askstring("Input", "Muscle length [mm] (0.25:0.25:4.25):")

            part = prepare_segment_iso(data_path, 'EDL_FLR', trial = float(l), f=1000, dt=1e-4) 
            Distimes = part['spike_times_sec']
            time_dt = part['t_hi']
            exp_force = part['force_hi']
            length = float(l)

            muscle = 'rat_EDL'
            
            use_yielding = False
            use_sag = False

            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [2.49, 5, 6.6, 6.6, 10*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) + length
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array (n+1 points)

        elif benchmark == 'dyn': # dynamic shortening

            trial = simpledialog.askstring("Input", "Displacement amplitude ('short' or 'length'):")
            fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (20, 40, 60, 120):")
            l = simpledialog.askstring("Input", "Norm. muscle length ('0.8', '0.95', '1.1'):")

            exp_force = np.load(path / f"{fs}_{l}_{trial}.npy")
            muscle = 'cat_CF' # dorsi/plantar

            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [15.4, 24.3, 21.84, float(l)*(21.84), 0] # MVC and M/T lengths
            
            use_yielding = False
            use_sag = False
           
            if fs == '120': # maximum stimulation trial
                t_end = 0.16
                Distimes = np.arange(0, t_end, 1/float(fs)) # exp. discharge times
                if trial == 'short': # distinguish between tetanic instantaneous FV curve (max) and subtetanic FV (sub)
                    a = 'max' 
                    mvc_sample = 1083 # last sample before displacement 
                elif trial == 'length':
                    a = 'max'
                    mvc_sample = 1090 # last sample before displacement 
            else:  # sub-maximal stimulation trials
                t_end = 0.17
                Distimes = np.arange(0, 0.15, 1/float(fs)) # exp. discharge times
                a = 'sub' 
                mvc_sample = 1119 # last sample before displacement 

            time_dt = np.arange(0, t_end, dt)
            exp_force = exp_force[0:len(time_dt)]

            disp = np.load(path / f"disp_{a}_{trial}_interp.npy") # load displacement
            l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0) 
            l_MT = np.empty((len(time_dt)+1), dtype=object) # full MT length array
            l_MT[0:-1] = np.ones((len(time_dt)), dtype=object)*l_MT_0 + (disp[0:len(time_dt)]*l_M_opt)*np.cos(alpha_0)     
  

if scale == 'MU': # motor-unit benchmarks

    use_FV = False
    use_PE = True
    scale_path = base_path / 'Motorunit'
    type = simpledialog.askstring("Input", "Select fibre type ('S'- slow, 'FF'- fast fatiguable, 'FR'- fast fatigue resistent):") # fibre type selection

    if type == 'S': # slow MU

        path = scale_path / 'MU_S'
        use_yielding = False
        use_sag = False
        
        fs = simpledialog.askstring("Input", "Stimulation type ('twitch', 'unfused', 'tetanus'):")
        exp_force = np.load(path / f"MU_S_{fs}.npy")
        
        if fs == 'twitch':
            Distimes = np.round([0], 3)
        elif fs == 'unfused':
            Distimes = np.arange(0, 17*0.0813, 0.0813) # 12 Hz ish
        elif fs == 'tetanus':
            Distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz
        
        t_end = 1.8
        muscle = 'cat_SOL' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.04, 0, 20, 20, 9.2*np.pi/180] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.full(len(time_dt), float(l_MT_0)) # full MT length array

    elif type == 'FF': # fast fatiguable MU

        path = scale_path / 'MU_FF'
        use_yielding = False
        use_sag = True
        
        fs = simpledialog.askstring("Input", "Stimulation type ('twitch', 'unfused', 'tetanus'):")
        exp_force = np.load(path / f"MU_FF_{fs}.npy")
        
        if fs == 'twitch':
            Distimes = np.round([0],3)
        elif fs == 'unfused':
            Distimes = np.arange(0, 20*0.04, 0.04) # 25 Hz
        elif fs == 'tetanus':
            Distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz

        t_end = 1
        muscle = 'cat_GM' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.81, 0, 20, 20, 9.2*np.pi/180] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.full(len(time_dt), float(l_MT_0)) # full MT length array
 
    elif type == 'FR': # fast fatigue resistent MU

        path = scale_path / 'MU_FR'
        use_yielding = False
        use_sag = True
        
        fs = simpledialog.askstring("Input", "Stimulation ('twitch', 'unfused', 'tetanus') or ('25', '30', '35', '40', '150') Hz:")
        exp_force = np.load(path / f"MU_FR_{fs}.npy")
        
        if fs == 'twitch':
            Distimes = np.round([0],3) # twitch (Burke 1974)
            t_end = 1.2
            muscle = 'cat_GM' # dorsi/plantar
        elif fs == 'unfused':
            Distimes = np.arange(0, 17*0.05, 0.05) # 20 Hz (Burke 1974)
            t_end = 1.2
            muscle = 'cat_GM' # dorsi/plantar
        elif fs == 'tetanus':
            Distimes = np.arange(0, 13*0.025, 0.025) # 40 Hz (Burke 1974)
            t_end = 1.2
            muscle = 'cat_GM' # dorsi/plantar
        else: 
            Distimes = np.load(path / f"MU_FR_{fs}_times.npy") # (Chelichowski 1999)
            t_end = 0.7
            muscle = 'rat_GM' # dorsi/plantar

        time_dt = np.arange(0, t_end, dt)
        if muscle == 'cat_GM':
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.39, 0, 20, 20, 9.2*np.pi/180] # MVC and M/T lengths
        elif muscle == 'rat_GM':
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.078, 0, 20, 20, 9.2*np.pi/180] # MVC and M/T lengths

        l_MT_0 = l_T_slack + (l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.full(len(time_dt), float(l_MT_0)) # full MT length array

    if muscle != 'rat_GM': # for rat GM, exp. force is already interpolated to time_dt
        time_exp = np.arange(0, exp_force[-1,0], dt)
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='linear')(time_exp)
        target_len = len(time_dt)
        if len(exp_force) < target_len:
            n_pad = target_len - len(exp_force)
            exp_force = np.concatenate([exp_force, np.zeros(n_pad)])

elif scale == 'Ca': # Test Ca dynamics (Hollingworth, Rincon exp. data)
    
    type = simpledialog.askstring("Input", "Fibre type? ('slow', 'fast'):")
    use_yielding = False
    use_sag = False
    use_PE = False
    use_FV = False
    
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

l_MT = np.asarray(l_MT, dtype=float).ravel() # make sure is numeric

parameters = {
    'muscle': muscle,
    'scale': scale,
    'time': time_dt,
    'dt': dt,
    'MVC': MVC,
    'l_T_slack': l_T_slack,
    'l_M_opt': l_M_opt,
    'alpha_0': alpha_0,
    'l_MT': l_MT,
    'vmax': 10,
}

states = {'MUAP_0': 0.0, 'Ca_0': 0.0, 'act_0': 1e-9, 'l_M_0': l_M_0, 'y_0': 1.0, 's_0': 1.0,}

# build model_config for the new model
# tendon is "on" when you provide a real tendon slack length (>0); MU benchmarks set l_T_slack=0 so this becomes False
use_tendon = bool(l_T_slack is not None and float(l_T_slack) > 0)

model_config = ModelConfig(
    use_tendon=use_tendon,
    use_PE=use_PE,
    use_FL=True,
    use_FV=use_FV,
    use_yielding=use_yielding,  
    use_sag=use_sag,
)


###############################################################################
""" PARAMETERS OPTIMIZATION (Nelder-Mead-lstsq)"""
# Decomment section and comment the "Running simulations and plotting solutions" block
###############################################################################

"""  1) MSisof [MVC, Ca_max, k1, k2] on 30Hz isometric trial (Perreault 2003) """

# def obj(x, parameters, states, Distimes, exp_force, model_config): 
    
#     parameters['MVC'] =  x[0]
#     parameters['Ca_max_slow'] = x[1]
#     parameters['k1_s'] =  x[2]
#     parameters['k2_s'] = x[3]

#     model = MU_model(parameters, states, Distimes) # Create an model class instance
#     out = model.run(output_force=True)
#     force_sim = out["force"]
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [27, 5e5, 10, 14]
# bnds = [(25, 30), (1e5, 1e6), (10, 20), (10, 20)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force, model_config), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("MVC =", res.x[0])
# print("Ca_max =", res.x[1])
# print("k1_s =", res.x[2])
# print("k2_s =", res.x[3])

""" 2a) MSdyn2 [MVC] on 0.05 mm dynamic trial (Krylow 1997)
    2b) MSdyn2 [af] on 2.00 mm dynamic trial (Krylow 1997)
    2c) MSisol [MVC] on 40 Hz isometric trial (Perreault 2003, digitized from Kim 2015)"""

# def obj(x, parameters, states, Distimes, exp_force, model_config): 
    
#     parameters['MVC'] = x[0]
#     # parameters['af_s'] = x[0]
    
#     model = MU_model(parameters, states, Distimes, model_config) # Create an model class instance
#     out = model.run(output_force=True)
#     force_sim = out["force"]
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [28] # 1.2 N, 0.4, 27 N
# bnds = [(25, 31)] # (1.1, 1.5), (0.1, 1), (26, 30)

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force, model_config), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("MVC =", res.x[0])
# #print("af_s =", res.x[0])

"""  5) SUB-MAXIMAL benchmark [k1, k2] on 40Hz length trial """

# def obj(x, parameters, states, Distimes, exp_force, model_config): 
    
#     parameters['k1'] =  x[0]
#     parameters['k2'] = x[1]
#     parameters['Ca_max'] = x[2]
    
#     model = MU_model(parameters, states, Distimes, model_config) # Create an model class instance
#     out = model.run(output_force=True)
#     force_sim = out["force"]
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [11, 17, 188000]
# bnds = [(10, 19), (10, 19), (5e4, 1e6)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force, model_config), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("k1 = ", res.x[0])
# print("k2 = ", res.x[1])
# print("Ca_max = ", res.x[2])

"""  6) MFisof [Ca_max, k1, k2] on 80 Hz isometric trial (Millard new exp. data) """

#def obj(x, parameters, states, Distimes, exp_force, model_config): 
    
    #parameters['Ca_max_f_M'] = x[0]
    #parameters['k1_f_M'] =  x[1]
    #parameters['k2_f_M'] = x[2]
#    states['l_M_0'] = x[0]
#    l_MT_0 = l_T_slack + x[0]*np.cos(alpha_0) 
#    parameters['l_MT'] = np.ones((len(time_dt)+1), dtype=object)*l_MT_0

#    model = MU_model(parameters, states, Distimes, model_config) # Create an model class instance
#    out = model.run(output_force=True)
#    force_sim = out["force"]
    
#    residuals = force_sim - exp_force  
#    return np.sum(residuals**2)  

#x0 = [5e5, 10, 10]
#bnds = [(1e5, 1e6), (10, 15), (10, 15)]

#x0 = [3]
#bnds = [(1, 6.6)]

#res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force, model_config), 
#               x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

#print("Optimized parameters:")
#print("Ca_max_f_M =", res.x[0])
#print("k1_f_M =", res.x[1])
#print("k2_f_M =", res.x[2])
#print("l_M_0 =", res.x[0])

"""  7) MFisodyn [af] on -3l0/s shortening trial at 120 Hz (Brown et al. 1999) """

#def obj(x, parameters, states, Distimes, exp_force, model_config): 
    
#    parameters['af_f'] = x[0]

#    model = MU_model(parameters, states, Distimes, model_config) # Create an model class instance
#    out = model.run(output_force=True)
#    force_sim = out["force"]
    
#    residuals = force_sim[mvc_sample:]/force_sim[mvc_sample] - exp_force[mvc_sample:] # only when displacement is applied
#    return np.sum(residuals**2)  

#x0 = [0.4]
#bnds = [(0.1, 3)]

#res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force, model_config), 
#            x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

#print("Optimized parameters:")
#print("af_f =", res.x[0])

"""  8) MUisof [MVC, Ca_max, k1, k2] on tetanic isometric trial (Burke exp. data) """

# def obj(x, parameters, states, Distimes, exp_force, model_config): 
    
#     parameters['MVC'] =  x[0]
#     parameters['Ca_max_f_MU'] = x[1]
#     parameters['k1_f_MU'] =  x[2]
#     parameters['k2_f_MU'] = x[3]

#     model = MU_model(parameters, states, Distimes, model_config) # Create an model class instance
#     out = model.run(output_force=True)
#     force_sim = out["force"]
    
#     residuals = force_sim[0:len(exp_force)] - exp_force  
#     return np.sum(residuals**2)  

# x0 = [0.35, 5e5, 10, 10]
# bnds = [(0.3, 0.4), (1e5, 1e6), (2, 15), (2, 15)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force, model_config), 
#               x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("MVC =", res.x[0])
# print("Ca_max_f_MU =", res.x[1])
# print("k1_f_MU =", res.x[2])
# print("k2_f_MU =", res.x[3])

"""  9) MUisof [As, Ts] on unfused tetanus isometric trial (Burke exp. data) """

# def obj(x, parameters, states, Distimes, exp_force, model_config): 
    
#     parameters['As_peak'] =  x[0]
#     parameters['As_decay'] =  x[1]
#     parameters['Ts'] = x[2]

#     model = MU_model(parameters, states, Distimes, model_config) # Create an model class instance
#     out = model.run(output_force=True)
#     force_sim = out["force"]
    
#     residuals = force_sim[0:len(exp_force)] - exp_force  
#     return np.sum(residuals**2)  

# x0 = [1.2, 0.9, 0.1]
# bnds = [(1, 2), (0.1, 0.9), (0.01, 0.9)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force, model_config), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("As_peak =", res.x[0])
# print("As_decay =", res.x[1])
# print("Ts =", res.x[2])


""" 10) Ca transient ODE [c1, c2, c3] parameters estimation """

# def obj(x, parameters, states, Distimes, exp_data, model_config): 
    
#     parameters['c1_f'] = x[0]
#     parameters['c2_f'] = x[1]
#     parameters['c3_f'] = x[2]
    
#     model = MU_model(parameters, states, Distimes, model_config) # Create an model class instance
#     out = model.run(output_force=False)  
#     Ca = out["Ca"]

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

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_data, model_config), 
#                 x0, method='Nelder-Mead', bounds=bnds, options={'disp': True, 'maxiter': 500})

# print("Optimized parameters:")
# print("c1 =", res.x[0])M
# parameters['c1_f'] = res.x[0]
# print("c2 =", res.x[1])
# parameters['c2_f'] = res.x[1]
# print("c3 =", res.x[2])
# parameters['c3_f'] = res.x[2]


# =====================================================================
# Simulation & plots
# =====================================================================

model = MU_model(parameters, states, Distimes, model_config)

if scale == 'M':

    out = model.run(output_force=True)  
    force_sim = out["force"]
    Ca = out["Ca"]
    act = out["act"]
    l_M = out["l_M"]

    plt.figure(figsize=(8, 4))
    if type == 'F' and benchmark == 'dyn' and a == 'max':
        force_sim = force_sim/force_sim[mvc_sample]
        plt.plot(time_dt, force_sim, label='Simulated Force', linewidth=2) # normalised once in Brown 1999
        plt.plot(time_dt, exp_force, 'k', label='Exp. Force', linewidth=1.5)
    elif type == 'F' and benchmark == 'dyn' and a == 'sub':
        force_sim = force_sim/force_sim[mvc_sample]
        exp_force = exp_force/exp_force[mvc_sample]
        plt.plot(time_dt, force_sim, label='Simulated Force', linewidth=2) # normalised once in Brown 1999
        plt.plot(time_dt, exp_force, 'k', label='Exp. Force', linewidth=1.5)
    else:
        plt.plot(time_dt, force_sim, label='Simulated Force', linewidth=2)
        plt.plot(time_dt, exp_force, 'k', label='Exp. Force', linewidth=1.5)

    plt.ylabel('Force [N]', fontsize=12)
    plt.xlabel('Time [s]', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

elif scale == 'MU': # MU

    out = model.run(output_force=True)  
    force_sim = out["force"]

    plt.figure(figsize=(8, 4))
    plt.plot(time_dt, force_sim, label='Simulated Force', linewidth=2)
    plt.plot(time_dt, exp_force, 'k', label='Exp. Force', linewidth=1.5)
    plt.ylabel('Force [N]', fontsize=12)
    plt.xlabel('Time [s]', fontsize=12)
    plt.title(f'Reconstructed {muscle} force', weight='bold')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

elif scale == 'Ca':

    out = model.run(output_force=False)  
    Ca = out["Ca"]

    plt.figure(figsize=(5, 3), dpi=300)
    if type == 'slow':
        plt.plot(time_dt, Ca * 1e6, 'g', label='Simulated')
        plt.plot((Ca_slow_23[:, 0] - Ca_slow_23[0, 0]) * 1e-3, Ca_slow_23[:, 1], 'k--',
                 label='Experimental (23°C)')
        plt.ylim([0,20])
    else:
        plt.plot(time_dt, Ca * 1e6, 'g', label='Simulated')
        plt.plot((Ca_fast_35[:, 0] - Ca_fast_35[0, 0]) * 1e-3, Ca_fast_35[:, 1], 'k--',
                 label='Experimental (35°C)')
        plt.ylim([0,20])

    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel(r'[$Ca^{2+}$] [$\mu$M]', fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim((0, 0.15))
    plt.grid()
    plt.tight_layout()
    plt.show()


# Save results
#os.chdir(r'C:\Users\z5517249\Dropbox\UNSW_Andrea_Luca_PhD\Code\Python_Scripts\Results_benchmarks\MU\sim')

#np.save('fast_unfused', force_sim, allow_pickle=True)

#os.chdir(r'C:\Users\z5517249\Dropbox\UNSW_Andrea_Luca_PhD\Code\Python_Scripts\Results_benchmarks\fast_M\exp')

#np.save('dyn_60_0.95_length', exp_force, allow_pickle=True)
