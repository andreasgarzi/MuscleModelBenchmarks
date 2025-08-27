"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Created on Tue Aug 19 17:54:16 2025
___________________________________

Animal benchmarks for testing MN-driven model based on Caillet et al. 2023 for simulating isometric and dynamic muscle contractions.

"""
import os
import scipy as sp
import numpy as np
import tkinter as tk
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from tkinter import simpledialog
import matplotlib.pyplot as plt
cwd = os.getcwd()


from MN_driven_model import MN_driven_model

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
    elif data_type == 'Millard_EDL':
      skip = 30

    data = np.loadtxt(file, delimiter='\t', skiprows=skip)

    return data

# os.chdir(r"C:\Users\z5517249\Dropbox\UNSW_Andrea_Luca_PhD\Code\Python_Scripts\pyHatze\Caillet2023-MN-driven-model")

# exp_path = '..\\biologicalBenchmark\\'
# exp_path = exp_path + 'fastMuscle_Millard\\'
# data = read_data(exp_path + 'FFR.ddf', 'Millard_EDL')

def prepare_segment(data, fs=1000, target_freq=30, dt=1e-4):

    freq_map = {30:0, 50:1, 60:2, 70:3, 80:4, 90:5, 100:6, 120:7} # Mappa frequenza → indice segmento (0..7)
    seg_idx = freq_map[target_freq]

    seg_len = 20000 # Ogni segmento è lungo 20000 campioni, a partire da 0
    start = seg_idx * seg_len

    N = 1000 # Considera la parte come se iniziasse da 0: prendi i primi 1000 campioni di quel segmento
    seg_slice = slice(start, start + N)

    length = np.asarray(data[seg_slice, 1], dtype=float) # Colonne (0-based): lunghezza=1, forza=2, spikes=11
    force  = np.asarray(data[seg_slice, 2], dtype=float)
    spikes = np.asarray(data[seg_slice, 11], dtype=float)

    t = np.arange(N, dtype=float) / fs # Tempo originale (0..(N-1))/fs

    prev = spikes[:-1] # Trova transizioni 0 -> 1 negli spike (dentro ai 1000 campioni) # Ignora il primissimo campione (non ha precedente)
    curr = spikes[1:]
    rising_edges = np.where((prev == 0) & (curr == 1))[0] + 1  # +1: indice del campione "1"
    spike_times_sec = t[rising_edges]

    t_end = t[-1]
    t_hi = np.arange(0.0, t_end + 1e-12, dt) # Interpolazione cubica su griglia a dt=1e-4

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

# part = prepare_segment(data, fs=1000, target_freq=70, dt=1e-4)
# plt.plot(part['t_hi'], part['length_hi'])


MN_pool = 1  # n. of theoretical MUs in the real pool 
Nr = 1 # n. of (exp.) MUs to represent in the pool
pool = 'n' # exp or pool
spread = 'evenly' # evenly or identified 
species = 'animal' # human/animal

dt = 1e-4 # time step (x-data)

exp_path = '..\\biologicalBenchmark\\'

save = 'n' # save results 'y' or 'n'

root = tk.Tk() # Initialise input window
root.withdraw()  # Hide the root window

benchmark = simpledialog.askstring("Input", "Select benchmark ('max', 'sub', 'len', 'fast', 'Ca'):") 


if benchmark == 'max': # MAXIMAL benchmark - SLOW MUSCLE (Sandercock 1997) """
    
    scale = simpledialog.askstring("Input", "Amplitude displacement scale (e.g., 0.05, 0.1, 0.25, 0.5, 1, 2):")
    scales = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0], dtype=float)
    idx_arr = np.where(np.isclose(scales, float(scale), rtol=0, atol=1e-9))[0]
    idx = int(idx_arr[0]) 

    exp_path = exp_path + 'slowMuscle_maximalActivation\\'

    t_end = 2
    muscle = 'SOL' # dorsi/plantar
    time_dt = np.arange(0, t_end, dt) # time for BB
    Distimes = np.arange(0, t_end, 1/70) # create array of dischare times at 70 Hz
    disp = read_data(exp_path + 'displacement.dat', 'displacement_M')
    disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # load displacement
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.36, 17.1, 17.1, 1, 6*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0) - 2 # Musculo-tendon length (mm)
    l_MT = l_MT_0 + disp*float(scale) # scaled MT length  

    exp_force_path = os.path.join(exp_path, f"force_trial{idx+1}.dat")
    exp_force = read_data(exp_force_path, 'force_M')
    exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='quadratic')(np.arange(0,2,dt)) # interpolate to have equal n of points
    yielding = 'n' # yielding not included
    sag = 'n'
    
    
elif benchmark == 'sub': #SUB-MAXIMAL benchmark - SLOW MUSCLE (Perreault 2003) 
    
    trial = simpledialog.askstring("Input", "Trial type? ('iso' or 'dyn'):")
    stim = simpledialog.askstring("Input", "Stimulation type? ('c' or 'v'):")
    fs = simpledialog.askstring("Input", "Stimulation frequency (Hz):")
    muscle = 'SOL' # dorsi/plantar
    yielding = 'y' # yielding included
    sag = 'y'
    
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [26.9, 65, 30, 1, 7.5*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0) - 4  # Musculo-tendon length (mm)
    
    t_end = 2
    time_dt = np.arange(0, t_end, dt) # time for BB
    exp_path_disp = exp_path + 'slowMuscle_submaximalActivation\\' 
    exp_path = exp_path_disp + stim + '_freq\\'
    Distimes = np.load(exp_path + fs + '_' + stim + '_times.npy') # exp. discharge times
    
    if trial == 'iso':
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # constant MT length
        exp_force = read_data(exp_path + 'force_isometric_' + stim + fs + '.dat', 'force_force_SM')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))
    
    elif trial == 'dyn':
        
        d = simpledialog.askstring("Input", "Displacement amplitude (1 or 8mm):")
        
        disp = read_data(exp_path_disp + 'displacement_' +  d + '.dat', 'displacement_SM') # load displacement
        disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate displacement
        l_MT = l_MT_0 + (disp + 8) # scaled MT length
        exp_force = read_data(exp_path + 'force_' + stim + fs + '_' + d + '.dat', 'force_SM')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))


elif benchmark == 'len': # TWITCH, SUB-TETANIC, TETANIC at different lengths (Kim 2015)
    
    l = simpledialog.askstring("Input", "Length? ('0', '8', or '16'):")
    fs = simpledialog.askstring("Input", '"Stimulation frequency (1, 10, 20, 40 Hz):"')
    yielding = 'n'
    sag = 'n'
    
    exp_path = exp_path + 'slowMuscle_lengthEffect\\'
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [30, 65, 30, 1, 7.5*np.pi/180] # MVC and M/T lengths
    t_end = 1.4
    muscle = 'SOL' # dorsi/plantar
    time_dt = np.arange(0, t_end, dt) # time for BB
    
    if l == '0':
        d = 8
    elif l == '8':
        d = 0 
    elif l == '16':
        d = -8
        
    l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0) - 4    # Musculo-tendon length (mm)
    l_MT_0 = l_MT_0 + d  # apply displacement
    l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array

    exp_force = np.load(exp_path + l + '_' + fs + '_interp.npy')
    if fs == '1':
        Distimes = np.round(np.array(np.load(exp_path + l + '_' + fs + '_times.npy')), 3)
    else:
        Distimes = np.load(exp_path + l + '_' + fs + '_times.npy') # exp. discharge times
    
    
elif benchmark == 'fast':  # Test fast muscle (Brown 1999 - Cat CF) 

    benchmark2 = simpledialog.askstring("Input", "muscle or MU: ")
    
    if benchmark2 == 'muscle':
        
        trial = simpledialog.askstring("Input", "Trial type? ('iso_f', 'iso_l', 'dyn'):")
        yielding = 'n'
        sag = 'n'

        if trial == 'iso_f': # isometric frequency variation

            exp_path = exp_path + 'fastMuscle_Millard\\'

            data = read_data(exp_path + 'FFR.ddf', 'Millard_EDL')

            f = simpledialog.askstring("Input", "Stimulation frequency in Hz (30, 50, 60, 70, 80, 90, 100, 120):")

            part = prepare_segment(data, fs=1000, target_freq = float(f), dt=1e-4)
            Distimes = part['spike_times_sec']
            time_dt = part['t_hi']
            exp_force = part['force_hi']
            muscle = 'EDL'
            
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.8, 17.1, 17.1, 1, 6*np.pi/180] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array

        elif trial == 'iso_l': # isometric length variation

            exp_path = exp_path + 'fastMuscle\\iso\\'
            
            l_M = simpledialog.askstring("Input", "Muscle length (0.8, 0.9, 1.0, 1.1, 1.2):")
            exp_force = np.load(exp_path + trial + '_' + l_M + '_interp.npy')
            
            Distimes = np.arange(0, 0.25, 1/30) # exp. discharge times
            t_end = 0.36
            muscle = 'CF' # dorsi/plantar
            time_dt = np.arange(0, t_end, dt)
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [15.4, 17, 0.37*56, float(l_M), 0] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
            
        elif trial == 'dyn': # dynamic length & frequency variation
            
            exp_path = exp_path + 'fastMuscle\\dyn\\'

            d = simpledialog.askstring("Input", "Displacement amplitude ('short' or 'length'):")
            
            t_end = 0.16
            Distimes = np.arange(0, t_end, 1/120) # exp. discharge times
            muscle = 'CF' # dorsi/plantar
            time_dt = np.arange(0, t_end, dt)
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [15.4, 17, 0.37*56, 0.95, 0] # MVC and M/T lengths
            exp_force = np.load(exp_path + '120_0.95_' + d + '_interp.npy')
            disp = np.load(exp_path + 'disp_' +  d + '_interp.npy') # load displacement
            l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)
            l_MT = np.empty((len(time_dt)+1), dtype=object) # full MT length array
            l_MT[0:-1] = np.ones((len(time_dt)), dtype=object)*l_MT_0 + (disp*l_M_opt)*np.cos(alpha_0)
            l_MT[-1] = l_MT[-2]
    
    
    elif benchmark2 == 'MU': # fast MU benchmark Chelichowski 1999
        
        exp_path = exp_path + 'fastMU\\'

        yielding = 'n'
        sag = 'y'
        
        fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (25, 30, 35, 40, 150):")
        exp_force = np.load(exp_path + 'iso_' + fs + '_interp.npy')
        
        Distimes = np.load(exp_path + 'iso_' + fs + '_times.npy') # exp. discharge times
        t_end = 0.7
        muscle = 'GM' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.078, 24.8, 13.5, 1, 0] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
        
    
elif benchmark == 'Ca': # Test Ca dynamics (Hollingworth, Rincon exp. data)
    
    fibre = simpledialog.askstring("Input", "Fibre type? ('slow', 'fast'):")
    yielding = 'n'
    sag = 'n'
    
    exp_path = exp_path + 'Ca_transients\\'

    t_end = 1.4
    time_dt = np.arange(0, t_end, dt)
    
    if fibre == 'slow':

        Ca_slow_23 = "Ca_slow_23_100Hz.csv" # load digitized slow fibre data
        Ca_slow_23 = pd.read_csv(exp_path + Ca_slow_23, delimiter = ' ')
        Ca_slow_23 = Ca_slow_23.to_numpy()  

        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1, 0, 30, 1, 0] # at optimal sarcomere length (assumption)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array
        T = 1/102 # frequency = 100 Hz (corrected)
        Distimes = np.arange(0, 0.04, T)
        muscle = 'SOL' # slow fibre muscle

    elif fibre == 'fast':

        Ca_fast_35 = "Ca_fast_35_125Hz.csv" # load digitized fast fibre data
        Ca_fast_35 = pd.read_csv(exp_path + Ca_fast_35, delimiter = ' ')
        Ca_fast_35 = Ca_fast_35.to_numpy()

        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1.6, 0, 30, 1.6, 0] # at longer sarcomere length (see article)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array
        T = 1/125 # frequency = 125 Hz
        Distimes = np.arange(0, 0.08, T)
        muscle = 'GM' # fast fibre muscle
    

" Create dictionary as model's input "
    
parameters = {
    'muscle': muscle, 
    'time': time_dt, 
    'pool': pool,
    'dt': dt,
    'MVC': MVC,
    'spread': spread,
    'MN_pool': MN_pool,
    'yielding': yielding,
    'sag': sag,
    'Nr': Nr,
    'l_T_slack': l_T_slack, # scaled with respect to Rajagopal model
    'l_M_opt': l_M_opt, # scaled with respect to Rajagopal model
    'alpha_0': alpha_0,
    'l_MT': l_MT,
    'species': species,
    'vmax': 10.5428 *l_M_opt,
}

states = {
    'MUAP_0': 0,
    'Ca_0': 0,
    'act_0': 1e-9,
    'l_M_0': l_M_0,
    'y_0': 1,
    's_0': 1
}


###############################################################################
""" PARAMETERS OPTIMIZATION (Nelder-Mead-lstsq)"""
# Decomment setion and comment the "Running simulations and plotting solutions" block
###############################################################################

""" 1) MAXIMAL benchmark [MVC, Vmax] estimation """

# def obj(x, parameters, states, Distimes, exp_force): 
    
#     parameters['MVC'] =  x[0]
#     parameters['vmax'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
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
    
#     parameters['MVC'] =  x[0]
#     parameters['Ca_max'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
#     force_sim, _, _, _, _, _, _ = model.run_MT_simulation() # Run the simulation
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [26.1, 5e5]
# bnds = [(25, 30), (1e5, 5e6)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("MVC =", res.x[0])
# print("Ca_max =", res.x[1])

"""  5) SUB-MAXIMAL benchmark [k1, k2] on 40Hz length trial """

# def obj(x, parameters, states, Distimes, exp_force): 
    
#     parameters['k1'] =  x[0]
#     parameters['k2'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
#     force_sim, _, _, _, _, _, _ = model.run_MT_simulation() # Run the simulation
    
#     residuals = force_sim - exp_force  
#     return np.sum(residuals**2)  

# x0 = [11, 15]
# bnds = [(5, 30), (5, 30)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})

# print("Optimized parameters:")
# print("k1 =", res.x[0])
# print("k2 =", res.x[1])

""" 6) Ca transient ODE [c1, c2, c3] parameters estimation """

# def obj(x, parameters, states, Distimes, exp_data): 
    
#     parameters['c_1'] = x[0]
#     parameters['c_3'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
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


""" Running simulations and plot solutions """

model = MN_driven_model(parameters, states, Distimes) # Create an model class instance


if benchmark == 'max' or benchmark == 'sub' or benchmark == 'len' or benchmark == 'fast':
    
    force_sim, _, Ca, a, l_M, _, _ = model.run_MT_simulation() # Run the simulation

    # Plot the force profiles
    plt.rcParams['figure.dpi'] = 100
    plt.figure(figsize=(8, 4))
    if 'benchmark2' in globals() and benchmark2 == 'muscle':
        plt.plot(time_dt, force_sim/MVC, 'r', label='Simulated Force')
    else:
        plt.plot(time_dt, force_sim, 'r', label='Simulated Force')
        #plt.plot(time_dt, l_M[0,:]/l_M_opt, 'r', label='Muscle length')
    plt.plot(time_dt, exp_force, 'k', label='Exp. Force')
    plt.ylabel('Force [N]', weight='bold', fontsize=12)
    plt.xlabel('Time [s]', weight='bold', fontsize=12)
    plt.title('Reconstructed ' + muscle + ' force for ' + str(Nr) + ' MUs', weight='bold')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    

elif benchmark == 'Ca':
    
    _, _, Ca, _, _ = model.run_M_simulation()
    
    plt.rcParams['figure.dpi'] = 100
    plt.figure(figsize=(10, 3))

    # Plot excitation/activation quantities
    if fibre == 'slow':
        plt.plot(time_dt, Ca[0,:]*10**6, 'g', label='Simulated (23°C)')
        plt.plot((Ca_slow_23[:,0] - Ca_slow_23[0,0])*10**-3, Ca_slow_23[:,1], 'k--', label = 'Rincon et al. 2021 (23°C)')
        plt.ylabel(r'Free [$Ca^{2+}$] [$\mu$M]', weight='bold', fontsize=12) 
        plt.xlabel('Time [s]', weight='bold', fontsize=12)
        plt.title(r'Free [$Ca^{2+}$] for slow mouse fibres', weight='bold', fontsize=15)
        plt.legend(loc='lower right', fontsize=15)
        plt.xlim((0, 0.12))
        #plt.ylim((-0.8, 20.8))
        plt.grid()
        plt.show()
    elif fibre == 'fast':
        plt.plot(time_dt, Ca[0,:]*10**6, 'g', label='Simulated (35°C)')
        plt.plot((Ca_fast_35[:,0] - Ca_fast_35[0,0])*10**-3, Ca_fast_35[:,1], 'k--', label = 'Hollingworth 1996 (35°C)')
        plt.ylabel(r'Free [$Ca^{2+}$] [$\mu$M]', weight='bold', fontsize=12) 
        plt.xlabel('Time [s]', weight='bold', fontsize=12)
        plt.title(r'Free [$Ca^{2+}$] for fast mouse fibres', weight='bold', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.xlim((0, 0.12))
        plt.ylim((-0.8, 20.8))
        plt.grid()
        plt.show()   



# # Error metrics (%mAE, %MAE)
# mean_abs_error = np.mean((np.abs(force_sim - exp_force)/MVC)*100)
# max_abs_error = np.max((np.abs(force_sim - exp_force)/MVC)*100)


# Save results

if save == 'y':
    print("Saving results...")
    np.save('a_30', a, allow_pickle=True)

