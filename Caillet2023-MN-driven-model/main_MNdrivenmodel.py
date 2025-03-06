# -*- coding: utf-8 -*-
"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Created on Wed Mar  5 16:50:33 2025
___________________________________

"""
import os
import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from MN_driven_model import MN_driven_model  # Import your class

#%%
# Get current working directory
ngrid = 2 # number of HDsEMG grids
save = 'n' # save results 'y' 'n'

#%%
" LOAD DISCHARGE TIMES "

cwd = os.getcwd()
os.chdir('C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\HDsEMG_study\\TD_group\\S01_TD_20250129')

# data = sio.loadmat('s6_med_2.mat_decomp.mat_edited.mat')
# data = sio.loadmat('s12_med_3.mat_decomp.mat_edited.mat')
data = sio.loadmat('S01_TD_10MVC_dorsi.otb+_decomp.mat_edited.mat')
d = {key: data[key] for key in data.keys() & {'edition', 'signal'}}
path = d['signal']['path'][0, 0]  # extract torque
Distimes = d['edition']['Distimeclean'][0, 0][0, 0]  # extract exp. edited discharge times
# Distimes = np.load('CP_06_30_predicted_distime.npy', allow_pickle=True) # extract reconstructed disch times of the complete MN pool

if ngrid == 2:  # check the second grid if there's one
    Distimes2 = d['edition']['Distimeclean'][0, 0][0, 1]  # extract data from second grid
    if np.size(Distimes, axis=1) == 0 and np.size(Distimes2, axis=1) != 0:  # check for empty data
        Distimes = Distimes2
    elif np.size(Distimes2, axis=1) == 0 and np.size(Distimes, axis=1) != 0:
        Distimes = Distimes
    else:
        Distimes = np.concatenate((Distimes, Distimes2), axis=1)

os.chdir(cwd)

#%%
" SET PARAMETERS "

muscle = 'GM'  # TA
T = 1/2048  # fd = 2048 Hz
time_real = np.arange(0, np.size(path, axis=1)*T, T)
dt = 0.0001  # dt for solution and arrays
time_dt = np.arange(0, np.size(path, axis=1)*dt, dt)  # time
MVC = 23.83 * 9.81  # muscle max. isom. force [N]    S6 = 9.625, S12 = 9.94 S1TD = 19.83
Nr = np.size(Distimes, axis=1)  # n. of MNs in the pool
l_T_slack = 240  # Tendon slack length (mm) S6 = 217.5, S12 = 225 S1TD = 240
l_M_opt = 80  # Optimal fiber length (mm) S6 = 72.5, S12 = 75 S1TD = 80
l_M_0 = 1  # Initial normalised fiber length (in optimal length units)
alpha_0 = 11.47*np.pi / 180  # initial pennation (pennation should make less than 2 % difference)
l_MT_0 = l_T_slack + (l_M_opt) * np.cos(alpha_0)  # Musculo-tendon length (mm)
vmax = 10.5428 * l_M_opt  # max velocity (from optimisation)

# Parameters dictionary
parameters = {
    'muscle': 'TA', # TA or GM
    'time': time_dt,  # End time
    'dt': dt,
    'MVC': MVC,
    'Nr': Nr,
    'l_T_slack': l_T_slack,
    'alpha_0': alpha_0,
    'l_MT_0': l_MT_0,
    'species': 'human', # or 'animal'
    'vmax': vmax,
    'l_M_0': l_M_0,
    'l_M_opt': l_M_opt
}

#%%

# Create an instance of the MN_driven_model class
model = MN_driven_model(parameters, Distimes)

# Run the simulation
force_sim = model.run_simulation()

#%%
" Filter simulated force "

b, a = signal.butter(4, 5/(2048/2), 'low') # LPF 
force_sim_filt = signal.filtfilt(b, a, force_sim) 
    
#%%
" Visual validation and result saving "  

plt.rcParams['figure.dpi'] = 400

plt.plot(time_real, (force_sim_filt/MVC)*100, 'r', label='Simulated Force')
plt.plot(time_real, path[0,:]*100, 'k', label='Exp. Force')
plt.ylabel('Normalised force [% MVC]', weight='bold')
#plt.xlabel('Time [s]', weight='bold')
plt.legend(loc='lower right')
plt.grid()  


if save == 'y':
    print("Saving results...")
    np.save('Predicted_force', force_sim, allow_pickle=True)
    
    