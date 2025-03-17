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
from MN_driven_model import MN_driven_model  # Import the model' class
cwd = os.getcwd()

#%%
###############################################################################
""" INSERT filename, musclename and contraction intensity """
###############################################################################

user = 'z5517249' # z5517249 or Andrea
test = 'S01_TD' # participant ID
muscle = 'GM' # dorsi/plantar
MN_pool = 400
MVC_trapez = '50' # theoretical plateau MVC
spread = 'identified' # evenly or identified
species = 'human' # human/animal
save = 'n' # save results 'y' or 'n'

#%%
" PARAMETERS SETTING FROM INPUT TEST CASE "

test_cases= np.array([['S01_TD', 'TA', 1, 0.31799, 23.83, 2, 11.47, 2048, 0.08, 0.085, 57.65],
                      ['S01_TD', 'GM', 1, 0.38196, 19.83, 2, 22.08, 2048, 0.08, 0.085, 70.12], 
                      ['S02_TD', 'TA', 1, 0.36343, 25.62, 2, 14.12, 2048, 0.1, 0.11, 92.58],
                      ['S02_TD', 'GM', 1, 0.42698, 22.12, 2, 22.19, 2048, 0.1, 0.11, 134.64], 
                      ['S03_TD', 'TA', 1, 0.29562, 17.21, 2, 10.63, 2048, 0.085, 0.092, 43.86],
                      ['S03_TD', 'GM', 1, 0.33665, 24.06, 2, 21.83, 2048, 0.085, 0.092, 61.26], 
                      ], dtype=object)  # lengths in m, freq in Hz, angles in degrees, PCSA in cm2

[name, muscle, l_M_0, l_MT_0, MVC_rec, ngrid, alpha_0, fs, a, h, vol] = test_cases[np.where((test_cases[:, 0] == test) & (test_cases[:, 1] == muscle))[0][0]]

#%%
" Load discharge times indeces "

os.chdir('C:\\Users\\' + user + '\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\HDsEMG_study\\TD_group\\' + test)

if muscle == 'TA':
    task = 'dorsi'
elif muscle == 'GM':
    task = 'plantar'

data = sio.loadmat(name + '_' + MVC_trapez + 'MVC_' + task + '.otb+_decomp.mat_edited.mat')
d = {key: data[key] for key in data.keys() & {'edition', 'signal'}}
path = d['signal']['path'][0, 0] # extract dynamometer recorded force [% MVC]
Distimes = d['edition']['Distimeclean'][0, 0][0, 0]  # extract exp. edited discharge times

if ngrid == 2:  # check the second grid if there's one
    Distimes2 = d['edition']['Distimeclean'][0, 0][0, 1]  # extract data from second grid
    if np.size(Distimes, axis=1) == 0 and np.size(Distimes2, axis=1) != 0:  # check for empty data
        Distimes = Distimes2
    elif np.size(Distimes2, axis=1) == 0 and np.size(Distimes, axis=1) != 0:
        Distimes = Distimes
    else:
        Distimes = np.concatenate((Distimes, Distimes2), axis=1)

sorted_indices = np.argsort([Distimes[0, i][0, 0] for i in range(np.size(Distimes, axis=1))])
Distimes = Distimes[:, sorted_indices]

os.chdir(cwd)

#%%
" Time, MVC and number of MUs "

T = 1/fs  # fd = 2048 Hz
time_real = np.arange(0, np.size(path, axis=1)*T, T) # actual time vector
dt = 0.0001  # dt for solution and arrays
time_dt = np.arange(0, np.size(path, axis=1)*dt, dt)  # time for solving ODE system
MVC_rec = MVC_rec * 9.81  # muscle max. isom. force [N]    
Nr = np.size(Distimes, axis=1)  # n. of MNs in the pool

x =  0.15 # OT Bioelettronica dynamometer heel to force-cell lenght [m]

#%%
" Scale model and extract single muscles contribution "

if muscle == 'TA':
    l_MT_Raj = 0.32642 # Rajagopal generic model [m]
    l_T_slack_Raj = 0.241 
    l_M_opt_Raj = 0.068
    r = 0.034  # moment arm
    M0_Raj = 41.03 # max. moment [Nm]
    Mtot = 68.26  # tot. moment from all dorsiflexors
elif muscle == 'GM':
    l_MT_Raj = 0.43418
    l_T_slack_Raj = 0.399
    l_M_opt_Raj = 0.051
    r = 0.04 
    M0_Raj = 52.6
    Mtot = 228

F0_Raj = M0_Raj/r
scale = l_MT_0/l_MT_Raj # scale factor from MT length
l_T_slack = l_T_slack_Raj*scale
l_M_opt = l_M_opt_Raj*scale

T_ankle = MVC_rec * ((np.sqrt((x-np.sqrt(h**2-a**2))**2 + a**2)) / np.cos(30*np.pi / 180)) # from dynamomter force to ankle moment [Nm]
MVC = ((vol*np.cos(alpha_0*np.pi / 180))/(l_M_opt*(10**2))) * 60 # muscle MVC from vol (PCSA) adn scaled opt fibre length
r_muscle = r * scale # scale the moment arm
M_MVC = MVC * r_muscle # muscle MVC ankle moment
M_ratio =  M_MVC/T_ankle # % of ankle torque produced by the muscle

#%%
" Create dictionary as model's input "

parameters = {
    'muscle': muscle, 
    'time': time_dt, 
    'path': path,
    'dt': dt,
    'MVC': MVC,
    'spread': spread,
    'MN_pool': MN_pool,
    'Nr': Nr,
    'l_T_slack': l_T_slack*(10**3), # scaled with respect to Rajagopal model
    'l_M_opt': l_M_opt*(10**3), # scaled with respect to Rajagopal model
    'alpha_0': alpha_0*np.pi / 180,
    'l_MT_0': l_MT_0*(10**3),
    'species': species,
    'vmax': 10.5428 *l_M_opt*(10**3),
    'l_M_0': l_M_0
}

#%%
# Create an instance of the MN_driven_model class
model = MN_driven_model(parameters, Distimes)

# Run the simulation
force_sim = model.run_simulation()

# Calculate muscle force contribution
exp_force = (T_ankle*path[0,:]*M_ratio)/r_muscle

#%%
" Filter simulated force "

b, a = signal.butter(4, 5/(2048/2), 'low') # LPF 
force_sim_filt = signal.filtfilt(b, a, force_sim) 
    
#%%
" Visual validation and result saving "  

plt.rcParams['figure.dpi'] = 400
fig, ax = plt.subplots()

# Plot the force profile
ax.plot(time_real, force_sim_filt, 'r', label='Simulated Force')
ax.plot(time_real, exp_force, 'k', label='Exp. Force')
ax.set_ylabel('Force [N]', weight='bold', fontsize=12)
ax.set_xlabel('Time [s]', weight='bold', fontsize=12)
ax.set_title('Reconstructed ' + muscle + ' force for ' + str(Nr) + ' MUs', weight='bold')
ax.legend(loc='upper right')
ax.grid()

ax2 = ax.twinx()  
cmap = plt.get_cmap("Greens")  
color = 'green'

MU_range = 20  # Height for the motor units axis 

for i in range(Nr):
    color = cmap(np.clip(i / Nr, 1, 1))  # Assign a unique color to each motor unit
    ax2.eventplot(Distimes[0, i]/fs, lineoffsets=i + 1, colors='green', linelengths=0.5, 
              linewidth=0.8, alpha=0.7)

ax2.set_ylabel('n.MU', color='green', weight='bold', fontsize=12)
ax2.set_ylim([0, MU_range])  # Place the raster plot at the bottom of the main plot

ax2.spines['right'].set_color('green') 
ax2.spines['right'].set_linewidth(2)   
ax2.tick_params(axis='y', colors='green')

plt.tight_layout()
plt.show()

#%%
# Error metrics (%mAE, %MAE)
indices = force_sim_filt > 1 # calculate MAE only where MU activity is detected
mean_abs_error = np.mean((np.abs(force_sim_filt[indices] - exp_force[indices])/exp_force[indices])*100)
max_abs_error = np.max((np.abs(force_sim_filt[indices] - exp_force[indices])/exp_force[indices])*100)

#%%
# Save the figure if needed
if save == 'y':
    print("Saving results...")
    np.save('Predicted_force', force_sim, allow_pickle=True)


    
    