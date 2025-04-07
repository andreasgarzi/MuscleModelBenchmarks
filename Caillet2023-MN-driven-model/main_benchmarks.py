"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Thu Mar 20 08:49:54 2025
_______________________________________________________________________________

"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from MN_driven_model import MN_driven_model  # Import the model' class
cwd = os.getcwd()

#%%
###############################################################################
""" MODIFY parameters according to the banchmark """
###############################################################################

user = 'z5517249'
benchmark = 'sub' # max or sub
muscle = 'SOL' # dorsi/plantar
sim = 'MT' # MT or act
MN_pool = 1  # n. of theoretical MUs in the real pool 
Nr = 1 # n. of (exp.) MUs to represent in the pool
pool = 'n' # exp or pool
spread = 'evenly' # evenly or identified 
species = 'animal' # human/animal
yielding = 'y'
stim = 'v' # c (constant) or v (variable)
trial = 'dynamic' # isometric or dynamic
d = 1 # displacement amplitude (1 or 8mm)
fs = 10 # Stimulation freq.
T = 1/fs # corresponding time period
dt = 1e-04 # time step (x-data)
t_end = 2 # total seconds
scale = 2  # 0.05, 0.1, 0.25, 0.5, 1, 2 amplitude disp. scales

input_folder = 'C:\\Users\\' + user + '\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\benchmark_input'

save = 'n' # save results 'y' or 'n'

#%%
" Load experimental input data from Sandercock et al. and Perreault et al. "

time_dt = np.arange(0, t_end, dt) # time for BB

if benchmark == 'max':
    os.chdir(input_folder + '\\' + benchmark)
    Distimes = np.arange(0, t_end, T) # create array of dischare times at 70 Hz
    disp = np.genfromtxt('displacement.dat', delimiter='')
    disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # load displacement
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.36, 17.1, 17.1, 1, 6*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0)-2 # Musculo-tendon length (mm)
    l_MT = l_MT_0 + disp*scale # scaled MT length    
    exp_force = np.genfromtxt('force' + str(scale) + '.dat', delimiter='') # load experimental force
    exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,2,dt)) # interpolate to have equal n of points
    
elif benchmark == 'sub':
    os.chdir(input_folder + '\\' + benchmark) # load displacement, MVC and update MT lengths
    disp = np.genfromtxt('displacement_' +  str(d) + '.dat', delimiter='') # load displacement
    disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate displacement
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [25.1, 65, 30, 1, 7.5*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0)-4 # Musculo-tendon length (mm)
    
    os.chdir(input_folder + '\\' + benchmark + '\\' + stim + '_freq') 
    Distimes = np.load(str(fs) + '_' + stim + '_times.npy') # exp. discharge times
    
    if trial == 'isometric':
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # constant MT length
        exp_force = np.genfromtxt('force_isometric_' + stim + str(fs) + '.dat', delimiter='')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))
    elif trial == 'dynamic':
        l_MT = l_MT_0 + (disp+8) # scaled MT length
        exp_force = np.genfromtxt('force_' + stim + str(fs) + '_' + str(d) + '.dat', delimiter='')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))

os.chdir(cwd)

#%%
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
    'sim': sim,
    'Nr': Nr,
    'l_T_slack': l_T_slack, # scaled with respect to Rajagopal model
    'l_M_opt': l_M_opt, # scaled with respect to Rajagopal model
    'alpha_0': alpha_0*np.pi / 180,
    'l_MT': l_MT,
    'species': species,
    'vmax': 10.5428 *l_M_opt
}

states = {
    'MUAP_0': 0,
    'Ca_0': 0,
    'act_0': 1e-9,
    'l_M_0': l_M_0,
    'y_0': 1 
}

#%%
# Create an instance of the MN_driven_model class
model = MN_driven_model(parameters, states, Distimes)

# Run the simulation
force_sim, _, _, _, _, _ = model.run_simulation()
# _, Ca, _ = model.run_simulation()
    
#%%
" Visual validation and result saving "  

plt.rcParams['figure.dpi'] = 400

# Plot the force profiles
plt.plot(time_dt, force_sim/MVC, 'r', label='Simulated Force')
plt.plot(time_dt, exp_force/MVC, 'k', label='Exp. Force')
plt.ylabel('Normalised force [F0]', weight='bold', fontsize=12)
plt.xlabel('Time [s]', weight='bold', fontsize=12)
plt.title('Reconstructed ' + muscle + ' force for ' + str(Nr) + ' MUs', weight='bold')
plt.legend(loc='upper right')
plt.grid()

#%%
# Error metrics (%mAE, %MAE)
mean_abs_error = np.mean((np.abs(force_sim - exp_force)/MVC)*100)
max_abs_error = np.max((np.abs(force_sim - exp_force)/MVC)*100)

#%%
# Save the figure if needed
if save == 'y':
    print("Saving results...")
    np.save('Predicted_force', force_sim, allow_pickle=True)