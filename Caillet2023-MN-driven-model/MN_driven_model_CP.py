"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue Jun  4 11:03:08 2024
___________________________________

MN-driven model with SE and PEE adapted for BB tests of Millard et. al 2023
"""

import sys
sys.path.insert(0,'Modules_subBB')
import os
cwd = os.getcwd()
import scipy.io as sio
import numpy as np
# import scipy as sp
# import scipy.interpolate
from scipy import signal
#import matplotlib
#from scipy.signal import find_peaks
#from scipy.optimize import minimize
#from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pennation_angle import penn_ang
from PE_force import PEE_force
from Tendon_force import T_force
from velocity_fFV import velo_fFV
from yielding import Yield
#from force_fFV import f_fFV
from MU_type_id_MOD import MU_type_id_func
from MU_AP_MOD import MU_AP_func
from MU_free_Ca_MOD import MU_free_Ca_func
from MU_active_state_MOD import MU_active_state_func
from Force_Length_MOD import Force_Length_func
from F0MU_distrib_MOD import F0MU_distrib_func

#%%
" Choose subject "
#s = 6
s = 12
ngrid = 2

" Saving the simulations (y/n)? "
save = 'n'

" Load DISCHARGE & FORCE data "

#os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\CP_children_Wiedemann\\CP_pre_post\\pre\\CP_' + str(s))
os.chdir('C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\HDsEMG_study\\TD_group\\S01_TD_20250129')

#data = sio.loadmat('s6_med_2.mat_decomp.mat_edited.mat')
#data = sio.loadmat('s12_med_3.mat_decomp.mat_edited.mat')
data = sio.loadmat('S01_TD_30MVC_plantar.otb+_decomp.mat_edited.mat')
d = {key: data[key] for key in data.keys() & {'edition', 'signal'}}
path = d['signal']['path'][0,0]  # extract torque
Distimes = d['edition']['Distimeclean'][0,0][0,0]  # extract exp. edited discharge times
#Distimes = np.load('CP_06_30_predicted_distime.npy', allow_pickle=True) # extract reconstructed disch times of the complete MN pool

if ngrid == 2:  # check the second grid if there's one
    Distimes2 = d['edition']['Distimeclean'][0,0][0,1] # extract data from second grid
    if np.size(Distimes, axis=1) == 0 and np.size(Distimes2, axis=1) != 0: # check for empty data
        Distimes = Distimes2
    elif np.size(Distimes2, axis=1) == 0 and np.size(Distimes, axis=1) != 0:
        Distimes = Distimes
    else:
        Distimes = np.concatenate((Distimes, Distimes2), axis = 1)

os.chdir(cwd)

#%%
" Sort MUs according to recruitment "
sorted_indices = np.argsort([Distimes[0, i][0, 0] for i in range(np.size(Distimes, axis=1))])
Distimes = Distimes[:, sorted_indices]

#%%
" SET PARAMETERS "

muscle = 'GM' # TA
T = 1/2048  # fd = 2048 Hz
time_real = np.arange(0, np.size(path, axis=1)*T, T)
dt = 0.0001 # dt for solution and arrays
time_dt = np.arange(0, np.size(path, axis=1)*dt, dt) # time 
MVC = 19.83 # muscle max. isom. force [N]   S6 = 9.625, S12 = 9.94 S1TD = 19.83
Nr = np.size(Distimes, axis=1) # n. of MNs in the pool
l_T_slack = 240 # Tendon slack length (mm) S6 = 217.5, S12 = 225 S1TD = 240
l_M_opt = 80 # Optimal fiber length (mm) S6 = 72.5, S12 = 75 S1TD = 80
l_M_0 = 1 # Initial normalised fiber length (in optimal length units)
alpha_0 = 20*np.pi/180 # initial pennation (pennation should make less than 2 % difference)
l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0) # Musculo-tendon length (mm)
vmax = 10.5428*l_M_opt # max velocity (from optimisation)

#ISOMETRIC CASE
l_MT = np.zeros((len(time_dt)+1), dtype=object) 
l_MT = l_MT + l_MT_0

#DYNAMIC CASE 
#l_MT = l_MT_0 + (disp_1_int+8) # scaled MT length
#l_MT = l_MT_0 + (disp_2_int+8) # scaled MT length

#%%
" PLOT SPIKES & FORCE "

# fig, axs = plt.subplots(2, 1,  figsize=(9, 6))

# f = lambda x,pos: str(x).rstrip('0').rstrip('.')
# ylabel='n. MUs'
# axs.set_xlabel('Time [s]', color='k',fontsize=20, weight='bold')
# axs.set_ylabel(ylabel, color='k',fontsize=20, weight='bold')
# axs.set_ylim(0,  Nr)
# #plt.grid()

# axs.ax2= axs.twinx()
# axs.ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
# axs.ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
# axs.ax2.set_ylabel('Exp. force (%MVC)',fontsize=15, weight='bold')
# width, fs = 0.99, 1
# colors1, lineoffsets1, linelengths1 = ['C{}'.format(i) for i in range(Nr)], np.arange(1,Nr+1,1), np.ones((1,Nr))[0]*width

# # Plot torque and spike trains
# axs.eventplot(Distimes[0:Nr]/fs, colors=colors1, lineoffsets=lineoffsets1,linelengths=linelengths1)
# axs.ax2.plot(time_real, path[0,:]*100,  '-g', linewidth=3, label='Experimental transducer force')#' Whole Muscle Force '+r'$F^{MT} (t)$' )

#%%
" ARRAYS PREALLOCATION "
    
alpha = np.empty((Nr,len(time_dt)+1), dtype=object) # Pennation
l_T = np.empty((Nr,len(time_dt)), dtype=object) # Tendon length
eps_T = np.empty((Nr,len(time_dt)), dtype=object) # Tendon strain
SE_force = np.empty((Nr,len(time_dt)), dtype=object) # Tendon force
CE_force = np.empty((Nr,len(time_dt)), dtype=object) # Contractile el. force
PE_force = np.empty((Nr,len(time_dt)), dtype=object) # Parallel elastic el. force
FL_force = np.empty((Nr,len(time_dt)), dtype=object) # Force-length relationship (force)
vel = np.empty((Nr,len(time_dt)), dtype=object) # velocity from FV relationship
F_M = np.empty((Nr,len(time_dt)), dtype=object) # force from FV relationship

l_M = np.empty((Nr, len(time_dt)), dtype=object) # MUs length in time
active_state = np.empty((Nr, len(time_dt)), dtype=object) # active state
MUAP_nerve = np.empty((Nr, len(time_dt)), dtype=object) # MU AP nerve signal
free_Ca = np.empty((Nr, len(time_dt)), dtype=object) # free [Ca] course
yielding = np.empty((Nr, len(time_dt)), dtype=object) # yieldin coeff.

#%%
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

print('There are ', Nr, ' discharging MUs in this simulation')
F0MU_distribution = F0MU_distrib_func(Nr, MVC) # F0MU distribution across the sample of MUs

for i in range (Nr):  # ...for each considered i-th MU
    print('Computing Force for MU n°', str(i+1))
       
    sp_matrix = time_dt[Distimes[0,i]]
    
    MU_type = MU_type_id_func(i, Nr, muscle)  # i-th MU type identification (fast/slow)
    Matrix_AP = sp_matrix.astype(float)  # i-th MU discharge times [s] 
    
    def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha, vmax):
       
        if int(t/dt) == 0:   # initial pennation angle
            alpha[int(t/dt)] = alpha_0
            
        l_T = l_MT[int(t/dt)] - (y[5])*np.cos(alpha[int(t/dt)]) # tendon length
        eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
        SE_force = T_force(eps_T) # tendon force
        PE_force = PEE_force(y[5]/l_M_opt) # passive el. force    
        CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
        alpha[int((t+dt)/dt)] = penn_ang(l_MT[int((t+dt)/dt)], y[5], l_T, l_M_0*l_M_opt, alpha_0) # update pennation angle
        
        dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # MUAP
            
        dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[5]/l_M_opt, MU_type, Matrix_AP) # Free Ca (remember to avoid negligible negative values)
            
        dadt = MU_active_state_func(t, y[2], y[4], y[6]) # Active state
        
        FL = Force_Length_func(y[5]/l_M_opt, y[4]) # F-L relationship 
            
        dldt = velo_fFV(t, y, CE_force/FL, FL, y[4], y[5]/l_M_opt, MU_type, vmax) # velocity
        
        dY = Yield(dt, t, y, dldt/vmax) # yielding (Brown 1999)
        
        return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dadt, dldt, dY]
     
    
    y0 = [0, 0, 0, 0, 1*10**-9, l_M_opt, 1] # set initial states (active state can't be 0 otherwise you'll divide by 0 in FV)
    p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha[i,:], vmax) # set ODE parameters
    sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/4) # solve IVP
    
    active_state[i,:] = sol.y[4]  # get active state
    l_M[i,:] = sol.y[5]  # get l_M
    #free_Ca[i,:] = sol.y[2] # get free [Ca]
    yielding[i,:] = sol.y[6] # get yielding coeff.
   
    # now recalculate data based on l_M values..
    alpha[i,0] = alpha_0
    for l in range(len(time_dt)):
        l_T[i,l] = l_MT[l] - (l_M[i,l])*np.cos(alpha[i,l]) # tendon length
        eps_T[i,l] = (l_T[i,l]-l_T_slack)/l_T_slack # new tendon strain    
        SE_force[i,l] = T_force(eps_T[i,l]) # tendon force
        PE_force[i,l] = PEE_force(l_M[i,l]/l_M_opt) # passive el. force 
        CE_force[i,l] = SE_force[i,l]/np.cos(alpha[i,l]) - PE_force[i,l] # contractile element force
        # FL_force[i,l] = Force_Length_func(l_M[i,l]/l_M_opt, active_state[i,l])
        # vel[i,l] = velo_fFV(0, 0, CE_force[i,l]/(FL_force[i,l]), FL_force[i,l], active_state[i,l], l_M[i,l]/l_M_opt, MU_type, vmax)
        # F_M[i,l] = f_fFV(vel[i,l]/vmax, FL_force[i,l], active_state[i,l], l_M[i,l]/l_M_opt, MU_type)
        
        alpha[i,l+1] = penn_ang(l_MT[l+1], l_M[i,l], l_T[i,l], l_M_opt, alpha_0) # update pennation angle 
      
MU_Force_list = yielding*active_state*CE_force + PE_force

F_MU_list = F0MU_distribution[0:Nr,:] * MU_Force_list # Newtons
Tot_Muscle_force = F_MU_list.sum(axis=0)/MVC # Total muscle force (in Newton)

#%%
b, a = signal.butter(4, 5/(2048/2), 'low') # LPF 
Tot_Muscle_force_filt = signal.filtfilt(b, a, Tot_Muscle_force) 
    
#%%
" Visual validation "
plt.rcParams['figure.dpi'] = 400

plt.plot(time_real, Tot_Muscle_force_filt*100, 'r', label='Simulated Force')
plt.plot(time_real, path[0,:]*100, 'k', label='Exp. Force')
plt.ylabel('Normalised force [% MVC]', weight='bold')
plt.xlabel('Time [s]', weight='bold')
plt.legend(loc='lower right')
plt.grid()  

#%%
# ERROR METRICS
# abs_p_err = np.empty((len(force_bb)), dtype=object)
# for r in range(len(force_bb)):
#     for rr in range(len(time_dt)):
#         if force_bb[r,0] - time_dt[rr] < 0.0001:
#             abs_p_err[r] = np.abs(force_bb[r,1]-Tot_Muscle_force[rr])*100
#             break
        
# mean_err = np.mean(abs_p_err) 
# max_err = np.max(abs_p_err)

# abs_p_err_int = np.empty((len(Tot_Muscle_force)), dtype=object)
# abs_p_err_int = np.abs(force_bb_int/muscle_F0M - Tot_Muscle_force/muscle_F0M)*100
# abs_p_err_int_disp = np.abs(force_bb_disp_int/muscle_F0M - Tot_Muscle_force/muscle_F0M)*100

# mean_err_int = np.mean(abs_p_err_int)
# max_err_int = np.max(abs_p_err_int)

# mean_err_int_disp = np.mean(abs_p_err_int_disp)
# max_err_int_disp = np.max(abs_p_err_int_disp)

#%%
#Saving data
if save =='y':
    os.chdir('C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\variablefreq_yielding')

    #np.save('time_sim', time_dt, allow_pickle=True) 
    #np.save('time_exp', time, allow_pickle=True) 
    #np.save('MU_act_list_pennation', active_state, allow_pickle=True) 
    #np.save('MU_force_list_pennation', F_MU_list, allow_pickle=True) 
    #np.save('F0MU_distrib', F0MU_distribution, allow_pickle=True) 
    np.save('R_var5', Tot_Muscle_force, allow_pickle=True)
    #np.save('scaled_exp_force_in_N', Exp_muscle_force, allow_pickle=True) 
    # np.save('Tendon_length_maxBB', l_T[2,:], allow_pickle=True)
    # np.save('Tendon_strain_maxBB', eps_T[2,:], allow_pickle=True)
    # np.save('Tendon_force_maxBB', SE_force[2,:], allow_pickle=True)
    # np.save('PEE_force_maxBB', PE_force[2,:], allow_pickle=True)
    
    #encore...
    #     np.save('MNAP_list', MN_AP_list, allow_pickle=True) 
    #     np.save('MUAP_list', MUAP_nerve, allow_pickle=True) 
    #     np.save('freeCa_list', free_Ca, allow_pickle=True) 
    #     np.save('boundCa_list', Ca_Tn, allow_pickle=True)         
     
os.chdir(cwd)    