"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Mon Jun 10 10:02:26 2024
___________________________________

Tuning of the active state ODE coefficients based on rat soleus MU twitch.
"""

#______________________________________________________________________________
import sys
sys.path.insert(0,'Modules_BB')
import os
cwd = os.getcwd()
#______________________________________________________________________________
import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from load_Input_Data_MOD import load_Input_Data_func
from pennation_angle import penn_ang
from input_spike_trains_MOD import input_spike_trains_func
from F_TA_MOD import F_TA_func
from PE_force import PEE_force
from Tendon_force import T_force
from velocity_fFV import velo_fFV
from MU_type_id_MOD import MU_type_id_func
from MU_AP_MOD import MU_AP_func
from MU_free_Ca_MOD import MU_free_Ca_func
from MU_bound_calcium_MOD import MU_bound_calcium_func
from MU_active_state_MOD import MU_active_state_func
from Force_Length_MOD import Force_Length_func
from F0MU_distrib_MOD import F0MU_distrib_func
from fibre_forces_MOD import fibre_forces_func
#______________________________________________________________________________
""" Load time, displacement and force form BB tests, create virtual MU spikes"""

dt = 0.0001 # time step (x-data)
t_end = 1 # total seconds
time_dt = np.arange(0, t_end, dt) # time for BB
fd = 50 # paper stimulation freq. Hz d.r. (to reach active_state = 1 always, without any recruitment)
T = 1/fd # correspondend d.r. period
Nr = 1 # n. of an average complete rat soleus MUs pool
muscle_F0M = 1.17 # muscle max. isom. force [N]

""" 1st CASE: d.r. at 70Hz constant"""
sp_matrix = np.empty((Nr, int(t_end/T)), dtype=float)
disch = np.arange(0,t_end,T) # create array of dischare times at 50 Hz
for i in range(Nr): # append Nr times
    sp_matrix[i] = disch

""" 2nd CASE: one spike only"""
# sp_matrix = np.zeros((Nr, int(t_end/T)), dtype=float)
# for i in range(Nr): # append Nr times
#     sp_matrix[i, 0] = 1

#______________________________________________________________________________      
print('There are ', Nr, ' discharging MUs in this simulation.')

# F0MU distribution across the sample of MUs
F0MU_distribution = F0MU_distrib_func(Nr, muscle_F0M)

#______________________________________________________________________________
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

l_T_slack = 17.1 # Tendon slack length (mm)
l_M_opt = l_T_slack # Optimal fiber length (mm)
l_M_0 = 1 # Initial fibre length normalized to l_M_0 (it would be l_MT - l_ST)
alpha_0 = 6*np.pi/180 # initial pennation (pennation should make less than 2 % difference)
l_MT = 34.2

alpha_store = np.empty((len(time_dt)+1), dtype=object) # Pennation for in-function iteration
alpha = np.empty((Nr,len(time_dt)), dtype=object) # Pennation
l_T = np.empty((Nr,len(time_dt)), dtype=object) # Tendon length
eps_T = np.empty((Nr,len(time_dt)), dtype=object) # Tendon strain
SE_force = np.empty((Nr,len(time_dt)), dtype=object) # Tendon force
CE_force = np.empty((Nr,len(time_dt)), dtype=object) # Contractile el. force
PE_force = np.empty((Nr,len(time_dt)), dtype=object) # Parallel elastic el. force
FL_force = np.empty((Nr,len(time_dt)), dtype=object) # Force-length relationship (force)

l_M = np.empty((Nr, len(time_dt)), dtype=object) # MUs length in time
active_state = np.empty((Nr, len(time_dt)), dtype=object) # active state
MUAP_nerve = np.empty((Nr, len(time_dt)), dtype=object) # MU AP nerve signal
free_Ca = np.empty((Nr, len(time_dt)), dtype=object) # free [Ca] course
Ca_Tn = np.empty((Nr, len(time_dt)), dtype=object) # Ca-Tn bound course

# ...for each considered i-th MU 
for i in range (Nr):  
    print('Computing Force for MU n°', str(i+1))
       
    MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
    Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
    
    def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha):
       
        if int(t/dt) == 0: # initial pennation angle
            alpha[int(t/dt)] = alpha_0
            
        l_T = l_MT - (y[6]*l_M_opt)*np.cos(alpha[int(t/dt)]) # tendon length
        eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
        SE_force = T_force(eps_T) # tendon force
        PE_force = PEE_force(y[6]) # passive el. force    
        CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
        alpha[int((t+dt)/dt)] = penn_ang(l_MT, y[6], l_T, l_M_0, alpha_0) # update pennation angle
        
        dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
        dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0]*0.85, y[6], MU_type, Matrix_AP) # Free Ca (remember to avoid negligible negative values)
        
        ddeltadt = MU_bound_calcium_func(t, y, y[2], y[6], MU_type, Matrix_AP) # Ca-Tn
            
        dadt = MU_active_state_func(t, y, y[4]) # Active state
            
        FL_force = Force_Length_func(y[6], y[5])*y[5] # F-L relationship (*active state)
            
        dldt = velo_fFV(t, y, CE_force, FL_force, y[5], y[6], MU_type) # velocity 
            
        return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, ddeltadt, dadt, dldt]
     
    
    y0 = [0, 0, 0, 0, 0, 0, l_M_0] # set initial states
    p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha_store) # set ODE parameters
    sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/2) # solve IVP
    
    active_state[i,:] = sol.y[5]  # get active state
    l_M[i,:] = sol.y[6]  # get l_M
    # MUAP_nerve[i] = sol.y[0]
    free_Ca[i,:] = sol.y[2]
    # Ca_Tn[i] = sol.y[4]

#%%
""" Compute HRT and TTP [s] """
# pk = np.argmax(active_state)
# TTP = time_dt[pk]
# for n in range(len(time_dt)):
#     if n > 259 and active_state[0,n] < 0.000001:
#         end = n+1
#         break
    
# HRT = (time_dt[end] - time_dt[pk])/2   
    
#%%------------------------------------------------------------------------------
# Visual validation
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, active_state[0,:], 'b--')
# plt.plot(time_dt[pk], active_state[0,pk], 'r*')
# plt.plot(time_dt[end], active_state[0,end], 'r*')
plt.xlabel('Time [s]')
plt.ylabel('Active state')
plt.legend()
plt.grid()
plt.show()
    
    