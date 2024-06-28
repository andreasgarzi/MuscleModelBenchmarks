"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Fri May 24 08:49:54 2024
_______________________________________________________________________________

Description: MN-driven MU resolution muscle model with added PEEs and elastic tendons (fFV included).
Each MU has its own PEE and SE (tendon) and develops force independently on the others.
"""

""" AVAILABLE EXPERIMENTAL DATASETS OF INPUT SPIKE TRAINS (to uncomment) """
test = 'S1_30_256' #30% MVC - 256 electrodes
# test = 'S1_30_64L'
# test = 'S1_30_36L'
#test = 'S1_50_256'
# test = 'S1_50_64L'

""" TYPE OF NEURAL INPUTS (experimental or reconstructed) """
Input_MU_pop='Nr' # the inputs are the Nr experimental spike trains
# Input_MU_pop='400' # the inputs are the spike trains of the reconstructed population of 400 MUs

""" If Input_MU_pop='Nr', type of distribution chosen for the f0^MU parameter """
#f0_MU_distrib = 'evenly'  # Evenly distributed across pool
f0_MU_distrib = 'identified' # identified with recruitment threshold

""" Saving the simulations? """
save = 'y'
# save='n'

#______________________________________________________________________________

import sys
sys.path.insert(0,'Modules')
from pathlib import Path
root = Path(".")                     
path_to_data = root / "Results"
import os
cwd = os.getcwd()
#______________________________________________________________________________

# Libraries and functions
import numpy as np
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
" Loading the pre-processed Experimental (Nr MUs) and reconstructed (N = 400 MUs) populations of MU spike trains + additional parameters "

time, time_dt, muscle, MVC, Transd_Force, muscle_F0M, Nb_MN, MN_pop, Real_MN_pop, exp_disch_times, Firing_times_sim, range_start, range_stop, t_start, plateau_time1, plateau_time2, end_force, d, dt, fs = load_Input_Data_func(test, path_to_data)

" Choose between N or Nr MNs based on the Input_MU_pop. "

# - N case: resize the 400-MNs matrix to select the actual firing MNs
# - Nr case: converts samples --> seconds 
Nr, sp_matrix = input_spike_trains_func(Input_MU_pop, Nb_MN, Firing_times_sim, exp_disch_times, fs)
print('There are ', Nr, ' discharging MUs in this simulation.')

#------------------------------------------------------------------------------
# Experimental isolated TA Force from Experimental total Force and Muscle Co-Contraction (bEMG)
# N.B. contains force --> torque conversion based on NEG1 geometry and TA subject-sp. moment arm
Exp_muscle_force, F_TA_norm = F_TA_func(Transd_Force, MVC, fs, range_start, range_stop, plateau_time1, plateau_time2)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# F0MU distribution across the sample of MUs
F0MU_distribution = F0MU_distrib_func(MVC, Nr, MN_pop, muscle_F0M, Input_MU_pop, Real_MN_pop, f0_MU_distrib)
#------------------------------------------------------------------------------

#______________________________________________________________________________
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

l_MT = 31.9 # Musculo-tendon length (cm)
l_T_slack = 24 # Tendon slack length (cm)
l_M_opt = 6.8 # Optimal fiber length (cm)
l_M_0 = 1.16 # Initial fibre length normalized to l_M_0 (it would be l_MT - l_ST)
alpha_0 = 11.2*np.pi/180 # initial pennation (pennation should make less than 2 % difference)

alpha = np.empty((Nr,len(time_dt)+1), dtype=object) # Pennation
l_T = np.empty((Nr,len(time_dt)), dtype=object) # Tendon length
eps_T = np.empty((Nr,len(time_dt)), dtype=object) # Tendon strain
SE_force = np.empty((Nr,len(time_dt)), dtype=object) # Tendon force
CE_force = np.empty((Nr,len(time_dt)), dtype=object) # Contractile el. force
PE_force = np.empty((Nr,len(time_dt)), dtype=object) # Parallel elastic el. force

l_M = np.empty((Nr, len(time_dt)), dtype=object) # MUs length in time
active_state = np.empty((Nr, len(time_dt)), dtype=object) # active state
MUAP_nerve = np.empty((Nr, len(time_dt)), dtype=object) # MU AP nerve signal
free_Ca = np.empty((Nr, len(time_dt)), dtype=object) # free [Ca] course
Ca_Tn = np.empty((Nr, len(time_dt)), dtype=object) # Ca-Tn bound course

MU_force = np.empty((Nr, len(time_dt)), dtype=object) # i-th MU, in dt force

# ...for each considered i-th MU 
for i in range (Nr):  
    print('Computing Force for MU n°', str(i+1))
       
    MU_type = MU_type_id_func(i, Input_MU_pop, f0_MU_distrib, Real_MN_pop, Nr)  # i-th MU type identification (fast/slow)
    Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
    
    def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha):
       
        if int(t/dt) == 0: # initial pennation angle
            alpha[int(t/dt)] = alpha_0
            
        l_T = l_MT - (y[6]*l_M_opt)*np.cos(alpha[int(t/dt)]) # tendon length
        eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
        SE_force = T_force(eps_T) # tendon force
        PE_force = PEE_force(y[6]) # passive el. force    
        CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile element force
        alpha[int((t+dt)/dt)] = penn_ang(l_MT, y[6], l_T, l_M_0, alpha_0) # update pennation angle
        
        dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
        dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0]*0.85, y[6], MU_type, Matrix_AP) # Free Ca (remember to avoid negligible negative values)
        
        if y[2] < 0:   # avoid negative values
            y[2] = 0
        
        ddeltadt = MU_bound_calcium_func(t, y, y[2], y[6], MU_type, Matrix_AP) # Ca-Tn
            
        dadt = MU_active_state_func(t, y, y[4]) # Active state
            
        FL_force = Force_Length_func(y[6], y[5])*y[5] # F-L relationship (*active state)
            
        dldt = velo_fFV(t, y, CE_force, FL_force, y[5], y[6], MU_type) # velocity 
            
        return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, ddeltadt, dadt, dldt]
     
    
    y0 = [0, 0, 0, 0, 0, 0, l_M_0] # set initial states
    p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha[i,:]) # set ODE parameters
    sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/2) # solve IVP
    
    active_state[i,:] = sol.y[5]  # get active state
    l_M[i,:] = sol.y[6]  # get l_M
    
    # MUAP_nerve[i,:] = sol.y[0]
    # free_Ca[i,:] = sol.y[2]
    # Ca_Tn[i,:] = sol.y[4]
   
    # now recalculate data based on l_M values..
    alpha[i,0] = alpha_0
    for l in range(len(time_dt)):
        l_T[i,l] = l_MT - (l_M[i,l]*l_M_opt)*np.cos(alpha[i,l])
        eps_T[i,l] = (l_T[i,l]-l_T_slack)/l_T_slack # new tendon strain    
        SE_force[i,l] = T_force(eps_T[i,l]) # tendon force
        PE_force[i,l] = PEE_force(l_M[i,l]) # passive el. force    
        CE_force[i,l] = SE_force[i,l]/np.cos(alpha[i,l]) - PE_force[i,l] # contractile element force
        alpha[i,l+1] = penn_ang(l_MT, l_M[i,l], l_T[i,l], l_M_0, alpha_0) # update pennation angle
    
    
MU_Force_list = active_state*CE_force + PE_force  # scaled MU force + PEE force     
#MU_Force_list = fibre_forces_func(Nr, muscle_F0M, F0MU_distribution, MU_force) #LPF accounting for the individual asynchronous activities of the fibers (random delays between 0-20ms)   

F_MU_list = F0MU_distribution[0:Nr,:] * MU_Force_list
Tot_Muscle_force = F_MU_list.sum(axis=0) # Total whole muscle force (in N)    


#------------------------------------------------------------------------------
# Visual validation
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, Tot_Muscle_force, 'r', label='Simulated Force')
plt.plot(time, Exp_muscle_force, 'k', label='Experimental Force')
plt.xlim(0, end_force)
plt.ylim(0, max(np.max(Exp_muscle_force), np.max(Tot_Muscle_force))*1.1)
plt.xlabel('Time [s]')
plt.ylabel('TA Force (N)')
plt.legend()
plt.grid()
plt.show()

#------------------------------------------------------------------------------
# Saving data
if save =='y':
    os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\pyHatze-Local\\Caillet2023-MN-driven-model\\Results_withTendon\\Nr_S1_30_256')
    if Input_MU_pop =='400': 
        k='400_'
        prefix = k+test+'_'
    else: 
        k='Nr_'
        prefix = k+test+'_'+f0_MU_distrib+'_'
    #np.save(prefix+'time_sim', time_dt, allow_pickle=True) 
    #np.save(prefix+'time_exp', time, allow_pickle=True) 
    #np.save(prefix+'MU_act_list_pennation', active_state, allow_pickle=True) 
    #np.save(prefix+'MU_force_list_pennation', F_MU_list, allow_pickle=True) 
    #np.save(prefix+'F0MU_distrib', F0MU_distribution, allow_pickle=True) 
    np.save(prefix+'total_muscle_force_tendon2', Tot_Muscle_force, allow_pickle=True)
    #np.save(prefix+'scaled_exp_force_in_N', Exp_muscle_force, allow_pickle=True) 
    #np.save(prefix+'Tendon_length_tendon2', l_T[2,:], allow_pickle=True)
    #np.save(prefix+'Tendon_strain_tendon2', eps_T[2,:], allow_pickle=True)
    #np.save(prefix+'Tendon_force_tendon2', SE_force[2,:], allow_pickle=True)
    #np.save(prefix+'PEE_force_tendon2', PE_force[2,:], allow_pickle=True)
    
    #encore...
    #     np.save(prefix+'MNAP_list', MN_AP_list, allow_pickle=True) 
    #     np.save(prefix+'MUAP_list', MUAP_nerve, allow_pickle=True) 
    #     np.save(prefix+'freeCa_list', free_Ca, allow_pickle=True) 
    #     np.save(prefix+'boundCa_list', Ca_Tn, allow_pickle=True)         
     
os.chdir(cwd)    