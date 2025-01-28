"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue Jun  4 11:03:08 2024
___________________________________

MN-driven model with SE and PEE adapted for BB tests of Millard et. al 2023
"""

""" Choose amplitude scale as well (0-2) """
s = 2

""" Saving the simulations (y/n)? """
save = 'n'

#%%
import sys
sys.path.insert(0,'Modules_subBB')
import os
cwd = os.getcwd()

#%%
import numpy as np
import scipy as sp
import scipy.interpolate
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import minimize
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pennation_angle import penn_ang
from PE_force import PEE_force
from Tendon_force import T_force
from velocity_fFV import velo_fFV
from force_fFV import f_fFV
from MU_type_id_MOD import MU_type_id_func
from MU_AP_MOD import MU_AP_func
from MU_free_Ca_MOD import MU_free_Ca_func
#from MU_bound_calcium_MOD import MU_bound_calcium_func
from MU_active_state_MOD import MU_active_state_func
from Force_Length_MOD import Force_Length_func
from F0MU_distrib_MOD import F0MU_distrib_func

#%% 
" Object function for OPTIMIZATION "
x0 = [9, 13, 5.4*10**-6]
bnds = ((5, 20), (5, 20), (1*10**-6, 1*10**-5))

def obj(x): 

    #%%
    """ Load time, displacement and force form BB tests, create virtual MU spikes"""

    dt = 0.0001 # time step (x-data)
    t_end = 2 # total seconds
    time_dt = np.arange(0, t_end, dt) # time for BB
    fd = [10, 20, 30] # paper stimulation freq. Hz d.r. 
    fd = fd[s]
    T = 1/fd # correspondend d.r. period
    Nr = 1 # n. of MNs in the pool
    muscle_F0M = 25.1 # muscle max. isom. force [N]
   
    #os.chdir("C:\\Users\\Andrea\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation") # max activation BB dir path
    os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation")
    
    disp_1 = np.genfromtxt('displacement_1mm.dat', delimiter='') #BB time & displacement data
    disp_2 = np.genfromtxt('displacement_8mm.dat', delimiter='') #BB time & displacement data
    disp_1_int = sp.interpolate.interp1d(disp_1[:,0], disp_1[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate with time_dt+1 points (l_MT must be longer)
    disp_2_int = sp.interpolate.interp1d(disp_2[:,0], disp_2[:,1], kind='cubic')(np.arange(0,2+dt,dt))
    
    force_bb = (np.genfromtxt('force_forcedot_trial'+str(s+1)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
    force_bb_disp = (np.genfromtxt('force_trial13.dat', delimiter=''))
    force_bb_disp_int = sp.interpolate.interp1d(force_bb_disp[:,0], force_bb_disp[:,1], kind='cubic')(np.arange(0,t_end,dt))
    
    os.chdir(cwd)
    
    force_bb_int = sp.interpolate.interp1d(force_bb[:,0], force_bb[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # cubic interpolation of force signal (with 20000+1 length to then differentiate)
    b, a = signal.butter(4, 120/(10000/2), 'low') # LPF over 120 Hz (applied to a signal that now is sampled at 10000 Hz)
    force_bb_int = signal.filtfilt(b, a, force_bb_int) # Filter for interpolated force signal 
    
    force_bb_diff = np.diff(force_bb_int) # get first derivative differentiating (from a 20000+1 length vector)
    
    # Select LPF cutoff freq. based on the imposed max. discharge freq. (2*mean d.r.)
    if s == 0 or s == 3:
        cutoff = 20
    elif s == 1 or s == 4:
        cutoff = 40
    elif s == 2 or s == 5:
        cutoff = 60
        
    b, a = signal.butter(4, cutoff/(10000/2), 'low') # LPF over 60 Hz = max. discharge freq. (applied to a signal that now is sampled at 10000 Hz)
    force_bb_diff = signal.filtfilt(b, a, force_bb_diff) # Filter for differentiated signal (to get true actual peaks)
    
    force_bb_int = force_bb_int[0:(int(t_end/dt))] # make force vector length 20000 again 

#%% D.R. PEAKS SELECTION
    
    pks, _ = find_peaks(-force_bb_diff, distance = 200) # find negative of force derivative (discharge instants)
    #pks, _ = find_peaks(force_bb_int, distance = 400)
    disch_start = 0
    for i in range (len(pks)-1):  # select only actual discharge times
        if force_bb_int[pks[i+1]] > 0.1 and disch_start == 0:   # select first discharge (when next force sample is positive)
            disch_start = i 
        
        elif force_bb_diff[pks[i]] == min(force_bb_diff[pks]):# select last discharge (min. of differentiated signal)
            if s == 1:
                disch_end = i
            elif s == 2:
                disch_end = i
            elif s == 0:
                disch_end = len(pks) - 6 # manual selection for now
            
    pks = pks[disch_start:disch_end] # actual discharge times indeces selected
    
#%% PLOTS
    
    # plt.rcParams['figure.dpi'] = 400
    # figure(figsize=(7, 10))
    # plt.subplot(4,1,1) # plot interpolated force
    # plt.plot(time_dt, force_bb_int, label='Exp. Force')
    # #plt.plot(time_dt, force_bb_disp_int,'k--')
    # #plt.plot(time_dt[pks], force_bb_int[pks], 'r*')
    # plt.ylabel('Force [N]', weight='bold')
    # plt.grid()
    
    # # plt.subplot(4,1,3) # plot displacement (20000+a points)
    # # plt.plot(np.arange(0, t_end+dt, dt), disp_1_int+8, label = '1mm')
    # # plt.plot(np.arange(0, t_end+dt, dt), disp_2_int+8, label = '8mm')
    # # plt.ylabel('Length [mm]')
    # # plt.legend()
    # # plt.grid()
    
    # plt.subplot(4,1,2) # plot force first derivative
    # plt.plot(time_dt, force_bb_diff, label='Force t-derivative')
    
    # plt.plot(time_dt[pks], force_bb_diff[pks], 'r*')
    
    # plt.ylabel('dF/dt [N/s]', weight='bold')
    # plt.grid()


#%%
    " NON-LINEAR TUNING of active state coefficients (depending on d.r.) "

# x = [10, 20, 30] # freqs.

# freq_long = np.zeros((len(time_dt)), dtype=object) # preallocate elongated freqs
# freq = np.round(1/np.diff(time_dt[pks])) # calculate actual freq. (will be one sample less)

# for z in range(len(freq)):
#     if freq[z] > 7 and freq[z] < 13:
#         freq[z] = 10
#     elif freq[z] > 17 and freq[z] < 23:
#         freq[z] = 20
#     elif freq[z] > 27 and freq[z] < 33:
#         freq[z] = 30

# for p in range(len(time_dt)): # for each time sample
#     for z in range(len(pks)-1): # for each peak (minus one)
#         if p >= pks[z] and p < pks[z+1]: # in between two consecutive peaks..
#             freq_long[p] = freq[z] # assign frequency
#     if p >= pks[len(pks)-1]: # only after the last peak..
#           freq_long[p] = freq[z]  # assign freq. corresponding to the lowest d1
#     # if p >= pks[len(pks)-8] and freq[len(pks)-2] > 26:
#     #     freq_long[p] = 0
#     #     #d3_tuned[p] = 0
    
#%%
    " CREATE D.R. ARRAY "

    " Based on peaks "
    # sp_matrix = np.zeros((Nr, len(pks)), dtype=float)
    # for i in range(Nr):  # append Nr times
    #     sp_matrix[i] = time_dt[pks]

    " Based on predefined d.r. "
    disch = np.arange(time_dt[pks[0]], time_dt[pks[len(pks)-1]], T) # create array of discharge times at 70 Hz
    sp_matrix = np.empty((Nr, len(disch)), dtype=float)
    for i in range(Nr): # append Nr times
        sp_matrix[i] = disch
    
#______________________________________________________________________________
    " LENGTH & DISPLACEMENT PARAMETERS "

    l_T_slack = 65 # Tendon slack length (mm)
    l_M_opt = 30 # Optimal fiber length (mm)
    l_M_0 = 1
    alpha_0 = 7.5*np.pi/180 # initial pennation (pennation should make less than 2 % difference)
    l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0) - 4 # Musculo-tendon length (mm)
    vmax = 10.5428*l_M_opt
    Kup, Kdown, Ca_max = x[0], x[1], x[2]

    #ISOMETRIC CASE
    l_MT = np.zeros((len(time_dt)+1), dtype=object) 
    l_MT = l_MT + l_MT_0

    #DYNAMIC CASE (displacement applied)
    #l_MT = l_MT_0 + (disp_1_int+8) # scaled MT length
    #l_MT = l_MT_0 + (disp_2_int+8) # scaled MT length

    #______________________________________________________________________________
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
    #MUAP_nerve = np.empty((Nr, len(time_dt)), dtype=object) # MU AP nerve signal
    free_Ca = np.empty((Nr, len(time_dt)), dtype=object) # free [Ca] course
    #Ca_Tn = np.empty((Nr, len(time_dt)), dtype=object) # Ca-Tn bound course

#______________________________________________________________________________
    " RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

    print('There are ', Nr, ' discharging MUs in this simulation.')
    F0MU_distribution = F0MU_distrib_func(Nr, muscle_F0M) # F0MU distribution across the sample of MUs

    for i in range (Nr):  # ...for each considered i-th MU
        print('Computing Force for MU n°', str(i+1))
       
        MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
        Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
    
        def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha, vmax, fd, Kup, Kdown, Ca_max):
       
            if int(t/dt) == 0: # initial pennation angle
                alpha[int(t/dt)] = alpha_0
            
            l_T = l_MT[int(t/dt)] - (y[5])*np.cos(alpha[int(t/dt)]) # tendon length
            eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
            SE_force = T_force(eps_T) # tendon force
            PE_force = PEE_force(y[5]/l_M_opt) # passive el. force    
            CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
            alpha[int((t+dt)/dt)] = penn_ang(l_MT[int((t+dt)/dt)], y[5], l_T, l_M_0*l_M_opt, alpha_0) # update pennation angle
        
            dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
            dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0], y[5]/l_M_opt, MU_type, Matrix_AP, fd) # Free Ca (remember to avoid negligible negative values)
        
            #ddeltadt = MU_bound_calcium_func(t, y, y[2], y[6]/l_M_opt, MU_type, Matrix_AP) # Ca-Tn
            
            dadt = MU_active_state_func(t, y[2], y[4], Kup, Kdown, Ca_max) # Active state
        
            FL = Force_Length_func(y[5]/l_M_opt, y[4]) # F-L relationship (*active state)
            
            dldt = velo_fFV(t, y, CE_force/(FL*y[4]), FL, y[4], y[5]/l_M_opt, MU_type, vmax) # velocity 
        
            return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dadt, dldt]
     
    
        y0 = [0, 0, 0, 0, 1*10**-9, l_M_opt] # set initial states (active state can't be 0 otherwise you'll divide by 0 in FV)
        p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha[i,:], vmax, fd, Kup, Kdown, Ca_max) # set ODE parameters
        sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/4) # solve IVP
    
        active_state[i,:] = sol.y[4]  # get active state
        l_M[i,:] = sol.y[5]  # get l_M

        # MUAP_nerve[i,:] = sol.y[0]
        free_Ca[i,:] = sol.y[2]
        #Ca_Tn[i,:] = sol.y[4]
   
        # now recalculate data based on l_M values..
        alpha[i,0] = alpha_0
        for l in range(len(time_dt)):
            l_T[i,l] = l_MT[l] - (l_M[i,l])*np.cos(alpha[i,l]) # tendon length
            eps_T[i,l] = (l_T[i,l]-l_T_slack)/l_T_slack # new tendon strain    
            SE_force[i,l] = T_force(eps_T[i,l]) # tendon force
            PE_force[i,l] = PEE_force(l_M[i,l]/l_M_opt) # passive el. force 
            CE_force[i,l] = SE_force[i,l]/np.cos(alpha[i,l]) - PE_force[i,l] # contractile element force
            # FL_force[i,l] = Force_Length_func(l_M[i,l]/l_M_opt, active_state[i,l])
            # vel[i,l] = velo_fFV(0, 0, CE_force[i,l]/(FL_force[i,l]*active_state[i,l]), FL_force[i,l], active_state[i,l], l_M[i,l]/l_M_opt, MU_type, vmax)
            # F_M[i,l] = f_fFV(vel[i,l]/vmax, FL_force[i,l], active_state[i,l], l_M[i,l]/l_M_opt, MU_type)
        
            alpha[i,l+1] = penn_ang(l_MT[l+1], l_M[i,l], l_T[i,l], l_M_opt, alpha_0) # update pennation angle 
      
    #MU_Force_list = active_state*F_M*FL_force + PE_force
    MU_Force_list = CE_force + PE_force

    F_MU_list = F0MU_distribution[0:Nr,:] * MU_Force_list # Newtons
    Tot_Muscle_force = F_MU_list.sum(axis=0) # Total muscle force (in Newton)
    
    f_obj =(np.sum(np.abs(Tot_Muscle_force/muscle_F0M - force_bb_int/muscle_F0M))/len(time_dt))*100  # % mean error (F0 units)
    #f_obj =((np.sum((Tot_Muscle_force-force_bb_int)**2))/(len(time_dt)))  # Mean quadratic error
    
    del alpha, l_T, eps_T, SE_force, CE_force, PE_force, FL_force, vel, F_M, l_M, active_state, free_Ca, MU_Force_list, F_MU_list
    
    return f_obj


res = minimize(obj, x0, method='nelder-mead', bounds=bnds,
               options={'disp': True})


#%% 
"Sensitivity functions"

# plt.rcParams['figure.dpi'] = 400
# figure(figsize=(12, 8))
# plt.subplot(1,2,1) 
# ddadt_dCa = (active_state + 15)/(5.4 * 10**-6)
# plt.plot(free_Ca[0,:], ddadt_dCa[0,:], '--k')

# # plt.subplot(1,2,2) # plot interpolated force
# # ddadt_dact = (free_Ca - 2 * active_state - 15)/(1 - 1*10**-9)
# # plt.plot(active_state[0,:], ddadt_dact[0,:], '--g')

# plt.subplot(1,2,2)
# ddadt_dK = (free_Ca - active_state)
# plt.plot(time_dt, ddadt_dK[0,:], '--g')


#%%
# " Visual validation "

# plt.subplot(4,1,1)
# plt.plot(time_dt, Tot_Muscle_force, 'r', label='Simulated Force')
# plt.plot(time_dt, force_bb_int, 'k', label='Exp. Force')
# #plt.plot(time_dt, force_bb_int, 'k--', label='Exp. Force')
# plt.legend(loc='lower right')

# plt.subplot(4,1,3) # plot force first derivative
# plt.plot(time_dt, active_state[0,:])      
# plt.ylabel('Active state', weight='bold')
# plt.grid()

# plt.subplot(4,1,4) # plot force first derivative
# plt.plot(time_dt, free_Ca[0,:]*10**6, 'g')   

# pks_Ca, _ = find_peaks(free_Ca[0,:], distance = 200)
# pks_Ca_mol = free_Ca[0,pks_Ca]
# max_Ca_pk = np.max(pks_Ca_mol)
# plt.plot(time_dt[pks_Ca], free_Ca[0,pks_Ca]*10**6, 'r*')

# plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]', weight='bold')
# plt.grid()
# plt.xlabel('Time [s]')  

# # plt.subplot(4,1,4) # plot force first derivative
# # plt.plot(time_dt, Ca_Tn[0,:]*10**6, 'm')       
# # plt.ylabel('CaTn [$\mu$M]', weight='bold')
# # plt.grid()
# # plt.xlabel('Time [s]')  

# plt.suptitle('Isometric trial, %i Hz' %fd, weight='bold', y=0.9)

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
# Saving data
# if save =='y':
#     os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation')

#     #np.save('time_sim', time_dt, allow_pickle=True) 
#     #np.save('time_exp', time, allow_pickle=True) 
#     #np.save('MU_act_list_pennation', active_state, allow_pickle=True) 
#     #np.save('MU_force_list_pennation', F_MU_list, allow_pickle=True) 
#     #np.save('F0MU_distrib', F0MU_distribution, allow_pickle=True) 
#     np.save('R0_isom', Tot_Muscle_force, allow_pickle=True)
#     #np.save('scaled_exp_force_in_N', Exp_muscle_force, allow_pickle=True) 
#     # np.save('Tendon_length_maxBB', l_T[2,:], allow_pickle=True)
#     # np.save('Tendon_strain_maxBB', eps_T[2,:], allow_pickle=True)
#     # np.save('Tendon_force_maxBB', SE_force[2,:], allow_pickle=True)
#     # np.save('PEE_force_maxBB', PE_force[2,:], allow_pickle=True)
    
#     #encore...
#     #     np.save('MNAP_list', MN_AP_list, allow_pickle=True) 
#     #     np.save('MUAP_list', MUAP_nerve, allow_pickle=True) 
#     #     np.save('freeCa_list', free_Ca, allow_pickle=True) 
#     #     np.save('boundCa_list', Ca_Tn, allow_pickle=True)         
     
# os.chdir(cwd)    