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

#%%
import numpy as np
import scipy as sp
import scipy.interpolate
from scipy import signal
from scipy.signal import find_peaks
#from scipy.optimize import minimize
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from pennation_angle import penn_ang
from PE_force import PEE_force
from Tendon_force import T_force
from velocity_fFV import velo_fFV
from yielding import Yield
from force_fFV import f_fFV
from MU_type_id_MOD import MU_type_id_func
from MU_AP_MOD import MU_AP_func
from MU_free_Ca_MOD import MU_free_Ca_func
from MU_active_state_MOD import MU_active_state_func
from Force_Length_MOD import Force_Length_func
from F0MU_distrib_MOD import F0MU_distrib_func

#%%

""" D.r. preset """
s = 4

os.chdir('C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\pyHatze\\Caillet2023-MN-driven-model\\Stimulation_inputs')

if s == 0:
    pks = np.load('10_c_index.npy')
    disch = np.load('10_c_times.npy')
elif s == 1:
    pks = np.load('20_c_index.npy')
    disch = np.load('20_c_times.npy')
elif s == 2:
    pks = np.load('30_c_index.npy')
    disch = np.load('30_c_times.npy')
elif s == 3:
    pks = np.load('10_v_index.npy')
    disch = np.load('10_v_times.npy')
elif s == 4:
    pks = np.load('20_v_index.npy')
    disch = np.load('20_v_times.npy')
elif s == 5:
    pks = np.load('30_v_index.npy')
    disch = np.load('30_v_times.npy')   

os.chdir(cwd)


""" Saving the simulations (y/n)? """
save = 'n'

#%%
" Load time, displacement and force form BB tests, create virtual MU spikes "

dt = 0.0001 # time step (x-data)
t_end = 0.16 # total seconds
fd = [10, 20, 40, 10, 20, 30] # paper stimulation freq. Hz d.r. 

# if d.r. is fixed (first 3 trials)
fd = fd[0]
T = 1/fd # correspondend d.r. period
    
Nr = 1 # n. of MNs in the pool
#muscle_F0M = 25.1 # muscle max. isom. force [N]
muscle_F0M = 23
   
#os.chdir("C:\\Users\\Andrea\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation") # max activation BB dir path
#os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation")
    
# disp_1 = np.genfromtxt('displacement_1mm.dat', delimiter='') #BB time & displacement data
# disp_2 = np.genfromtxt('displacement_8mm.dat', delimiter='') #BB time & displacement data
# disp_1_int = sp.interpolate.interp1d(disp_1[:,0], disp_1[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate with time_dt+1 points (l_MT must be longer)
# disp_2_int = sp.interpolate.interp1d(disp_2[:,0], disp_2[:,1], kind='cubic')(np.arange(0,2+dt,dt))

#force_bb = (np.genfromtxt('force_forcedot_trial'+str(s+1)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
 
#os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\fixedfreq_yielding")
#os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\fixedfreq_yielding")

os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\benchmark_input\\fast\\muscle\\dyn')

path = 'disp_short'
pathh = path + '.csv'
exp_force = pd.read_csv(pathh, delimiter = ' ', decimal = ',')
exp_force = exp_force.to_numpy() 
#exp_force = exp_force[0:-1,:] 
n = 100
start_time = np.linspace(0, exp_force[0,0]-0.001, n)  # n equidistant values from 0 to t_end
start = np.column_stack((start_time, np.ones(n)))

#exp_force = np.vstack(([0,0], [0.016,0.02], exp_force[8:-1,:], [t_end-0.1, 0], [t_end-0.07, 0], [t_end, 0])) # add zero final value
exp_force = np.vstack((start, exp_force)) # add zero final value
#exp_force[:,0] = exp_force[:,0] - exp_force[0,0] 

for i in range(len(exp_force)):
    if exp_force[i,0] < 0.107:
        exp_force[i,1] = 0.1

time_dt = np.arange(0, t_end, dt) # time for BB
exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(time_dt)
# for i in range(len(exp_force)):
#     if exp_force[i] < 0:
#         exp_force[i] = 0

#idx = np.argmin(exp_force[100:-1000])

# for i in range(len(exp_force)):
#     if i > idx:
#         exp_force[i] = 0

# disch = np.arange(0, 0.54, 1/150)
#disch = time_dt[start]


# from scipy.signal import butter, filtfilt
# b, a = butter(2, 50/(0.5 * 1/dt), btype='low')
# exp_force = filtfilt(b, a, exp_force)

# disch_indices = np.searchsorted(time_dt, disch)
# force_at_disch = exp_force[disch_indices]

# for i in range(len(exp_force)):
#     if exp_force[i] < 0:
#         exp_force[i] = 0

# os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\benchmark_input\\fast')

# exp_force = np.load('iso_15_1_interp.npy')
# exp_force = exp_force[0:6000]

plt.rcParams['figure.dpi'] = 400
plt.plot(time_dt, exp_force, 'k')
#plt.scatter(disch, force_at_disch)
plt.grid()


#%%
np.save(path + '_interp', exp_force, allow_pickle=True) 
#np.save(path + '_times', disch, allow_pickle=True)

#%%



#force_bb_disp = (np.genfromtxt('force_trial'+str(s)+'.dat', delimiter=''))
# force_bb_disp = (np.genfromtxt('force_trial5.dat', delimiter=''))
# force_bb_disp_int = sp.interpolate.interp1d(force_bb_disp[:,0], force_bb_disp[:,1], kind='cubic')(np.arange(0,t_end,dt))
    
os.chdir(cwd)
    
# force_bb_int = sp.interpolate.interp1d(force_bb[:,0], force_bb[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # cubic interpolation of force signal (with 20000+1 length to then differentiate)
# b, a = signal.butter(4, 120/(10000/2), 'low') # LPF over 120 Hz (applied to a signal that now is sampled at 10000 Hz)
# force_bb_int = signal.filtfilt(b, a, force_bb_int) # Filter for interpolated force signal 
    
# force_bb_diff = np.diff(force_bb_int) # get first derivative differentiating (from a 20000+1 length vector)
  

#%%  
# Select LPF cutoff freq. based on the imposed max. discharge freq. (2*mean d.r.)
if s == 0:
    cutoff = 20
elif s == 3:
    cutoff = 30
elif s == 1:
    cutoff = 40
elif s == 4:
    cutoff = 90
elif s == 2: 
    cutoff = 60
elif s == 5:
    cutoff = 100
        
# b, a = signal.butter(4, cutoff/(10000/2), 'low') # LPF over 60 Hz = max. discharge freq. (applied to a signal that now is sampled at 10000 Hz)
# force_bb_diff = signal.filtfilt(b, a, force_bb_diff) # Filter for differentiated signal (to get true actual peaks)
    
# force_bb_int = force_bb_int[0:(int(t_end/dt))] # make force vector length 20000 again 

#%% 

# "Manual D.R. Selection, (for variable d.r. trials)"    
# if s > 2:
#     pks_d, _ = find_peaks(-force_bb_diff, distance = 100) # force derivative peaks
#     pks_f, _ = find_peaks(-force_bb_int, distance = 200)  # force peaks

#     pks_d = pks_d.tolist()
#     pks_f = pks_f.tolist()
# elif s <= 2:
#     pks, _ = find_peaks(-force_bb_diff, distance = 200) # force derivative peaks
#     #pks, _ = find_peaks(-force_bb_int, distance = 200)  # force peaks


# # to be maually merged...
# if s == 3:
#     # Create a list of arrays to concatenate
#     pks = pks_f[3:12]
#     pks[0] = 1070
#     pks_d[3] = 1600
#     pks_d[5] = 3040
#     pksadd1 = 12400
#     pksadd2 = 12650
#     pksadd3 = 13050
#     pks.append(pksadd1)
#     pks.append(pksadd2)
#     pks.append(pksadd3)
#     for i in [3, 5]:
#         pks.append(pks_d[i])
#     pks = np.sort(pks)    
# elif s == 4:
#     # Create a list of arrays to concatenate
#     pks = pks_f[3:13]
#     pks.append(pks_f[15])
#     pks.append(pks_f[16])
#     pks.append(pks_f[17])
#     pks.append(pks_f[19])
#     pksadd = 11070
#     pks.append(pksadd)
#     for i in [34, 36, 38, 39, 45, 64]:
#         pks.append(pks_d[i])
    
#     pks = np.sort(pks)   
# elif s == 5:
#     # Create a list of arrays to concatenate
#     pks = pks_f[4:13]
#     pks.append(pks_f[15])
#     pks.append(pks_f[16])
#     pks.append(pks_f[18])
#     for i in [8, 10, 12, 13, 14, 17, 18, 20, 27, 28, 30, 32, 38, 43, 45, 48, 54, 56, 59, 62, 67, 70, 73, 75, 77, 79]:
#         pks.append(pks_d[i]) # not sure about 20, 37, 68, 69
#     pks = np.sort(pks) 
    
# if s == 5:
#     pks[37] = 15000
    
#%%    
# "Only for the first 3 trials"
# if s < 3:
#     disch_start = 0
#     for i in range (len(pks)-1):  # select only actual discharge times
#         if force_bb_int[pks[i+1]] > 0.1 and disch_start == 0:   # select first discharge (when next force sample is positive)
#             disch_start = i 
        
#         elif force_bb_diff[pks[i]] == min(force_bb_diff[pks]):   # select last discharge (min. of differentiated signal)
#             if s == 1:
#                 disch_end = i
#             elif s == 2:
#                 disch_end = i
#             elif s == 0:
#                 disch_end = len(pks) - 6 # manual selection for now
            
#     pks = pks[disch_start:disch_end] # actual discharge times indeces selected

#%%    
# plt.rcParams['figure.dpi'] = 400
# #figure(figsize=(10, 5))
# #plt.subplot(4,1,1) # plot interpolated force
# plt.plot(time_dt, force_bb_int, label='Exp. Force')
# #plt.plot(time_dt, force_bb_disp_int,'k--')
# #plt.plot(time_dt[pks_f], force_bb_int[pks_f], 'r*')
# plt.plot(time_dt[pks], force_bb_int[pks], 'g*', linewidth = 5)
# plt.ylabel('Force [N]', weight='bold')
# plt.grid()
    
# plt.subplot(4,1,3) # plot displacement (20000+a points)
# plt.plot(np.arange(0, t_end+dt, dt), disp_1_int+8, label = '1mm')
# plt.plot(np.arange(0, t_end+dt, dt), disp_2_int+8, label = '8mm')
# plt.ylabel('Length [mm]')
# plt.legend()
# plt.grid()
    
# plt.subplot(4,1,2) # plot force first derivative
# plt.plot(time_dt, force_bb_diff, label='Force t-derivative')
# plt.plot(time_dt[pks_d], force_bb_diff[pks_d], 'r*')
    
# plt.ylabel('dF/dt [N/s]', weight='bold')
# plt.grid()

#%%
" CREATE D.R. ARRAY "

# if s < 3: # fixed d.r.
#     " Based on predefined d.r. "
#     if s == 0:  # w adjusts for electrophysiological delays
#         w = 220
#     elif s == 1 or s == 2:
#         w = 100
        
#     disch = np.arange(time_dt[pks[0] + w], time_dt[pks[len(pks)-1] + w], T) # create array of discharge times 
#     sp_matrix = np.empty((Nr, len(disch)), dtype=float)
#     for i in range(Nr):   # append Nr times
#         sp_matrix[i] = disch

        
# else:  # variable d.r.
#     " Based on peaks "
#     sp_matrix = np.zeros((Nr, len(pks)), dtype=float)
#     disch = time_dt[pks]
#     for i in range(Nr):  # append Nr times
#         sp_matrix[i] = disch
           
#%%
" LENGTH & DISPLACEMENT PARAMETERS "

l_T_slack = 65 # Tendon slack length (mm)
l_M_opt = 30 # Optimal fiber length (mm)
l_M_0 = 1
alpha_0 = 7.5*np.pi/180 # initial pennation (pennation should make less than 2 % difference)
l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0) - 4 # Musculo-tendon length (mm)
vmax = 10.5428*l_M_opt

#Kup, Kdown, Ca_max = 13.79, 9.8329, 6.0877*10**-6   # results from optimization at 30Hz
#Kup, Kdown, Ca_max = 13.79, 15.8329, 5.5877*10**-6

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
MUAP_nerve = np.empty((Nr, len(time_dt)), dtype=object) # MU AP nerve signal
free_Ca = np.empty((Nr, len(time_dt)), dtype=object) # free [Ca] course
yielding = np.empty((Nr, len(time_dt)), dtype=object) # yieldin coeff.

#______________________________________________________________________________
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

sp_matrix = np.empty((Nr, len(disch)), dtype=float)
for i in range(Nr):  # append Nr times
    sp_matrix[i] = disch

print('There are ', Nr, ' discharging MUs in this simulation.')
F0MU_distribution = F0MU_distrib_func(Nr, muscle_F0M) # F0MU distribution across the sample of MUs

for i in range (Nr):  # ...for each considered i-th MU
    print('Computing Force for MU n°', str(i+1))
       
    MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
    Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
    
    def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha, vmax):
       
        if int(t/dt) == 0:   # initial pennation angle
            alpha[int(t/dt)] = alpha_0
            
        l_T = l_MT[int(t/dt)] - (y[5])*np.cos(alpha[int(t/dt)]) # tendon length
        eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
        SE_force = T_force(eps_T) # tendon force
        PE_force = PEE_force(y[5]/l_M_opt) # passive el. force    
        CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
        alpha[int((t+dt)/dt)] = penn_ang(l_MT[int((t+dt)/dt)], y[5], l_T, l_M_0*l_M_opt, alpha_0) # update pennation angle
        
        dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
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
        FL_force[i,l] = Force_Length_func(l_M[i,l]/l_M_opt, active_state[i,l])
        vel[i,l] = velo_fFV(0, 0, CE_force[i,l]/(FL_force[i,l]), FL_force[i,l], active_state[i,l], l_M[i,l]/l_M_opt, MU_type, vmax)
        F_M[i,l] = f_fFV(vel[i,l]/vmax, FL_force[i,l], active_state[i,l], l_M[i,l]/l_M_opt, MU_type)
        
        alpha[i,l+1] = penn_ang(l_MT[l+1], l_M[i,l], l_T[i,l], l_M_opt, alpha_0) # update pennation angle 
      
MU_Force_list = yielding*active_state*CE_force + PE_force

F_MU_list = F0MU_distribution[0:Nr,:] * MU_Force_list # Newtons
Tot_Muscle_force = F_MU_list.sum(axis=0) # Total muscle force (in Newton)
    

#%%
" Visual validation "

plt.rcParams['figure.dpi'] = 400
figure(figsize=(7, 10))

plt.subplot(3,1,1)
plt.plot(time_dt, Tot_Muscle_force, 'r', label='Simulated Force')
plt.plot(time_dt, force_bb_int, 'k', label='Exp. Force')
plt.plot(disch, force_bb_int[pks], 'k*', linewidth = 5)
# plt.plot(time_dt, force_bb_disp_int, 'k--', label='Exp. Force')
plt.ylabel('Force [N]', weight='bold')
plt.legend(loc='lower right')
plt.grid()

# plt.subplot(3,1,2) # plot force first derivative
# plt.plot(time_dt, active_state[0,:])      
# plt.ylabel('Active state', weight='bold')
# plt.grid()

plt.subplot(3,1,3) # plot force first derivative
plt.plot(time_dt, (free_Ca[0,:]*10**6), 'g')   

pks_Ca, _ = find_peaks(free_Ca[0,:], distance = 200)
pks_Ca_mol = free_Ca[0,pks_Ca]
# max_Ca_pk = np.max(pks_Ca_mol)
# plt.plot(time_dt[pks_Ca], free_Ca[0,pks_Ca]*10**6, 'r*')

plt.ylabel('Trans. [$Ca^{2+}$] [$\mu$M]', weight='bold')
plt.grid()
# plt.xlabel('Time [s]', weight='bold')  

# plt.subplot(4,1,4) # plot force first derivative
# plt.plot(time_dt, Ca_Tn[0,:]*10**6, 'm')       
# plt.ylabel('CaTn [$\mu$M]', weight='bold')
# plt.grid()
# plt.xlabel('Time [s]')  

#plt.suptitle('Isometric trial, %i Hz' %fd, weight='bold', y=0.9)
plt.subplot(3,1,2)
plt.plot(time_dt, active_state[0,:], 'm', label='Simulated')
#plt.plot(time_dt, force_bb_int/(CE_force[0,:]*25.1), 'k', label='Inversely calculated')
#plt.plot(time_dt, force_bb_disp_int, 'k--', label='Exp. Force')
plt.ylabel('Active state', weight='bold')
plt.legend(loc='lower right')
plt.grid()
plt.xlabel('Time [s]', weight='bold')  

#%%
# ERROR METRICS
abs_p_err = np.empty((len(force_bb)), dtype=object)
for r in range(len(force_bb)):
    for rr in range(len(time_dt)):
        if force_bb[r,0] - time_dt[rr] < 0.0001:
            abs_p_err[r] = np.abs(force_bb[r,1]-Tot_Muscle_force[rr])*100
            break
        
mean_err = np.mean(abs_p_err) 
max_err = np.max(abs_p_err)

abs_p_err_int = np.empty((len(Tot_Muscle_force)), dtype=object)
abs_p_err_int = np.abs(force_bb_int/muscle_F0M - Tot_Muscle_force/muscle_F0M)*100
abs_p_err_int_disp = np.abs(force_bb_disp_int/muscle_F0M - Tot_Muscle_force/muscle_F0M)*100

mean_err_int = np.mean(abs_p_err_int)
max_err_int = np.max(abs_p_err_int)

mean_err_int_disp = np.mean(abs_p_err_int_disp)
max_err_int_disp = np.max(abs_p_err_int_disp)

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