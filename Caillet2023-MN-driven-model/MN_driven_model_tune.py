"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue Jun  4 11:03:08 2024
___________________________________

MN-driven model with SE and PEE adapted for BB tests of Millard et. al 2023
"""

" Choose frequency (0 = 1Hz, 1  = 10Hz, 2 = 20Hz, 3 = 40Hz) "
fr = 1

" Choose l_MT length variation (0 = 0mm, 1  = 8mm, 2 = 16mm) "
s = 2

" Choose the .csv corresponding file path"
path = "16_10Hz.csv"  #.csv files locations

" Saving the simulations (y/n)? "
save = 'n'

#%%
import sys
sys.path.insert(0,'Modules_subBB')
import os
cwd = os.getcwd()

#%%
import numpy as np
import scipy as sp
import pandas as pd
import scipy.interpolate
from scipy import signal
from scipy.signal import find_peaks
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
from MU_bound_calcium_MOD import MU_bound_calcium_func
from MU_active_state_MOD import MU_active_state_func
from Force_Length_MOD import Force_Length_func
from F0MU_distrib_MOD import F0MU_distrib_func

#%%
" Time, n. of MUs and MVC paramters settings "

dt = 0.0001 # time step (x-data)
fd = [1, 10, 20, 40] # stimulation freq. Hz d.r. 
fd = fd[fr] #choice based on given input
T = 1/fd # correspondend d.r. period
Nr = 1 # n. of an average complete cat soleus MUs pool
muscle_F0M = 25.1 # muscle max. isom. force [N]

#%%
" Load time, displacement and force form BB tests "

#os.chdir("C:\\Users\\Andrea\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation") # max activation BB dir path
# os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation")
    
# disp_1 = np.genfromtxt('displacement_1mm.dat', delimiter='') #BB time & displacement data
# disp_2 = np.genfromtxt('displacement_8mm.dat', delimiter='') #BB time & displacement data
# disp_1_int = sp.interpolate.interp1d(disp_1[:,0], disp_1[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate with time_dt+1 points (l_MT must be longer)
# disp_2_int = sp.interpolate.interp1d(disp_2[:,0], disp_2[:,1], kind='cubic')(np.arange(0,2+dt,dt))
    
# force_bb = (np.genfromtxt('force_forcedot_trial'+str(s+1)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
# force_bb_disp = (np.genfromtxt('force_trial7.dat', delimiter=''))
    
# os.chdir(cwd) 

#%%
" Load Joyce 1969 FV curve data (35 Hz) for tuning "

# os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\literature_data\\forcevelocity")

# path = "Joyce1969_CatSoleusForceVelocity.csv"  #.csv files locations
# data_FV = pd.read_csv(path, delimiter = ',')
# data_FV = data_FV.to_numpy() 
# time_FV = np.arange(data_FV[0,0], data_FV[len(data_FV)-1,0], 0.01)

# data_FV = sp.interpolate.interp1d(data_FV[:,0], data_FV[:,1], kind='cubic')(time_FV)
# data_FV = (data_FV*9.81)/19.21 # Kg to N and normalized by F0M
# time_FV = time_FV/30

# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_FV, data_FV)

# plt.grid()

# os.chdir(cwd)

#%%    
" Load digitized data from Kim & Sandercock 2017 "
    
os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code")
    
data = pd.read_csv(path, delimiter = ' ', dtype = float)
data = data.to_numpy() 
data[:,0] = data[:,0]-data[0,0] # offset time due to manual digitalization

a_test_x = np.arange(data[0,0],data[len(data)-1,0],dt)

t_end = a_test_x[-1] # total seconds
#time_dt = np.arange(0, t_end, dt) # time for BB
time_dt = a_test_x

a_test = sp.interpolate.interp1d(data[:,0], data[:,1], kind='cubic')(a_test_x)  # cubic interpolation to smooth data
a_test[a_test < 0.000001] = 0  #set negative values to 0 

pks, _ = find_peaks(a_test, distance = 200) # find force peaks to sort of indentify t_end of stimulation (last peak)

os.chdir(cwd)

#%%
" NON-LINEAR TUNING of active state coefficients (depending on d.r.) "

# x = [10, 20, 30] # freqs.

# y1 = [2.1*10**-6, 1*10**-6, 2.1*10**-6]
# y1 = [2.1*10**-6, 1*10**-6, 2.1*10**-6]
# y2 = [0.7, 0.7, 0.2]
# y3 = [0.04, 0.04, 0.02]

# y1 = [4*10**-6, 1*10**-6, 2.3*10**-6]
# y2 = [0.4, 0.8, 0.3]
# y3 = [0.09, 0.04, 0.04]

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
            
# xval = np.linspace(10, 30, len(time_dt))

# p1 = np.polyfit(x, y1, 2) 
# Ca_max = np.polyval(p1, freq_long)
# p2 = np.polyfit(x, y2, 2) 
# t_up = np.polyval(p2, freq_long)
# p3 = np.polyfit(x, y3, 2) 
# t_down = np.polyval(p3, freq_long)

# plt.subplot(5,1,5)
# plt.plot(xval, d1_tuned, label = 'd1')
# plt.plot(xval, d3_tuned, label = 'd3')
# plt.xlabel('Freq. [hz]')
# plt.ylabel('A(t) coeffs')           
# plt.legend()
# plt.grid()

#%% 
" CREATE D.R. ARRAY "

" Based on predefined d.r. "

for i in range(len(a_test)): # first find the initial stimulation instant
    if a_test[i] > 0:
        t_0_stim = a_test_x[i]  # save relative initial instant
        ind_0 = i
        break
    
t_end_stim = a_test_x[pks[len(pks)-1]] # t_end of stimulation approx. as last force peak
    
sp_matrix = np.empty((Nr, int((t_end_stim-t_0_stim)/T)+1), dtype=float)
disch = np.arange(t_0_stim, t_end_stim, T) # create array of discharge times at 70 Hz
for i in range(Nr): # append Nr times
    sp_matrix[i] = disch
    
#%%
" LENGTH & DISPLACEMENT PARAMETERS "

l_T_slack = 65 # Tendon slack length (mm)
l_M_opt = 30 # Optimal fiber length (mm)
l_M_0 = 1
vmax = 10*l_M_opt
alpha_0 = 7.5*np.pi/180 # initial pennation (pennation should make less than 2 % difference)
delta_l_MT = [0, 8, 16] # Displacement from l_MT with l_M = opt
l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0) - delta_l_MT[s]  # Musculo-tendon length (mm)
                                                        
#ISOMETRIC CASE
l_MT = np.zeros((len(time_dt)+1), dtype=object) 
l_MT = l_MT + l_MT_0

#DYNAMIC CASE (displacement applied)
#l_MT = l_MT_0 + (disp_1_int+8) # scaled MT length
#l_MT = l_MT_0 + (disp_2_int+8) # scaled MT length

#______________________________________________________________________________
" ARRAYS PREALOCATION "

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
Ca_Tn = np.empty((Nr, len(time_dt)), dtype=object) # Ca-Tn bound course

#______________________________________________________________________________
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

print('There are ', Nr, ' discharging MUs in this simulation.')
F0MU_distribution = F0MU_distrib_func(Nr, muscle_F0M) # F0MU distribution across the sample of MUs

for i in range (Nr):  
    print('Computing Force for MU n°', str(i+1))
       
    MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
    Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
    
    def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha, vmax):
       
        if int(t/dt) == 0: # initial pennation angle
            alpha[int(t/dt)] = alpha_0
            
        l_T = l_MT[int(t/dt)] - (y[6])*np.cos(alpha[int(t/dt)]) # tendon length
        eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
        SE_force = T_force(eps_T) # tendon force
        PE_force = PEE_force(y[6]/l_M_opt) # passive el. force    
        CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
        alpha[int((t+dt)/dt)] = penn_ang(l_MT[int((t+dt)/dt)], y[6], l_T, l_M_0*l_M_opt, alpha_0) # update pennation angle
        
        dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
        dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0], y[6]/l_M_opt, MU_type, Matrix_AP) # Free Ca (remember to avoid negligible negative values)
        
        ddeltadt = MU_bound_calcium_func(t, y, y[2], y[6]/l_M_opt, MU_type, Matrix_AP) # Ca-Tn
            
        dadt = MU_active_state_func(t, y[4], y[5]) # Active state
        
        FL = Force_Length_func(y[6]/l_M_opt, y[5]) # F-L relationship (*active state)
            
        dldt = velo_fFV(t, y, CE_force/(FL*y[5]), FL, y[5], y[6], MU_type, vmax) # velocity 
            
        return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, ddeltadt, dadt, dldt]
     
    
    y0 = [0, 0, 0, 0, 0, 1*10**-9, l_M_opt] # set initial states (active state can't be 0 otherwise you'll divide by 0 in FV)
    p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha[i,:], vmax) # set ODE parameters
    sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/4) # solve IVP
    
    active_state[i,:] = sol.y[5]  # get active state
    l_M[i,:] = sol.y[6]  # get l_M

    # MUAP_nerve[i,:] = sol.y[0]
    # free_Ca[i,:] = sol.y[2]
    # Ca_Tn[i,:] = sol.y[4]
   
    # now recalculate data based on l_M values..
    alpha[i,0] = alpha_0
    for l in range(len(time_dt)):
        l_T[i,l] = l_MT[l] - (l_M[i,l])*np.cos(alpha[i,l]) # tendon length
        eps_T[i,l] = (l_T[i,l]-l_T_slack)/l_T_slack # new tendon strain    
        SE_force[i,l] = T_force(eps_T[i,l]) # tendon force
        PE_force[i,l] = PEE_force(l_M[i,l]/l_M_opt) # passive el. force 
        CE_force[i,l] = SE_force[i,l]/np.cos(alpha[i,l]) - PE_force[i,l] # contractile element force
        FL_force[i,l] = Force_Length_func(l_M[i,l]/l_M_opt, active_state[i,l])
        vel[i,l] = velo_fFV(0, 0, CE_force[i,l]/(FL_force[i,l]*active_state[i,l]), FL_force[i,l], active_state[i,l], l_M[i,l], MU_type, vmax)
        F_M[i,l] = f_fFV(vel[i,l]/vmax, FL_force[i,l], active_state[i,l], l_M[i,l]/l_M_opt, MU_type)
        
        alpha[i,l+1] = penn_ang(l_MT[l+1], l_M[i,l], l_T[i,l], l_M_opt, alpha_0) # update pennation angle 
      
MU_Force_list = active_state*F_M*FL_force + PE_force

F_MU_list = F0MU_distribution[0:Nr,:] * MU_Force_list # Newtons
Tot_Muscle_force = F_MU_list.sum(axis=0) # Total muscle force (in Newton)

#%%
" Visual validation "

plt.rcParams['figure.dpi'] = 360
plt.subplot(2,1,1)
plt.plot(time_dt, Tot_Muscle_force, 'r', label='Simulated Force')
#plt.plot(force_bb[:,0], force_bb[:,1], 'k', label='Exp. Force')
#plt.plot(time_dt, force_bb_int, 'k--', label='Exp. Force')
plt.plot(a_test_x, a_test, 'k', label = 'Experimental force')
plt.plot(a_test_x[pks[len(pks)-1]], a_test[pks[len(pks)-1]], 'r*') # end peak
plt.plot(a_test_x[ind_0], a_test[ind_0], 'r*') # initial stimulation instant
#plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.grid()
plt.legend(loc='lower right')

plt.subplot(2,1,2) # plot force first derivative
plt.plot(time_dt, active_state[0,:])
plt.xlabel('Time [s]')        
plt.ylabel('Active state')
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


#%%----------------------------------------------------------------------------
# Saving data
if save =='y':
    os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation')

    #np.save('time_sim', time_dt, allow_pickle=True) 
    #np.save('time_exp', time, allow_pickle=True) 
    #np.save('MU_act_list_pennation', active_state, allow_pickle=True) 
    #np.save('MU_force_list_pennation', F_MU_list, allow_pickle=True) 
    #np.save('F0MU_distrib', F0MU_distribution, allow_pickle=True) 
    np.save('R0_isom', Tot_Muscle_force, allow_pickle=True)
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