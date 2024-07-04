"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Mon Jun 10 10:02:26 2024
___________________________________

Tuning of the active state ODE coefficients based on rat soleus MU twitch.
"""

#%%____________________________________________________________________________
import sys
sys.path.insert(0,'Modules_BB')
import os
cwd = os.getcwd()
#%%____________________________________________________________________________
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.integrate import solve_ivp
from pennation_angle import penn_ang
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
from Shift_length import shift_fun
from Activity import activity
#%%____________________________________________________________________________
""" Load time, displacement and force form BB tests, create virtual MU spikes"""

dt = 0.0001 # time step (x-data)
t_end = 1 # total seconds

#t_end = 0.04 # case of 5 shocks at 100HZ

time_dt = np.arange(0, t_end, dt) # time 
fd = [10, 20, 30, 40, 50, 100, 125] # paper stimulation freq. Hz d.r. (to reach active_state = 1 always, without any recruitment)
T = [1/fd[0], 1/fd[1], 1/fd[2], 1/fd[3], 1/fd[4], 1/fd[5], 1/fd[6]]# correspondent d.r. period
Nr = 2
muscle_F0M = 1.17 # muscle max. isom. force [N]

"""1st CASE: d.r. at n*10Hz constant for 2 different MU types"""
sp_matrix10 = np.empty((Nr,int(t_end/T[0])), dtype=float)
sp_matrix20 = np.empty((Nr,int(t_end/T[1])), dtype=float)
sp_matrix30 = np.empty((Nr,int(t_end/T[2])), dtype=float)
sp_matrix40 = np.empty((Nr,int(t_end/T[3])), dtype=float)
sp_matrix50 = np.empty((Nr,int(t_end/T[4])), dtype=float)
sp_matrix100 = np.empty((Nr,int(0.05/T[5])), dtype=float) # manually impose the n. of impulses for literature comparisons
sp_matrix125 = np.empty((Nr,int(0.08/T[6])), dtype=float)

disch10 = np.arange(0,t_end,T[0]) # create array of dischare times at 10 Hz
for i in range (Nr):
    sp_matrix10[i,:] = disch10

disch20 = np.arange(0,t_end,T[1]) # create array of dischare times at 20 Hz
for i in range (Nr):
    sp_matrix20[i,:] = disch20

disch30 = np.arange(0,t_end,T[2]) # create array of dischare times at 30 Hz
for i in range (Nr):
    sp_matrix30[i,:] = disch30

disch40 = np.arange(0,t_end,T[3]) # create array of dischare times at 40 Hz
for i in range (Nr):
    sp_matrix40[i,:] = disch40

disch50 = np.arange(0,t_end,T[4]) # create array of dischare times at 50 Hz
for i in range (Nr):
    sp_matrix50[i,:] = disch50
    
disch100 = np.arange(0,0.05,T[5]) # create array of dischare times at 50 Hz
for i in range (Nr):
    sp_matrix100[i,:] = disch100
        
disch125 = np.arange(0,0.08,T[6]) # create array of dischare times at 50 Hz
for i in range (Nr):
    sp_matrix125[i,:] = disch125

"""2nd CASE: one spike only"""
# sp_matrix = np.zeros((Nr, int(t_end/T)), dtype=float)
# for i in range(Nr): # append Nr times
#     sp_matrix[i, 0] = 1

#______________________________________________________________________________      
print('There are ', Nr, ' discharging MUs in this simulation.')

# F0MU distribution across the sample of MUs
F0MU_distribution = F0MU_distrib_func(Nr, muscle_F0M)

#______________________________________________________________________________
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

l_MT = 31.9 # Musculo-tendon length (cm)
l_T_slack = 24 # Tendon slack length (cm)
l_M_opt = 6.8 # Optimal fiber length (cm)
l_M_0 = 1.6 # Initial fibre length normalized to l_M_0 (it would be l_MT - l_ST)
alpha_0 = 11.2*np.pi/180 # initial pennation (pennation should make less than 2 % difference)

alpha = np.empty((len(time_dt)+1), dtype=object) # Pennation

active_state = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # active state
free_Ca = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # free [Ca] course
l_M = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # MU lengths
#MUAP_nerve = np.empty((len(fd), Nr, len(time_dt)), dtype=object)

# ...for each frequency considered assign the correspond
for f in range (7):
    
    if f == 0:
        sp_matrix = sp_matrix10
    elif f == 1:
        sp_matrix = sp_matrix20
    elif f == 2:
        sp_matrix = sp_matrix30
    elif f == 3:
        sp_matrix = sp_matrix40
    elif f == 4:
        sp_matrix = sp_matrix50
    elif f == 5:
        sp_matrix = sp_matrix100
    elif f == 6:
        sp_matrix = sp_matrix125
        
    # ...for each considered i-th MU 
    for i in range (Nr):  
        print('Computing Force for MU n°', str(i+1))
       
        MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
        Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
    
        def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha):
       
            if int(t/dt) == 0: # initial pennation angle
                alpha[int(t/dt)] = alpha_0
            
            # l_T = l_MT - (y[5]*l_M_opt)*np.cos(alpha[int(t/dt)]) # tendon length
            # eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
            # SE_force = T_force(eps_T) # tendon force
            # PE_force = PEE_force(y[5]) # passive el. force    
            # CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
            # alpha[int((t+dt)/dt)] = penn_ang(l_MT, y[5], l_T, l_M_0, alpha_0) # update pennation angle
        
            dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
            dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0]*0.85, l_M_0, MU_type, Matrix_AP) # Free Ca (remember to avoid negligible negative values)
            
            #__________________________________________________________________
            # Rockenfeller, Gunther & Hatze's approach
            #w_l = shift_fun(y[4])
           
            #a = activity(y[2], w_l) 
            #__________________________________________________________________
            # Arnault's original ODEs
            #ddeltadt = MU_bound_calcium_func(t, y, y[2], y[6], MU_type, Matrix_AP) # Ca-Tn
            
            dadt = MU_active_state_func(t, y, y[2], MU_type) # Active state
            #__________________________________________________________________
            
            #FL_force = Force_Length_func(y[5], y[4])*y[4] # F-L relationship (*active state)
            
            #dldt = velo_fFV(t, y, CE_force, FL_force, y[4], y[5], MU_type) # velocity 
            
            #return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, ddeltadt, dadt, dldt]
            return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dadt]
    
        y0 = [0, 0, 0, 0, 0] # set initial states
        p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha) # set ODE parameters
        sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/2) # solve IVP
    
        active_state[f,i,:] = sol.y[4]  # get active state
        #l_M[f,i,:] = sol.y[5]  # get l_M
        #MUAP_nerve[f,i,:] = sol.y[0]
        # Ca_Tn[i] = sol.y[4]
        free_Ca[f,i,:] = sol.y[2] # get [Ca2+]
        
        for l in range (len(time_dt)):  # correct the negative values of free [Ca++]
            if free_Ca[f,i,l] < 0:
                free_Ca[f,i,l] = 0
    
# for f in range (5):   
#     for i in range (Nr):
#         for l in range (len(time_dt)):
        
#             active_state[f,i,l] = activity(free_Ca[f,i,l], shift_fun(l_M[f,i,l])) # get activity

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

"""Create color map and plot relationships"""
   
c1 = "#0b165e"  #dark blue
c2 = "#00eeff"  #light blue

c3 = '#e31010'  #red
c4 = '#fcff01'  #orange

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

cg1 = get_color_gradient(c1, c2, 5)
cg2 = get_color_gradient(c3, c4, 5)


#%%______________________________________________________________________________
""" Extract experimental digitized data from Rincon 2021 """

os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code") # max activation BB dir path

path_slow = "slow_23_mouse_100hz_spaceseparator.csv"  #.csv files locations
path_fast = "fast_23_mouse_100hz_spaceseparator.csv"
path_fast_35 = "fast_35_mouse_125hz.csv"

data_slow = pd.read_csv(path_slow, delimiter = ' ')
data_slow = data_slow.to_numpy()    
            
data_fast = pd.read_csv(path_fast, delimiter = ' ')
data_fast = data_fast.to_numpy()

data_fast_35 = pd.read_csv(path_fast_35, delimiter = ' ')
data_fast_35 = data_fast_35.to_numpy()

os.chdir(cwd)


# figure(figsize=(12, 10))
# plt.subplot(2,1,1)
# plt.rcParams['figure.dpi'] = 360
# plt.plot((data_slow[:,0]-data_slow[0,0])*10**-3, data_slow[:,1], 'k', label = 'Rincon 2021 (100Hz)') # offset and in seconds


# plt.rcParams['figure.dpi'] = 360
# plt.plot((data_fast_35[:,0]-data_fast_35[0,0])*10**-3, data_fast_interp[:,1], 'k', label = 'Rincon 2021 (100Hz)') # offset and in seconds

#%% Visual validation
# ACTIVATION
plt.rcParams['figure.dpi'] = 360
figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.plot(time_dt, active_state[0,0,:], color=cg1[0], label='d.r. = 10Hz, slow')
plt.plot(time_dt, active_state[1,0,:], color=cg1[1], label='d.r. = 20Hz, slow')
plt.plot(time_dt, active_state[2,0,:], color=cg1[2], label='d.r. = 30Hz, slow')
plt.plot(time_dt, active_state[3,0,:], color=cg1[3], label='d.r. = 40Hz, slow')
plt.plot(time_dt, active_state[4,0,:], color=cg1[4], label='d.r. = 50Hz, slow')

plt.ylabel('Active state')
plt.legend(loc='lower right')
plt.title('Slow type fibre')
plt.grid()

plt.subplot(2,1,2)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, active_state[0,1,:], color=cg2[0], label='d.r. = 10Hz, fast')
plt.plot(time_dt, active_state[1,1,:], color=cg2[1], label='d.r. = 20Hz, fast')
plt.plot(time_dt, active_state[2,1,:], color=cg2[2], label='d.r. = 30Hz, fast')
plt.plot(time_dt, active_state[3,1,:], color=cg2[3], label='d.r. = 40Hz, fast')
plt.plot(time_dt, active_state[4,1,:], color=cg2[4], label='d.r. = 50Hz, fast')

# plt.plot(time_dt[pk], active_state[0,pk], 'r*')
# plt.plot(time_dt[end], active_state[0,end], 'r*')
plt.xlabel('Time [s]')
plt.ylabel('Active state')
plt.title('Fast type fibre')
plt.legend(loc='lower right')
plt.grid()
plt.suptitle('MN-driven model sensitivity (activation)', weight='bold', y=0.94)
plt.show()
    
#%%
# EXCITATION
pks_slow, _ = find_peaks(-free_Ca[4,0,:]) # 3.64*10**-6
pks_fast, _ = find_peaks(-free_Ca[4,1,:]) # 5.34*10**-7

figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, free_Ca[0,0,:]*10**6, label='d.r. = 10Hz')
plt.plot(time_dt, free_Ca[1,0,:]*10**6, label='d.r. = 20Hz')
plt.plot(time_dt, free_Ca[2,0,:]*10**6, label='d.r. = 30Hz')
plt.plot(time_dt, free_Ca[3,0,:]*10**6, label='d.r. = 40Hz')
plt.plot(time_dt, free_Ca[4,0,:]*10**6, label='d.r. = 50Hz')
#plt.plot(time_dt[pks_slow], free_Ca[4,0,pks_slow]*10**6, 'r*')

plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')           
plt.legend(loc='lower right')
plt.grid()
plt.title('Slow type fibre')
plt.xlim((0, 0.3))

plt.subplot(2,1,2)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, free_Ca[0,1,:]*10**6, label='d.r. = 10Hz')
plt.plot(time_dt, free_Ca[1,1,:]*10**6, label='d.r. = 20Hz')
plt.plot(time_dt, free_Ca[2,1,:]*10**6, label='d.r. = 30Hz')
plt.plot(time_dt, free_Ca[3,1,:]*10**6, label='d.r. = 40Hz')
plt.plot(time_dt, free_Ca[4,1,:]*10**6, label='d.r. = 50Hz')
#plt.plot(time_dt[pks_fast], free_Ca[4,1,pks_fast]*10**6, 'r*')

plt.xlabel('Time [s]')
plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')
plt.legend(loc='lower right')
plt.grid()
plt.title('Fast type fibre')
plt.xlim((0, 0.3))

plt.suptitle('MN-driven model sensitivity (excitation, [$Ca^{2+}$]) ', weight='bold',  y=0.94)
plt.show()

#%% COMPARISON WITH RINCON & HOLLINGORTH DATA

figure(figsize=(10, 12))
plt.subplot(3,1,1)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, free_Ca[5,0,:]*10**6, label='Simulated')

plt.plot((data_slow[:,0]-data_slow[0,0])*10**-3, data_slow[:,1], 'k--', label = 'Rincon et al. 2021 (23°C)') # offset and in seconds

plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')           
plt.legend(loc='lower right')
plt.grid()
plt.title('Type I fibre (5 impulses at 100Hz)')
plt.xlim((0, 0.1))

plt.subplot(3,1,2)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, free_Ca[5,1,:]*10**6, label='Simulated')

plt.plot((data_fast[:,0]-data_fast[0,0])*10**-3, data_fast[:,1], 'k--', label = 'Rincon et al. 2021 (23°C)') # offset and in seconds

plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')
plt.legend(loc='lower right')
plt.grid()
plt.title('Type IIB fibre (5 impulses at 100Hz)')
plt.xlim((0, 0.1))

plt.subplot(3,1,3)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, free_Ca[6,1,:]*10**6, label='Simulated')

plt.plot((data_fast_35[:,0]-data_fast_35[0,0])*10**-3, data_fast_35[:,1], 'g--', label = 'Hollingworth 1996 (35°C)') # offset and in seconds

plt.xlabel('Time [s]')
plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')
plt.legend(loc='lower right')
plt.grid()
plt.title('Type IIB fibre (10 impulses at 120Hz)')
plt.xlim((0, 0.13))

plt.suptitle('Simulated vs. literature free [$Ca^{2+}$] for I/IIB mouse fibres', weight='bold',  y=0.94)
plt.show()



