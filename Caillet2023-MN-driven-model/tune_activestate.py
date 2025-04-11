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
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.integrate import solve_ivp
from MU_type_id_MOD import MU_type_id_func
from MU_AP_MOD import MU_AP_func
from MN_AP_MOD import MN_AP_func
from MU_free_Ca_MOD import MU_free_Ca_func
from A_1 import active_state_1
from A_2 import active_state_2

#%%____________________________________________________________________________
""" Load time, displacement and force form BB tests, create virtual MU spikes"""

dt = 0.0001 # time step (x-data)
t_end = 2 # total seconds
t_end_a = 1.3 # total stimulation time

#t_end = 0.04 # case of 5 shocks at 100HZ

#______________________________________________________________________________

time_dt = np.arange(0, t_end, dt) # time 
fd = [10, 20, 30, 40, 64.5, 102, 125] # paper stimulation freq. Hz d.r. (to reach active_state = 1 always, without any recruitment)
T = [1/fd[0], 1/fd[1], 1/fd[2], 1/fd[3], 1/fd[4], 1/fd[5], 1/fd[6]]# correspondent d.r. period
#Nr = len(T)
Nr = 1
#muscle_F0M = 1.17 # muscle max. isom. force [N]

"""1st CASE: d.r. at n*10Hz constant for Nr different MU types"""
#sp_matrix_twitch = np.empty((Nr, 1), dtype=float)
sp_matrix10 = np.empty((Nr,int(t_end_a/T[0])), dtype=float)
sp_matrix20 = np.empty((Nr,int(t_end_a/T[1])), dtype=float)
sp_matrix30 = np.empty((Nr,int(t_end_a/T[2])), dtype=float)
sp_matrix40 = np.empty((Nr,int(t_end_a/T[3])), dtype=float)
sp_matrix67 = np.empty((Nr,int(0.08/T[4])), dtype=float)
sp_matrix100 = np.empty((Nr,int(0.05/T[5])), dtype=float) # manually impose the n. of impulses for literature comparisons
sp_matrix125 = np.empty((Nr,int(0.08/T[6])), dtype=float)

# disch_twitch = np.arange(0,T[0],T[0]) # create array of dischare times at 10 Hz
# for i in range (Nr):
#     sp_matrix_twitch[i,:] = disch_twitch

disch10 = np.arange(0,t_end_a,T[0]) # create array of dischare times at 10 Hz
for i in range (Nr):
    sp_matrix10[i,:] = disch10

disch20 = np.arange(0,t_end_a,T[1]) # create array of discharge times at 20 Hz
for i in range (Nr):
    sp_matrix20[i,:] = disch20

disch30 = np.arange(0,t_end_a,T[2]) # create array of discharge times at 30 Hz
for i in range (Nr):
    sp_matrix30[i,:] = disch30

disch40 = np.arange(0,t_end_a,T[3]) # create array of discharge times at 40 Hz
for i in range (Nr):
    sp_matrix40[i,:] = disch40

disch67 = np.arange(0,0.07,T[4]) # create array of discharge times at 50 Hz
for i in range (Nr):
    sp_matrix67[i,:] = disch67
    
disch100 = np.arange(0,0.04,T[5]) # create array of discharge times at 100 Hz
for i in range (Nr):
    sp_matrix100[i,:] = disch100
        
disch125 = np.arange(0,0.08,T[6]) # create array of discharge times at 125 Hz
for i in range (Nr):
    sp_matrix125[i,:] = disch125

"""2nd CASE: one spike only"""
# sp_matrix = np.zeros((Nr, int(t_end/T)), dtype=float)
# for i in range(Nr): # append Nr times
#     sp_matrix[i, 0] = 1

#______________________________________________________________________________      
print('There are ', Nr, ' discharging MUs in this simulation.')

# F0MU distribution across the sample of MUs
#F0MU_distribution = F0MU_distrib_func(Nr, muscle_F0M)
#______________________________________________________________________________
" LENGTH & DISPLACEMENT PARAMETERS "

l_T_slack = 65 # Tendon slack length (mm)
l_M_opt = 30 # Optimal fiber length (mm)

alpha_0 = 7.5*np.pi/180 # initial pennation (pennation should make less than 2 % difference)
l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0) - 4 # Musculo-tendon length (mm)

#ISOMETRIC CASE
# l_MT = np.zeros((len(time_dt)+1), dtype=object) 
# l_MT = l_MT + l_MT_0

#______________________________________________________________________________
" ARRAYS PREALLOCATION "

Ca_Tn = np.empty((len(fd), Nr, len(time_dt)), dtype=object)
free_Ca = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # free [Ca] course
l_M = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # MU lengths
MUAP_nerve = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # MU AP
MN_nerve = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # MN AP
a_1 = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # active state 1
a_2 = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # active state 2
#______________________________________________________________________________
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

# ...for each frequency considered assign the corresponding d.r. array
for f in range(3):
    
    # if f == 0:
    #     sp_matrix = sp_matrix_twitch
    if f == 0:
        sp_matrix = sp_matrix10
    elif f == 1:
        sp_matrix = sp_matrix20
    elif f == 2:
        sp_matrix = sp_matrix30
    elif f == 3:
        sp_matrix = sp_matrix40
    elif f == 4:
        #sp_matrix = sp_matrix50
        sp_matrix = sp_matrix67
    elif f == 5:
        sp_matrix = sp_matrix100
    elif f == 6:
        sp_matrix = sp_matrix125
        
    dr = fd[f]  # MU discharge rate
        
    # ...for each considered i-th MU 
    for i in range(Nr):  
        print('Computing Force for MU n°', str(i+1))
       
        if f == 4 or f == 6:
            l_M_0 = 1.8 # Baylor or Hollingworth
        else:
            l_M_0 = 1  # Rincon & Giraldo
        
        MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
        Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
        
        def ODE_system(t, y, l_M_0, Matrix_AP, MU_type, dr, i, f):
            
            dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
            dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0], l_M_0, MU_type, Matrix_AP, i, f) # Free Ca (remember to avoid negligible negative values)
            
            dadt1 = active_state_1(t, y) # Ca-Tn
            
            dadt2 = active_state_2(t, y) # Active state
            
            return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dadt1, dadt2]
    
        y0 = [0, 0, 0, 0, 1*10**-9, 1*10**-9] # set initial states (active state can't be 0 otherwise you'll divide by 0 in FV)
        p = (l_M_0, Matrix_AP, MU_type, dr, i, f) # set ODE parameters
        sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/2) # solve IVP
    
        a_2[f,i,:] = sol.y[5]  # get active state
        a_1[f,i,:] = sol.y[4]
        MUAP_nerve[f,i,:] = sol.y[0]
        free_Ca[f,i,:] = sol.y[2]  # get [Ca2+]
        
        for l in range (len(time_dt)):  # correct the negative values of free [Ca++]
            MN_nerve[f,i,l] = MN_AP_func(time_dt[l], Matrix_AP)
            if free_Ca[f,i,l] < 0:
                free_Ca[f,i,l] = 0
    
#%%------------------------------------------------------------------------------

"""Create color map and plot relationships"""
   
c1 = "#0b165e"  #dark blue
c2 = "#00eeff"  #light blue

c3 = '#e31010'  #red
c4 = '#fcff01'  #orange

c5 = '#003800'  #dark green
c6 = '#00fb00'

c7 = '#fb03f3'  #fucsia
c8 = '#fba0e8'

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

cg1 = get_color_gradient(c1, c2, 4)
cg2 = get_color_gradient(c3, c4, 3)
cg3 = get_color_gradient(c5, c6, 3)
cg4 = get_color_gradient(c7, c8, 5)

#%%
""" Extract experimental digitized data from Rincon 2021 """

os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Digitized_Hollingworth_Rincon_Ca") # max activation BB dir path

path_slow = "slow_16_mouse_67Hz.csv"  #.csv files locations
path_slow_23 = "slow_23_mouse_100hz_spaceseparator.csv"
path_fast = "fast_16_mouse_67Hz.csv"
path_fast_35 = "fast_35_mouse_125hz.csv"

data_slow = pd.read_csv(path_slow, delimiter = ' ', decimal=',')
data_slow = data_slow.to_numpy()    

data_slow_23 = pd.read_csv(path_slow_23, delimiter = ' ')
data_slow_23 = data_slow_23.to_numpy()    
         
data_fast = pd.read_csv(path_fast, delimiter = ' ', decimal=',')
data_fast = data_fast.to_numpy()

data_fast_35 = pd.read_csv(path_fast_35, delimiter = ' ')
data_fast_35 = data_fast_35.to_numpy()

os.chdir(cwd)

#%% Visual validation
#ACTIVATION General
#Single twitch & 20Hz
figure(figsize=(17, 10))
plt.subplot(1,2,1)
plt.rcParams['figure.dpi'] = 360

plt.plot(time_dt, a_1[0,0,:], color=cg1[0], label='d.r. = 10Hz')
plt.plot(time_dt, a_1[1,0,:], color=cg1[1], label='d.r. = 20Hz')
plt.plot(time_dt, a_1[2,0,:], color=cg1[2], label='d.r. = 30Hz')
# # plt.plot(time_dt, active_state[3,0,:], color=cg1[3], label='d.r. = 40Hz')
#plt.plot(time_dt, active_state[4,0,:], color=cg1[3], label='d.r. = 67Hz')
plt.ylabel('Active state (intermediate)', weight='bold', fontsize=17)
plt.xlabel('Time [s]', weight='bold', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
#plt.title('Slow MU - A(t)', weight='bold', fontsize=15)
plt.grid()

plt.subplot(1,2,2)
plt.rcParams['figure.dpi'] = 360

plt.plot(time_dt, a_2[0,0,:], color=cg1[0], label='d.r. = 10Hz')
plt.plot(time_dt, a_2[1,0,:], color=cg1[1], label='d.r. = 20Hz')
plt.plot(time_dt, a_2[2,0,:], color=cg1[2], label='d.r. = 30Hz')
# # plt.plot(time_dt, active_state[3,0,:], color=cg1[3], label='d.r. = 40Hz')
#plt.plot(time_dt, active_state[4,0,:], color=cg1[3], label='d.r. = 67Hz')
plt.ylabel('Active state', weight='bold', fontsize=17)
plt.xlabel('Time [s]', weight='bold', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
#plt.title('Slow MU - A(t)', weight='bold', fontsize=15)
plt.grid()


#%%
# plt.rcParams['figure.dpi'] = 360
# fig, ax1 = plt.subplots()

# delay = 0.5*10**-3 + 3.0*10**-3 + 0.5*10**-3

# #plt.subplot(2,1,1)
# ax1.set_xlabel('Time [s]', weight ='bold', fontsize=14)
# ax1.set_ylabel('Nerve depolarization [mV]', color='k', weight = 'bold', fontsize=14)
# ax1.plot(time_dt, MN_nerve[2,0,:], color='k')
# ax1.tick_params(axis='y', labelcolor='k')
# plt.xlim([-0.008,0.08])
# plt.grid()
# ax2 = ax1.twinx()
# ax2.set_ylabel('Fibre depolarization [mV]', color='r', weight = 'bold', fontsize=14)
# ax2.plot(time_dt+delay, MUAP_nerve[2,0,:], color='r')
# ax2.tick_params(axis='y', labelcolor='r')
# plt.xlim([-0.008,0.08])

# plt.show()

#%%
# plt.rcParams['figure.dpi'] = 360
# figure(figsize=(6, 5))
# #plt.subplot(2,1,1)
# plt.plot(time_dt, Ca_Tn[0,0,:]*10**6, 'm', label='slow fibre')
# plt.plot(time_dt, Ca_Tn[0,1,:]*10**6, 'g', label='fast fibre')
# plt.xlim([-0.02, 0.5])
# plt.legend(loc='lower right', fontsize=17)
# plt.ylabel('Ca-Tn concentration [$\mu$M]', weight='bold', fontsize=17)
# plt.xlabel('Time [s]', weight='bold', fontsize=17)
# #plt.title('Slow MU - A(t)', weight='bold', fontsize=15)
# plt.grid()

#%%

# cwd = os.getcwd()
# os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation")
# #os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\fixedfreq_yielding")
# force_sample = np.empty((3, len(time_dt)), dtype=object)

# force_sample1 = (np.genfromtxt('force_forcedot_trial1.dat', delimiter=''))
# force_sample2 = (np.genfromtxt('force_forcedot_trial2.dat', delimiter=''))
# force_sample3 = (np.genfromtxt('force_forcedot_trial3.dat', delimiter=''))

# os.chdir(cwd)

# plt.rcParams['figure.dpi'] = 360
# figure(figsize=(10, 8))
# plt.plot(force_sample1[:,0], force_sample1[:,1], color=cg1[0], label='d.r. = 10Hz')
# plt.plot(force_sample2[:,0], force_sample2[:,1], color=cg1[1], label='d.r. = 20Hz')
# plt.plot(force_sample3[:,0], force_sample3[:,1], color=cg1[2], label='d.r. = 30Hz')

# plt.ylabel('Force [N]', weight='bold', fontsize=17)
# plt.xlabel('Time [s]', weight='bold', fontsize=17)
# plt.legend(loc='lower right', fontsize=17)
# #plt.title('Slow MU - A(t)', weight='bold', fontsize=15)
# plt.grid()


#%% Visual validation
# # ACTIVATION Thelen
# #Single twitch
# plt.rcParams['figure.dpi'] = 360
# figure(figsize=(12, 9))
# plt.subplot(2,2,1)
# plt.plot(time_dt[0:5000], active_state[0,0,0:5000], color=cg1[0], label='t_up = 0.01')
# plt.plot(time_dt[0:5000], active_state[0,1,0:5000], color=cg1[1], label='t_up = 0.04')
# plt.plot(time_dt[0:5000], active_state[0,2,0:5000], color=cg1[2], label='t_up = 0.8')
# plt.plot(time_dt[0:5000], active_state[0,3,0:5000], color=cg1[3], label='t_up = 1.2')
# plt.plot(time_dt[0:5000], active_state[0,4,0:5000], color=cg1[4], label='t_up = 1.6')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,2)
# plt.plot(time_dt[0:5000], active_state[0,5,0:5000], color=cg2[0], label='t_down = 0.01')
# plt.plot(time_dt[0:5000], active_state[0,6,0:5000], color=cg2[1], label='t_down = 0.04')
# plt.plot(time_dt[0:5000], active_state[0,7,0:5000], color=cg2[2], label='t_down = 0.8')
# plt.plot(time_dt[0:5000], active_state[0,8,0:5000], color=cg2[3], label='t_down = 1.2')
# plt.plot(time_dt[0:5000], active_state[0,9,0:5000], color=cg2[4], label='t_down = 1.6')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,3) 
# plt.plot(time_dt[0:5000], active_state[0,10,0:5000], color=cg3[0], label='a = 0.2')
# plt.plot(time_dt[0:5000], active_state[0,11,0:5000], color=cg3[1], label='a = 0.5')
# plt.plot(time_dt[0:5000], active_state[0,12,0:5000], color=cg3[2], label='a = 0.8')
# plt.plot(time_dt[0:5000], active_state[0,13,0:5000], color=cg3[3], label='a = 1.2')
# plt.plot(time_dt[0:5000], active_state[0,14,0:5000], color=cg3[4], label='a = 1.5')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,4)
# plt.plot(time_dt[0:5000], active_state[0,15,0:5000], color=cg4[0], label='b = 0.2')
# plt.plot(time_dt[0:5000], active_state[0,16,0:5000], color=cg4[1], label='b = 0.5')
# plt.plot(time_dt[0:5000], active_state[0,17,0:5000], color=cg4[2], label='b = 0.8')
# plt.plot(time_dt[0:5000], active_state[0,18,0:5000], color=cg4[3], label='b = 1.2')
# plt.plot(time_dt[0:5000], active_state[0,19,0:5000], color=cg4[4], label='b = 1.5')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.suptitle('A(t) sensitivity - Single twitch', weight='bold', y = 0.92)

#%% # 20 Hz Thelen
# plt.rcParams['figure.dpi'] = 360
# figure(figsize=(12, 9))
# plt.subplot(2,2,1)
# plt.plot(time_dt, active_state[2,0,:], color=cg1[0], label='t_up = 0.01')
# plt.plot(time_dt, active_state[2,1,:], color=cg1[1], label='t_up = 0.04')
# plt.plot(time_dt, active_state[2,2,:], color=cg1[2], label='t_up = 0.8')
# plt.plot(time_dt, active_state[2,3,:], color=cg1[3], label='t_up = 1.2')
# plt.plot(time_dt, active_state[2,4,:], color=cg1[4], label='t_up = 1.6')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,2)
# plt.plot(time_dt, active_state[2,5,:], color=cg2[0], label='t_down = 0.01')
# plt.plot(time_dt, active_state[2,6,:], color=cg2[1], label='t_down = 0.04')
# plt.plot(time_dt, active_state[2,7,:], color=cg2[2], label='t_down = 0.8')
# plt.plot(time_dt, active_state[2,8,:], color=cg2[3], label='t_down = 1.2')
# plt.plot(time_dt, active_state[2,9,:], color=cg2[4], label='t_down = 1.6')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,3) 
# plt.plot(time_dt, active_state[2,10,:], color=cg3[0], label='a = 0.2')
# plt.plot(time_dt, active_state[2,11,:], color=cg3[1], label='a = 0.5')
# plt.plot(time_dt, active_state[2,12,:], color=cg3[2], label='a = 0.8')
# plt.plot(time_dt, active_state[2,13,:], color=cg3[3], label='a = 1.2')
# plt.plot(time_dt, active_state[2,14,:], color=cg3[4], label='a = 1.5')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,4)
# plt.plot(time_dt, active_state[2,15,:], color=cg4[0], label='b = 0.2')
# plt.plot(time_dt, active_state[2,16,:], color=cg4[1], label='b = 0.5')
# plt.plot(time_dt, active_state[2,17,:], color=cg4[2], label='b = 0.8')
# plt.plot(time_dt, active_state[2,18,:], color=cg4[3], label='b = 1.2')
# plt.plot(time_dt, active_state[2,19,:], color=cg4[4], label='b = 1.5')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.suptitle('A(t) sensitivity - 20 Hz d.r.', weight='bold', y = 0.92)
    
#%% pars = [12, 17, 20, 22.5, 25, 5, 10, 15, 19.2, 22.5]
# ACTIVATION Hussein
#Single twitch
# plt.rcParams['figure.dpi'] = 360
# figure(figsize=(12, 9))
# plt.subplot(2,2,1)
# plt.plot(time_dt[0:5000], active_state[0,0,0:5000], color=cg3[0], label='k1 = 12')
# plt.plot(time_dt[0:5000], active_state[0,1,0:5000], color=cg3[1], label='k1 = 17')
# plt.plot(time_dt[0:5000], active_state[0,2,0:5000], color=cg3[2], label='k1 = 20')
# plt.plot(time_dt[0:5000], active_state[0,3,0:5000], color=cg3[3], label='k1 = 22.5')
# plt.plot(time_dt[0:5000], active_state[0,4,0:5000], color=cg3[4], label='k1 = 25')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,2)
# plt.plot(time_dt[0:5000], active_state[0,5,0:5000], color=cg4[0], label='k2 = 5')
# plt.plot(time_dt[0:5000], active_state[0,6,0:5000], color=cg4[1], label='k2 = 10')
# plt.plot(time_dt[0:5000], active_state[0,7,0:5000], color=cg4[2], label='k2 = 15')
# plt.plot(time_dt[0:5000], active_state[0,8,0:5000], color=cg4[3], label='k2 = 19.2')
# plt.plot(time_dt[0:5000], active_state[0,9,0:5000], color=cg4[4], label='k2 = 22.5')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.suptitle('A(t) sensitivity - Single twitch & 20 Hz', weight='bold', y = 0.92)

# plt.subplot(2,2,3) 
# plt.plot(time_dt, active_state[2,0,:], color=cg3[0], label='k1 = 12')
# plt.plot(time_dt, active_state[2,1,:], color=cg3[1], label='k1 = 17')
# plt.plot(time_dt, active_state[2,2,:], color=cg3[2], label='k1 = 20')
# plt.plot(time_dt, active_state[2,3,:], color=cg3[3], label='k1 = 22.5')
# plt.plot(time_dt, active_state[2,4,:], color=cg3[4], label='k1 = 25')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

# plt.subplot(2,2,4)
# plt.plot(time_dt, active_state[2,5,:], color=cg4[0], label='k2 = 5')
# plt.plot(time_dt, active_state[2,6,:], color=cg4[1], label='k2 = 10')
# plt.plot(time_dt, active_state[2,7,:], color=cg4[2], label='k2 = 15')
# plt.plot(time_dt, active_state[2,8,:], color=cg4[3], label='k2 = 19.2')
# plt.plot(time_dt, active_state[2,9,:], color=cg4[4], label='k2 = 22.5')
# plt.ylabel('Active state')
# plt.xlabel('Time [s]')
# plt.legend(loc='lower right')
# plt.grid()

#%% EXCITATION (2)

# figure(figsize=(12, 10))
# plt.subplot(2,1,1)
# plt.rcParams['figure.dpi'] = 360
# #plt.plot(time_dt, Ca_Tn[0,0,:]*10**6, label='d.r. = 10Hz')
# # plt.plot(time_dt, free_Ca[1,0,:]*10**6, label='d.r. = 20Hz')
# # plt.plot(time_dt, free_Ca[2,0,:]*10**6, label='d.r. = 30Hz')
# # plt.plot(time_dt, free_Ca[3,0,:]*10**6, label='d.r. = 40Hz')
# plt.plot(time_dt, Ca_Tn[4,0,:]*10**6, label='d.r. = 50Hz')
# #plt.plot(time_dt[pks_slow], free_Ca[4,0,pks_slow]*10**6, 'ro')

# #plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')           
# plt.legend(loc='lower right')
# plt.grid()
# plt.title('Slow type fibre')
# plt.xlim((0, 0.3))

# plt.subplot(2,1,2)
# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_dt, Ca_Tn[0,1,:]*10**6, label='d.r. = 10Hz')
# # plt.plot(time_dt, free_Ca[1,1,:]*10**6, label='d.r. = 20Hz')
# # plt.plot(time_dt, free_Ca[2,1,:]*10**6, label='d.r. = 30Hz')
# # plt.plot(time_dt, free_Ca[3,1,:]*10**6, label='d.r. = 40Hz')
# plt.plot(time_dt, Ca_Tn[4,1,:]*10**6, label='d.r. = 50Hz')
# #plt.plot(time_dt[pks_fast], free_Ca[4,1,pks_fast]*10**6, 'ro')

# plt.xlabel('Time [s]')
# #plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')
# plt.legend(loc='lower right')
# plt.grid()
# plt.title('Fast type fibre')
# plt.xlim((0, 0.3))

# plt.suptitle('MN-driven model sensitivity (excitation, [Ca_Tn]) ', weight='bold',  y=0.94)
# plt.show()


#%%
# EXCITATION (3)
# pks_slow, _ = find_peaks(-free_Ca[4,0,:]) # 3.64*10**-6
# pks_fast, _ = find_peaks(-free_Ca[4,1,:]) # 5.34*10**-7

# figure(figsize=(12, 10))
# plt.subplot(2,1,1)
# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_dt, free_Ca[0,0,:]*10**6, label='d.r. = 10Hz')
# #plt.plot(time_dt, free_Ca[1,0,:]*10**6, label='d.r. = 20Hz')
# #plt.plot(time_dt, free_Ca[2,0,:]*10**6, label='d.r. = 30Hz')
# #plt.plot(time_dt, free_Ca[3,0,:]*10**6, label='d.r. = 40Hz')
# plt.plot(time_dt, free_Ca[4,0,:]*10**6, label='d.r. = 50Hz')
# #plt.plot(time_dt[pks_slow], free_Ca[4,0, pks_slow]*10**6, 'ro')

# plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')           
# plt.legend(loc='lower right')
# plt.grid()
# plt.title('Slow type fibre')
# plt.xlim((0, 0.3))

# plt.subplot(2,1,2)
# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_dt, free_Ca[0,1,:]*10**6, label='d.r. = 10Hz')
# #plt.plot(time_dt, free_Ca[1,1,:]*10**6, label='d.r. = 20Hz')
# #plt.plot(time_dt, free_Ca[2,1,:]*10**6, label='d.r. = 30Hz')
# #plt.plot(time_dt, free_Ca[3,1,:]*10**6, label='d.r. = 40Hz')
# plt.plot(time_dt, free_Ca[4,1,:]*10**6, label='d.r. = 50Hz')
# #plt.plot(time_dt[pks_fast], free_Ca[4,1,pks_fast]*10**6, 'ro')

# plt.xlabel('Time [s]')
# plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')
# plt.legend(loc='lower right')
# plt.grid()
# plt.title('Fast type fibre')
# plt.xlim((0, 0.3))

# plt.suptitle('MN-driven model sensitivity (excitation, [$Ca^{2+}$]) ', weight='bold',  y=0.94)
# plt.show()

#%% COMPARISON WITH RINCON & HOLLINGORTH DATA

# _, ind_slow = np.unique(data_slow[:,0], return_index=True)
# time_slow = (np.linspace(data_slow[0,0],data_slow[-1,0],len(time_dt)))
# data_slow = data_slow[ind_slow,:]

# _, ind_slow_23 = np.unique(data_slow_23[:,0], return_index=True)
# time_slow_23 = (np.linspace(data_slow_23[0,0],data_slow_23[-1,0],len(time_dt)))
# data_slow_23 = data_slow_23[ind_slow_23,:]

# _, ind_fast = np.unique(data_fast[:,0], return_index=True)
# time_fast = (np.linspace(data_fast[0,0],data_fast[-1,0],len(time_dt)))
# data_fast = data_fast[ind_fast,:]

# _, ind_fast_35 = np.unique(data_fast_35[:,0], return_index=True)
# time_fast_35 = (np.linspace(data_fast_35[0,0],data_fast_35[-1,0],len(time_dt)))
# data_fast_35 = data_fast_35[ind_fast_35,:]

# data_slow = sp.interpolate.interp1d(data_slow[:,0], data_slow[:,1], kind='cubic')(time_slow) # interpolate with time_dt points (l_MT must be longer)
# data_slow_23 = sp.interpolate.interp1d(data_slow_23[:,0], data_slow_23[:,1], kind='cubic')(time_slow_23) # interpolate with time_dt points (l_MT must be longer)
# data_fast = sp.interpolate.interp1d(data_fast[:,0], data_fast[:,1], kind='cubic')(time_fast) # interpolate with time_dt points (l_MT must be longer)
# data_fast_35 = sp.interpolate.interp1d(data_fast_35[:,0], data_fast_35[:,1], kind='cubic')(time_fast_35)

# fig, axs = plt.subplots(2, 1, figsize=(13, 8))

# plt.subplot(2,1,1)
# plt.rcParams['figure.dpi'] = 400
# #plt.plot(time_dt, free_Ca[6,0,:]*10**6, 'b', label='Simulated (35°C)')
# #plt.plot(time_dt, free_Ca[4,0,:]*10**6, 'r', label='Simulated (16°C)')
# plt.plot(time_dt, free_Ca[5,0,:]*10**6, 'g', label='Simulated (23°C)')

# #plt.plot((data_slow[:,0]-data_slow[0,0])*10**-3, data_slow[:,1], 'k--', label = 'Baylor et al. 2003 (16°C)') # offset and in seconds
# plt.plot((data_slow_23[:,0]-data_slow_23[0,0])*10**-3, data_slow_23[:,1], 'k--', label = 'Rincon et al. 2021 (23°C)')
# plt.gca().tick_params(axis='x', which='both', labelbottom=False)
# #plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')           
# plt.legend(loc='upper right', fontsize=16)
# plt.grid()
# #plt.title('Slow MU', weight='bold', fontsize=16)
# plt.xlim((0, 0.12))

# # plt.subplot(3,1,2)
# # plt.rcParams['figure.dpi'] = 360
# # plt.plot(time_dt, free_Ca[4,1,:]*10**6, 'r', label='Simulated (16°C)')

# # plt.plot((data_fast[:,0]-data_fast[0,0])*10**-3, data_fast[:,1], 'k--', label = 'Baylor et al. 2003 (16°C)') # offset and in seconds
# # plt.gca().tick_params(axis='x', which='both', labelbottom=False)
# # #plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]')
# # plt.legend(loc='lower right', fontsize=14)
# # plt.grid()
# # plt.title('Fast fibre', weight='bold', fontsize=16)
# # plt.xlim((0, 0.1))

# plt.subplot(2,1,2)
# plt.rcParams['figure.dpi'] = 400
# plt.plot(time_dt, free_Ca[6,1,:]*10**6, 'g', label='Simulated (35°C)')

# plt.plot((data_fast_35[:,0]-data_fast_35[0,0])*10**-3, data_fast_35[:,1], 'k--', label = 'Hollingworth 1996 (35°C)') # offset and in seconds

# plt.xlabel('Time [s]', weight='bold', fontsize=16)
# #fig.supylabel('Free [$Ca^{2+}$] [$\mu$M]', x=0.06, weight='bold', fontsize=15)
# fig.supylabel('[$Ca^{2+}$] [$\mu$M]', x=0.06, weight='bold', fontsize=17)
# plt.legend(loc='upper right', fontsize=16)
# plt.grid()
# #plt.title('Fast MU', weight='bold', fontsize=16)
# plt.xlim((0, 0.12))

# #plt.suptitle('Simulated vs. literature free [$Ca^{2+}$] for I/IIB mouse fibres', weight='bold',  y=0.94, fontsize=15)
# plt.show()

#%% Calculate fitting error interpolating exp data (you have to change equal time values due to manual digitization)
# ind_slow = np.isin(time_dt, np.round(np.unique(data_slow[:,0]*10**-3),4))
# ind_slow_2 = np.isin(np.round(np.unique(data_slow[:,0]*10**-3),4), time_dt)
# mean_abs_error_slow = np.mean((np.abs(free_Ca[4,0,ind_slow]*10**6 - data_slow[ind_slow_2,1])/free_Ca[4,0,ind_slow]*10**6)*100)


# ind_slow_23 = time_dt == data_slow_23[:,0]
# mean_abs_error_slow_23 = np.mean((np.abs(free_Ca[5,0,ind_slow_23]*10**6 - data_slow_23[:,1])/free_Ca[5,0,ind_slow_23]*10**6)*100)
# ind_fast = time_dt == data_fast[:,0]
# mean_abs_error_fast = np.mean((np.abs(free_Ca[4,1,ind_fast]*10**6 - data_fast[:,1])/free_Ca[4,1,ind_fast]*10**6)*100)
# ind_fast_35 = time_dt == data_fast_35[:,0]
# mean_abs_error_fast_35 = np.mean((np.abs(free_Ca[6,1,:ind_fast_35]*10**6 - data_fast_35[:,1])/free_Ca[6,1,ind_fast_35]*10**6)*100)