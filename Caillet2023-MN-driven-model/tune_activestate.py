"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Mon Jun 10 10:02:26 2024
___________________________________

Tuning of the active state ODE coefficients based on rat soleus MU twitch.
"""

import sys
sys.path.insert(0,'Modules_BB')
import os
cwd = os.getcwd()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.integrate import solve_ivp
from MU_type_id_MOD import MU_type_id_func
from MU_AP_MOD import MU_AP_func
from MN_AP_MOD import MN_AP_func
from MU_free_Ca_MOD import MU_free_Ca_func
from Ca_Tn import CaTn
from A import active_state

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

def Sag(y):
        
    Ts = 0.043
    if y[4] < 0.1:
        As = 1.76
    else:
         As = 0.96
            
    dS = (As - y[5])/Ts
        
    return dS


#______________________________________________________________________________
" ARRAYS PREALLOCATION "

Ca_Tn = np.empty((len(fd), Nr, len(time_dt)), dtype=object)
free_Ca = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # free [Ca] course
l_M = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # MU lengths
MUAP_nerve = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # MU AP
MN_nerve = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # MN AP
Ca_Tn = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # active state 1
a = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # active state 2
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
            l_M_0 = 1.6 # Baylor, Hollingworth
        else:
            l_M_0 = 1  # Rincon & Giraldo
        
        MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
        Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
        
        def ODE_system(t, y, l_M_0, Matrix_AP, MU_type, dr, i, f):
            
            dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
            dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0], l_M_0, MU_type, Matrix_AP, i, f) # Free Ca (remember to avoid negligible negative values)
            
            dCaTndt = CaTn(t, y, MU_type) # Ca-Tn
            
            dadt = active_state(y, MU_type) # Active state
            
            #dS = Sag(y)
            
            return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dCaTndt, dadt]
    
        y0 = [0, 0, 0, 0, 0, 1*10**-9] # set initial states (active state can't be 0 otherwise you'll divide by 0 in FV)
        p = (l_M_0, Matrix_AP, MU_type, dr, i, f) # set ODE parameters
        sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/2) # solve IVP
        
        #a_1[f,i,:] = sol.y[4]
        MUAP_nerve[f,i,:] = sol.y[0]
        free_Ca[f,i,:] = sol.y[2]  # get [Ca2+]
        
        for l in range (len(time_dt)):  # correct the negative values of free [Ca++]
            if free_Ca[f,i,l] <= 0:
                free_Ca[f,i,l] = 1e-20
         
        Ca_Tn[f,i,:] = sol.y[4]  
        a[f,i,:] = sol.y[5]

    
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

cg1 = get_color_gradient(c1, c2, 3)
cg2 = get_color_gradient(c3, c4, 5)
cg3 = get_color_gradient(c5, c6, 3)
cg4 = get_color_gradient(c7, c8, 5)

#%%
""" Extract experimental digitized data from Rincon 2021 """

# os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\Digitized_Hollingworth_Rincon_Ca") # max activation BB dir path

# path_slow = "Ca_slow_16_67Hz.csv"  #.csv files locations
# path_slow_23 = "Ca_slow_23_100hz.csv"
# path_fast = "Ca_fast_16_67Hz.csv"
# path_fast_35 = "Ca_fast_35_125hz.csv"

# data_slow = pd.read_csv(path_slow, delimiter = ' ', decimal=',')
# data_slow = data_slow.to_numpy()    

# data_slow_23 = pd.read_csv(path_slow_23, delimiter = ' ')
# data_slow_23 = data_slow_23.to_numpy()    
         
# data_fast = pd.read_csv(path_fast, delimiter = ' ', decimal=',')
# data_fast = data_fast.to_numpy()

# data_fast_35 = pd.read_csv(path_fast_35, delimiter = ' ')
# data_fast_35 = data_fast_35.to_numpy()

# os.chdir(cwd)

#%%
""" Extract experimental digitized data from Blinks, Konishi """

# os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\Digitized_Blinks_Konishi") # max activation BB dir path

# konishi = pd.read_csv("Konishi_l_Capeak.csv", delimiter = ' ')
# konishi = (konishi.to_numpy())    

# konishi2 = pd.read_csv("Konishi_l_Catimetopeak.csv", delimiter = ' ')
# konishi2 = (konishi2.to_numpy())  

# blinks = pd.read_csv("Blinks_l_Capeak.csv", delimiter = ' ')
# blinks = (blinks.to_numpy()) 
# blinks = np.delete(blinks, 4, axis=0)
# blinks[:,1] = blinks[:,1]/100

# #l_compr = 

# os.chdir(cwd)

# l_Ca = np.concatenate((blinks[0:4,:], konishi))
# l_Ca = l_Ca[np.argsort(l_Ca[:,0])]
# l_Ca[:,0] = l_Ca[:,0]/2.1

# p = np.polyfit(l_Ca[:, 0], l_Ca[:, 1], 4)
# fit = np.poly1d(p)
# l_sm = np.linspace(l_Ca[:, 0].min(), l_Ca[:, 0].max(), 500)
# #l_sm = np.linspace(0.5, 3, 500)
# Ca_sm = (l_sm**4)*p[0] + (l_sm**3)*p[1] + (l_sm**2)*p[2] + (l_sm)*p[3] + p[4]
# #Ca_sm = fit(l_sm)

# l_Cattp = konishi2
# l_Cattp = l_Cattp[np.argsort(l_Cattp[:,0])]
# l_Cattp[:,0] = l_Cattp[:,0]/2.1

# p2 = np.polyfit(l_Cattp[:, 0], l_Cattp[:, 1], 2)
# fit2 = np.poly1d(p2)
# l_smttp = np.linspace(l_Cattp[:, 0].min(), l_Cattp[:, 0].max(), 500)
# Ca_sm = (l_sm**4)*p[0] + (l_sm**3)*p[1] + (l_sm**2)*p[2] + (l_sm)*p[3] + p[4]
# Ca_smttp = fit2(l_smttp)

# plt.subplot(2,1,1)
# plt.rcParams['figure.dpi'] = 400
# plt.plot(l_Ca[0:5,0], l_Ca[0:5,1], 'rx', label = 'Blinks 1978')
# plt.plot(l_Ca[5:-1,0], l_Ca[5:-1,1], 'gx', label = 'Konishi 1991')
# plt.plot(l_sm, Ca_sm, 'k')
# #plt.xlabel('Normalised sarcomere length', weight='bold')
# plt.ylabel('Norm. p $\Delta$[$Ca^{2+}$]', weight='bold')
# plt.legend(loc = 'upper right')
# plt.grid()

# plt.subplot(2,1,2)
# plt.rcParams['figure.dpi'] = 400
# plt.plot(l_Cattp[:,0], l_Cattp[:,1], 'gx', label = 'Konishi 1991')
# plt.plot(l_smttp, Ca_smttp, 'k')
# plt.xlabel('Normalised sarcomere length', weight='bold')
# plt.ylabel('Norm. ttp $\Delta$[$Ca^{2+}$]', weight='bold')
# plt.legend(loc = 'upper right')
# plt.grid()

#%%
fs = [25, 30, 35, 40]
#tt = [0.01, 0.11, 0.16, 0.17]
tt = [1, 1.4, 1.5, 1.6]

p = np.polyfit(tt, fs, 1)
fit = np.poly1d(p)
ttt = np.linspace(1, 1.6, 200)
fss = (ttt)*p[0] + p[1]
#fss = (ttt**3)*p[0] + (ttt**2)*p[1] + (ttt)*p[2] + p[3]

plt.plot(tt, fs, 'rx')
plt.plot(ttt, fss, 'k')

#%%
""" Extract experimental digitized data from Matsuo 2010 """

# os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\Digitized_Blinks_Konishi") # max activation BB dir path

# matsuo = pd.read_csv("Matsuo_CaTn.csv", delimiter = ' ')
# matsuo = (matsuo.to_numpy())   
# #matsuo = [[0,0], [matsuo]]

# os.chdir(cwd)

# plt.rcParams['figure.dpi'] = 360
# plt.plot(matsuo[:,0]*1e-3, matsuo[:,1], 'k--', label='Matsuo 2010')
# plt.plot(time_dt, Ca_Tn[0,1,:]*10**6, 'r', label = 'Simulated')
# plt.ylabel('CaTn [$\mu$M]', weight='bold')
# plt.xlabel('Time [s]', weight='bold')
# plt.legend(loc='lower right')
# plt.xlim([0,matsuo[-1,0]*1e-3])
# plt.grid()

#%% Visual validation
#ACTIVATION General
#Single twitch & 20Hz
plt.rcParams['figure.dpi'] = 400


#plt.subplot(1,2,1)
plt.plot(time_dt, a[0,0,:], color=cg1[0], label='d.r. = 10Hz')
plt.plot(time_dt, a[1,0,:], color=cg1[1], label='d.r. = 20Hz')
plt.plot(time_dt, a[2,0,:], color=cg1[2], label='d.r. = 30Hz')
plt.ylabel('Active state', weight='bold', fontsize=17)
plt.xlabel('Time [s]', weight='bold', fontsize=17)
plt.title('Slow MU, at optimal $\overline {L^M}$', weight='bold', fontsize=15)
plt.ylim((-0.05,1.05))
plt.grid()
plt.legend(loc='lower right')

# plt.subplot(1,2,2)
# plt.plot(time_dt, a[0,1,:], color=cg2[0], label='d.r. = 10Hz')
# plt.plot(time_dt, a[1,1,:], color=cg2[1], label='d.r. = 20Hz')
# plt.plot(time_dt, a[2,1,:], color=cg2[2], label='d.r. = 30Hz')
# plt.xlabel('Time [s]', weight='bold', fontsize=17)
# plt.title('Fast MU, at optimal $\overline {L^M}$', weight='bold', fontsize=15)
# plt.ylim((-0.05,1.05))
# plt.grid()
# plt.legend(loc='lower right')

#%%

# os.chdir('C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\benchmark_input\\sub\\c_freq')

# iso_slow_10 = np.genfromtxt('force_isometric_c10.dat', delimiter='')
# iso_slow_20 = np.genfromtxt('force_isometric_c20.dat', delimiter='')
# iso_slow_30 = np.genfromtxt('force_isometric_c30.dat', delimiter='')

# plt.rcParams['figure.dpi'] = 400
# figure(figsize=(12, 5))

# plt.subplot(1,2,1)
# plt.plot(iso_slow_10[:,0], iso_slow_10[:,1], color=cg1[0], label='d.r. = 10Hz')
# plt.plot(iso_slow_20[:,0], iso_slow_20[:,1], color=cg1[1], label='d.r. = 20Hz')
# plt.plot(iso_slow_30[:,0], iso_slow_30[:,1], color=cg1[2], label='d.r. = 30Hz')
# plt.xlabel('Time [s]', weight='bold', fontsize=17)
# plt.ylabel('Force [N]', weight='bold', fontsize=17)
# plt.title('Isometric trials - cat SOL', weight='bold', fontsize=15)
# plt.grid()
# plt.legend(loc='lower right')

# os.chdir('C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\benchmark_input\\fast')

# iso_fast_25 = np.load('iso_25_interp.npy')
# iso_fast_30 = np.load('iso_30_interp.npy')
# iso_fast_35 = np.load('iso_35_interp.npy')
# iso_fast_40 = np.load('iso_40_interp.npy')
# iso_fast_150 = np.load('iso_150_interp.npy')
# time = np.arange(0, 0.7, dt) # time 

# plt.subplot(1,2,2)
# plt.plot(time, iso_fast_25, color=cg2[0], label='d.r. = 25Hz')
# plt.plot(time, iso_fast_30, color=cg2[1], label='d.r. = 30Hz')
# plt.plot(time, iso_fast_35, color=cg2[2], label='d.r. = 35Hz')
# plt.plot(time, iso_fast_40, color=cg2[3], label='d.r. = 40Hz')
# plt.plot(time, iso_fast_150, color=cg2[4], label='d.r. = 150Hz')

# plt.xlabel('Time [s]', weight='bold', fontsize=17)
# plt.title('Isometric trials - rat FR MU', weight='bold', fontsize=15)
# plt.grid()
# plt.legend(loc='lower right')

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


#%% COMPARISON WITH RINCON & HOLLINGORTH DATA

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
# fig.supylabel('$\Delta$[$Ca^{2+}$] [$\mu$M]', x=0.06, weight='bold', fontsize=17)
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