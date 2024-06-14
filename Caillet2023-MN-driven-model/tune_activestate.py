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
#______________________________________________________________________________
""" Load time, displacement and force form BB tests, create virtual MU spikes"""

dt = 0.0001 # time step (x-data)
t_end = 1 # total seconds
time_dt = np.arange(0, t_end, dt) # time 
fd = [10, 20, 30, 40, 50] # paper stimulation freq. Hz d.r. (to reach active_state = 1 always, without any recruitment)
T = [1/fd[0], 1/fd[1], 1/fd[2], 1/fd[3], 1/fd[4]]# correspondend d.r. period
Nr = 2
muscle_F0M = 1.17 # muscle max. isom. force [N]

"""1st CASE: d.r. at n*10Hz constant for 2 different MU types"""
sp_matrix10 = np.empty((Nr,int(t_end/T[0])), dtype=float)
sp_matrix20 = np.empty((Nr,int(t_end/T[1])), dtype=float)
sp_matrix30 = np.empty((Nr,int(t_end/T[2])), dtype=float)
sp_matrix40 = np.empty((Nr,int(t_end/T[3])), dtype=float)
sp_matrix50 = np.empty((Nr,int(t_end/T[4])), dtype=float)

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
l_M_0 = 1.16 # Initial fibre length normalized to l_M_0 (it would be l_MT - l_ST)
alpha_0 = 11.2*np.pi/180 # initial pennation (pennation should make less than 2 % difference)

alpha = np.empty((len(time_dt)+1), dtype=object) # Pennation

active_state = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # active state
free_Ca = np.empty((len(fd), Nr, len(time_dt)), dtype=object) # free [Ca] course

# ...for each frequency considered assign the correspond
for f in range (5):
    
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
        p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha) # set ODE parameters
        sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/2) # solve IVP
    
        active_state[f,i,:] = sol.y[5]  # get active state
        #l_M[i,:] = sol.y[6]  # get l_M
        # MUAP_nerve[i] = sol.y[0]
        free_Ca[f,i,:] = sol.y[2]
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


#______________________________________________________________________________

#%% Visual validation
plt.rcParams['figure.dpi'] = 360
figure(figsize=(12, 10))

plt.plot(time_dt, active_state[0,0,:], color=cg1[0], label='d.r. = 10Hz, slow')
plt.plot(time_dt, active_state[1,0,:], color=cg1[1], label='d.r. = 20Hz, slow')
plt.plot(time_dt, active_state[2,0,:], color=cg1[2], label='d.r. = 30Hz, slow')
plt.plot(time_dt, active_state[3,0,:], color=cg1[3], label='d.r. = 40Hz, slow')
plt.plot(time_dt, active_state[4,0,:], color=cg1[4], label='d.r. = 50Hz, slow')

plt.plot(time_dt, active_state[0,1,:], color=cg2[0], label='d.r. = 10Hz, fast')
plt.plot(time_dt, active_state[1,1,:], color=cg2[1], label='d.r. = 20Hz, fast')
plt.plot(time_dt, active_state[2,1,:], color=cg2[2], label='d.r. = 30Hz, fast')
plt.plot(time_dt, active_state[3,1,:], color=cg2[3], label='d.r. = 40Hz, fast')
plt.plot(time_dt, active_state[4,1,:], color=cg2[4], label='d.r. = 50Hz, fast')

# plt.plot(time_dt[pk], active_state[0,pk], 'r*')
# plt.plot(time_dt[end], active_state[0,end], 'r*')
plt.xlabel('Time [s]')
plt.ylabel('Active state')
plt.legend(loc='lower right')
plt.grid()
plt.title('MN-driven model sensitivity (activation)', weight='bold')
plt.show()
    



figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, free_Ca[0,0,:], label='d.r. = 10Hz')
plt.plot(time_dt, free_Ca[1,0,:], label='d.r. = 20Hz')
plt.plot(time_dt, free_Ca[2,0,:], label='d.r. = 30Hz')
plt.plot(time_dt, free_Ca[3,0,:], label='d.r. = 40Hz')
plt.plot(time_dt, free_Ca[4,0,:], label='d.r. = 50Hz')
plt.ylabel('Free [$Ca^{++}$] [$\mu$M]')
           
plt.legend(loc='lower right')
plt.grid()
plt.title('Slow MU')
plt.ylim((-0.10*10**-5, 1.3*10**-5))
plt.xlim((0, 0.3))

plt.subplot(2,1,2)
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, free_Ca[0,1,:], label='d.r. = 10Hz')
plt.plot(time_dt, free_Ca[1,1,:], label='d.r. = 20Hz')
plt.plot(time_dt, free_Ca[2,1,:], label='d.r. = 30Hz')
plt.plot(time_dt, free_Ca[3,1,:], label='d.r. = 40Hz')
plt.plot(time_dt, free_Ca[4,1,:], label='d.r. = 50Hz')
plt.xlabel('Time [s]')
plt.ylabel('Free [$Ca^{++}$] [$\mu$M]')
plt.legend(loc='lower right')
plt.grid()
plt.title('Fast MU')
plt.ylim((-0.10*10**-5, 2*10**-5))
plt.xlim((0, 0.3))

plt.suptitle('MN-driven model sensitivity (excitation)', weight='bold',  y=0.94)
plt.show()



    