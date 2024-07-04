"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue Jun  4 11:03:08 2024
___________________________________

MN-driven model with SE and PEE adapted for BB tests of Millard et. al 2023
"""

"""Select BB trial (to UNCOMMENT)"""
Trial = 'BBmax'
# Trial = 'BBsub'

""" Choose amplitude scale as well (0-5) """
s = 0

""" Saving the simulations (y/n)? """
save = 'n'

#%%____________________________________________________________________________
import sys
sys.path.insert(0,'Modules_BB')
import os
cwd = os.getcwd()
#%%____________________________________________________________________________
import numpy as np
import scipy as sp
import scipy.interpolate
#from scipy.signal import find_peaks
#from matplotlib.pyplot import figure
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
#from Activity import activity
#from Shift_length import shift_fun
#%%____________________________________________________________________________
""" Load time, displacement and force form BB tests, create virtual MU spikes"""

dt = 0.0001 # time step (x-data)
t_end = 2 # total seconds
time_dt = np.arange(0, t_end, dt) # time for BB
fd = 70 # paper stimulation freq. Hz d.r. (to reach active_state = 1 always, without any recruitment)
T = 1/fd # correspondend d.r. period
Nr = 1 # n. of an average complete rat soleus MUs pool
muscle_F0M = 1.17 # muscle max. isom. force [N]
scale = [0.05, 0.1, 0.25, 0.5, 1, 2] # amplitude disp. scales

if Trial == 'BBmax': # maximal biological benchmarks
    
    os.chdir("C:\\Users\\z5517249\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\maximalActivation") # max activation BB dir path
    disp_bb = np.genfromtxt('displacement.dat', delimiter='') #BB time & displacement data
    disp_bb_int = sp.interpolate.interp1d(disp_bb[:,0], disp_bb[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate with time_dt+1 points (l_MT must be longer)
    force_bb = (np.genfromtxt('force_trial'+str(s+1)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)

os.chdir(cwd)

# Visualize original and interpolated Biol.Benchmark imposed displacement
# plt.plot(disp_bb[:,0], disp_bb[:,1]*scale[s])
# plt.plot(np.linspace(0,2+dt,20001), disp_bb_int*scale[s])

sp_matrix = np.empty((Nr, int(t_end/T)), dtype=float)
disch = np.arange(0, t_end, T) # create array of dischare times at 70 Hz
for i in range(Nr): # append Nr times
    sp_matrix[i] = disch
    
#______________________________________________________________________________      
print('There are ', Nr, ' discharging MUs in this simulation.')

# F0MU distribution across the sample of MUs
F0MU_distribution = F0MU_distrib_func(Nr, muscle_F0M)

#______________________________________________________________________________
" RUNNING THE MN-DRIVEN MODEL FOR ALL FIRING MUS USED AS INPUTS "

l_T_slack = 17.1 # Tendon slack length (mm)
l_M_opt = l_T_slack # Optimal fiber length (mm)
l_M_0 = 1 - 2/l_M_opt # Initial fibre length normalized to l_M_0 (it would be l_MT - l_ST)
#l_M_0 = 1
alpha_0 = 6*np.pi/180 # initial pennation (pennation should make less than 2 % difference)
#l_MT_0 = l_T_slack + (l_M_opt)*np.cos(alpha_0)-2 # Musculo-tendon length (mm)
l_MT_0 = l_T_slack + (l_M_opt-2)*np.cos(alpha_0)
l_MT = l_MT_0 + disp_bb_int*scale[s] # scaled MT length

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


# ...for each considered i-th MU 
for i in range (Nr):  
    print('Computing Force for MU n°', str(i+1))
       
    MU_type = MU_type_id_func(i)  # i-th MU type identification (fast/slow)
    Matrix_AP = sp_matrix[i].astype(float)  # i-th MU discharge times [s] 
    
    def ODE_system(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha):
       
        if int(t/dt) == 0: # initial pennation angle
            alpha[int(t/dt)] = alpha_0
            
        l_T = l_MT[int(t/dt)] - (y[5]*l_M_opt)*np.cos(alpha[int(t/dt)]) # tendon length
        eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
        SE_force = T_force(eps_T) # tendon force
        PE_force = PEE_force(y[5]) # passive el. force    
        CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
        alpha[int((t+dt)/dt)] = penn_ang(l_MT[int(t+dt/dt)], y[5], l_T, l_M_0, alpha_0) # update pennation angle
        
        dbetadt, DDbetaDDt = MU_AP_func(t, y, Matrix_AP) # remember to multiply by Vmax_factor = 0.85
            
        dgammadt, DDgammaDDt = MU_free_Ca_func(t, y, y[0]*0.85, y[5], MU_type, Matrix_AP) # Free Ca (remember to avoid negligible negative values)
        
        #______________________________________________________________________
        # Rockenfeller, Gunther & Hatze's approach
        # w_l = shift_fun(y[4])
        
        # a = activity(y[2], w_l) 
            
        #______________________________________________________________________
        # Arnault's original ODEs
        # ddeltadt = MU_bound_calcium_func(t, y, y[2], y[6], MU_type, Matrix_AP) # Ca-Tn
            
        dadt = MU_active_state_func(t, y, y[2], MU_type) # Active state
        #______________________________________________________________________
        
        FL = Force_Length_func(y[5], y[4]) # F-L relationship (*active state)
            
        dldt = velo_fFV(t, y, CE_force, FL, y[4], y[5], MU_type) # velocity 
            
        #return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, ddeltadt, dadt, dldt]
        return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dadt, dldt]
     
    
    y0 = [0, 0, 0, 0, 1, l_M_0] # set initial states
    p = (l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha[i,:]) # set ODE parameters
    sol = solve_ivp(ODE_system, [time_dt[0], time_dt[-1]], y0, args=p, method='LSODA', t_eval = time_dt, max_step = dt/2) # solve IVP
    
    active_state[i,:] = sol.y[4]  # get active state
    l_M[i,:] = sol.y[5]  # get l_M
    
    # MUAP_nerve[i,:] = sol.y[0]
    # free_Ca[i,:] = sol.y[2]
    # Ca_Tn[i,:] = sol.y[4]
   
    # now recalculate data based on l_M values..
    alpha[i,0] = alpha_0
    for l in range(len(time_dt)):
        l_T[i,l] = l_MT[l] - (l_M[i,l]*l_M_opt)*np.cos(alpha[i,l]) # tendon length
        eps_T[i,l] = (l_T[i,l]-l_T_slack)/l_T_slack # new tendon strain    
        SE_force[i,l] = T_force(eps_T[i,l]) # tendon force
        PE_force[i,l] = PEE_force(l_M[i,l]) # passive el. force 
        CE_force[i,l] = SE_force[i,l]/np.cos(alpha[i,l]) - PE_force[i,l] # contractile element force
        FL_force[i,l] = Force_Length_func(l_M[i,l], active_state[i,l])
        vel[i,l] = velo_fFV(0, 0, CE_force[i,l], FL_force[i,l], active_state[i,l], l_M[i,l], MU_type)
        F_M[i,l] = f_fFV(vel[i,l], FL_force[i,l], active_state[i,l], l_M[i,l], MU_type)
        
        alpha[i,l+1] = penn_ang(l_MT[l+1], l_M[i,l], l_T[i,l], l_M_0, alpha_0) # update pennation angle 
    
    
MU_Force_list = F_M + PE_force  # scaled MU force + PEE force     

F_MU_list = F0MU_distribution[0:Nr,:] * MU_Force_list
Tot_Muscle_force = F_MU_list.sum(axis=0) # Total normalized muscle force   

#%%----------------------------------------------------------------------------
# Visual validation
plt.rcParams['figure.dpi'] = 360
plt.plot(time_dt, Tot_Muscle_force, 'r', label='Simulated Force')
plt.plot(force_bb[:,0], force_bb[:,1], 'k', label='Exp. Force')
plt.xlabel('Time [s]')
plt.ylabel('Norm. Force')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#pks, _ = find_peaks(free_Ca[2,:]) # 3.64*10**-6
#Ca_0 = free_Ca[2,pks[10]]

# plt.subplot(2,1,1)
# plt.plot(time_dt, l_T[1,:]/l_T_slack, 'b', label='Tendon')
# plt.plot(time_dt, l_M[1,:], 'g', label='Fiber')
# plt.ylabel('Norm. Length')
# plt.legend()
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(time_dt, SE_force[1,:], 'b', label='SE')
# plt.plot(time_dt, PE_force[1,:], 'r', label='PE')
# plt.xlabel('Time [s]')
# plt.ylabel('Norm. force')
# plt.legend()
# plt.grid()

#%%----------------------------------------------------------------------------
# Saving data
if save =='y':
    os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests')

    #np.save('time_sim', time_dt, allow_pickle=True) 
    #np.save('time_exp', time, allow_pickle=True) 
    #np.save('MU_act_list_pennation', active_state, allow_pickle=True) 
    #np.save('MU_force_list_pennation', F_MU_list, allow_pickle=True) 
    #np.save('F0MU_distrib', F0MU_distribution, allow_pickle=True) 
    np.save('total_muscle_force_maxBB', Tot_Muscle_force, allow_pickle=True)
    #np.save('scaled_exp_force_in_N', Exp_muscle_force, allow_pickle=True) 
    np.save('Tendon_length_maxBB', l_T[2,:], allow_pickle=True)
    np.save('Tendon_strain_maxBB', eps_T[2,:], allow_pickle=True)
    np.save('Tendon_force_maxBB', SE_force[2,:], allow_pickle=True)
    np.save('PEE_force_maxBB', PE_force[2,:], allow_pickle=True)
    
    #encore...
    #     np.save('MNAP_list', MN_AP_list, allow_pickle=True) 
    #     np.save('MUAP_list', MUAP_nerve, allow_pickle=True) 
    #     np.save('freeCa_list', free_Ca, allow_pickle=True) 
    #     np.save('boundCa_list', Ca_Tn, allow_pickle=True)         
     
os.chdir(cwd)    