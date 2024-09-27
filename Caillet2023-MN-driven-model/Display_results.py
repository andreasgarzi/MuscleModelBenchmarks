""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2023
Imperial College London
Department of Civil Engineering
This code displays the results presented in the manuscript Caillet et al. 'Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy' (2023)
---------
Figure 6: validation of the Common Control computed with Nr and N MN spike trains against experimental Force
Figure 8: validation of the predicted muscle forces (with the 3 different neural controls and F0MU distribution) against experimental Force
Figures 7A-D: distributions across the MU samples of MU active states, FMU, and F0MU

"""

import warnings
warnings.filterwarnings("ignore")
# from IPython import get_ipython;   
# get_ipython().magic('reset -sf')
import sys
sys.path.insert(0,'Modules')
import numpy as np
import os 
cwd = os.getcwd()
from pathlib import Path
root = Path(".")
path_to_data = root / "Results" 
#path_to_data = root
import scipy 
import matplotlib.pyplot as plt
import math
from matplotlib.pyplot import figure
from MN_properties_relationships_MOD import  Fth_distrib_func
from dataname_MOD import dataname_func #Builds the paths to the data files
from find_MUmax_MOD import find_MUmax_func
from metrics_MOD import metrics_func
from distribute_MOD import distribute_func
from norm_CC_MOD import norm_CC_func
from RMS_func import RMS_func

#------------------------------------------------------------------------------
#Choose Trial, the results of which you wish to plot
# Trial = 'S1_30_256' #30% MVC - 256 electrodes
Trial = 'S1_30_64L'
# Trial = 'S1_30_36L'
# Trial = 'S1_50_256'
# Trial = 'S1_50_64L'
# Trial = 'BBmax'
# Trial = 'BBsub'
#------------------------------------------------------------------------------
# Loading preliminary data (preprocessed MN spike trains - subject-speciific MSK data - recorded force - other simultaiton parameters)
from load_Input_Data_MOD import load_Input_Data_func
time, time_dt, muscle, MVC, Transd_Force, muscle_F0M, Nb_MN, MN_pop, Real_MN_pop, exp_disch_times, Firing_times_sim, range_start,range_stop, t_start, plateau_time1, plateau_time2, end_force, d, dt, fs = load_Input_Data_func(Trial, path_to_data)

#------------------------------------------------------------------------------
# Paths to the files that store the Force results obtained with the MN-driven model
path_to_data = path_to_data / Trial 
path = np.array([    path_to_data / 'evenly',       path_to_data / 'identified',        path_to_data / '400' ])
Data = np.array(['time', 'time_exp', 'scaled_exp_force_in_N', 'MNAP_list', 'MUAP_list', 'freeCa_list' , 'boundCa_list','MU_act_list', 'norm_force_list', 'total_muscle_force', 'F0MU_distrib' ])
N_Nr_Input = np.array([ 'Nr_', 'Nr_', '400_'])
distrib_approach = np.array(['_evenly_', '_identified_', '_'])

#------------------------------------------------------------------------------
# LOADING THE EXPERIMENTAL FORCE (specifically from the 400_Trial_Data files)
data_npy_files_list = dataname_func(N_Nr_Input, Trial, distrib_approach, Data, 2) 
time_exp = np.load(path[2] / data_npy_files_list[1], allow_pickle=True) # follows the "Data" order
FM_exp = np.load(path[2] / data_npy_files_list[2], allow_pickle=True) 
time_sim = np.arange(0,38,0.0001)
time_red = np.linspace(0,10,380000)

#Load Caillet simulated forces
data_npy_files_list = dataname_func(N_Nr_Input, Trial, distrib_approach, Data, 0)
FM_sim_ev = np.load(path[0] / data_npy_files_list[9], allow_pickle=True) #Load simulated force by Caillet 2023
data_npy_files_list = dataname_func(N_Nr_Input, Trial, distrib_approach, Data, 1)
FM_sim_id = np.load(path[1] / data_npy_files_list[9], allow_pickle=True)

#%%

# os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\pyHatze-Local\\Caillet2023-MN-driven-model\\Results_withTendon\\Nr_S1_30_64L')

# F_LPF = np.load('Nr_S1_30_64L_identified_total_muscle_force_nopennation&delays_LPFfibres.npy', allow_pickle=True)
# F_noLPF = np.load('Nr_S1_30_64L_identified_total_muscle_force_nopennation&delays.npy', allow_pickle=True)
# F_noLPF = np.array(F_noLPF, dtype=float)
# F_id = np.load('Nr_S1_30_64L_identified_total_muscle_force_tendon2.npy', allow_pickle=True)
# F_ev = np.load('Nr_S1_30_64L_evenly_total_muscle_force_tendon2.npy', allow_pickle=True)
#f_T = np.load('Nr_S1_30_64L_identified_Tendon_force_nopennation.npy', allow_pickle=True)
#eps_T = np.load('Nr_S1_30_64L_identified_Tendon_strain_nopennation.npy', allow_pickle=True)
#l_T = np.load('Nr_S1_30_64L_identified_Tendon_length_nopennation.npy', allow_pickle=True)
#f_PE = np.load('Nr_S1_30_64L_identified_PEE_force_nopennation.npy', allow_pickle=True)

os.chdir(cwd)

#%% 
"""Some error metrics"""
#RMSE_noLPF = metrics_func(FM_exp, Tot_force_noLPF, time_exp, time_sim, MVC, plateau_time1, plateau_time2)
#RMSE_LPF = metrics_func(FM_exp, Tot_force_LPF, time_exp, time_sim, MVC, plateau_time1, plateau_time2)
# MSE = np.square(np.subtract(F_LPF ,F_noLPF)).mean() 
# RMSE_LPF = math.sqrt(MSE)
# RMSE_perc_LPF = np.sqrt(np.mean(np.square(((F_LPF - F_noLPF) / F_noLPF)*100), axis=0))
# RMSE_ev = metrics_func(FM_exp, FM_sim_ev, time_exp, time_sim, MVC, plateau_time1, plateau_time2)
# RMSE = metrics_func(FM_exp, FM_sim_id, time_exp, time_sim, MVC, plateau_time1, plateau_time2)

# #Linear regression between LPFed force and not
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F_LPF, F_noLPF)

# coef = np.polyfit(F_LPF, F_noLPF, 1)
# poly1d_fn = np.poly1d(coef) 
# # poly1d_fn is now a function which takes in x and returns an estimate for y
# plt.figure()
# plt.scatter(F_LPF, F_noLPF, c='None', marker='o', edgecolors='g', alpha=0.4) 
# plt.plot(F_LPF, poly1d_fn(F_LPF), 'r')
# plt.grid()
# plt.xlabel('Predicted Force-LPF (N)')
# plt.ylabel('Predicted Force-noLPF (N)')
# plt.title('Fibers LPF effect: Linear regression')
# plt.axis('square')

#%%

# LPF Effect
# plt.figure()
# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_dt, F_LPF, 'b', linestyle = 'dashed', label = 'fLPF')
# plt.plot(time_dt, F_noLPF, 'y', linestyle = 'dotted', label = 'no fLPF')
# plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('TA Force (N)')
# plt.legend()

# Identified distributed MUs result comparison
# plt.figure()
# figure(figsize=(12, 4))
# plt.subplot(1,2,1)
# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_dt, FM_sim_id, 'b', label = 'Caillet et al.2023 model')
# plt.plot(time_dt, F_id, 'r', label = 'Updated model')
# plt.plot(time_exp, FM_exp, 'k', label = 'Exp.TA force')
# plt.grid()
# plt.title('35 MUs, identified distribution')
# plt.xlabel('Time [s]')
# plt.ylabel('TA Force (N)')
# plt.legend(loc='lower center')

# # Evenly distributed MUs result comparison
# plt.subplot(1,2,2)
# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_dt, FM_sim_ev, 'b--', label = 'Caillet et al.2023 model')
# plt.plot(time_dt, F_ev, 'r--', label = 'Updated model')
# plt.plot(time_exp, FM_exp, 'k', label = 'Exp.TA force')
# plt.grid()
# plt.title('35 MUs, even distribution')
# plt.xlabel('Time [s]')
# plt.ylabel('TA Force (N)')
# plt.legend(loc='lower center')

# plt.plot(time_dt, f_T[2,:], 'b', label = 'Norm. tendon force')
# plt.plot(time_dt, f_PE[2,:], 'k', label = 'Norm. PEE force')
# plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Norm. F')
# plt.legend()

# plt.subplot(2,2,3)
# plt.plot(time_dt, eps_T[2,:], 'b', label = 'Tendon strain')
# plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Tendon strain')

# plt.subplot(2,2,4)
# plt.plot(time_dt, l_T[2,:], 'b', label = 'Norm. l_T')
# plt.grid()
# plt.xlabel('Time [s]')
# plt.ylabel('Norm. l_T')


#%%
#------------------------------------------------------------------------------
# COMPUTING THE COMMON CONTROLS
# norm_CC_exp = norm_CC_func(Nb_MN, time, exp_disch_times, plateau_time1, plateau_time2, fs, 'sample') # from Nr spike trains
# norm_CC_reconstructed = norm_CC_func(MN_pop, time, Firing_times_sim, plateau_time1, plateau_time2, fs, 'sec') # From N=400 spike trains (reconstructed MN pool)
# norm_FM_exp = FM_exp / np.mean(FM_exp[int(plateau_time1*fs): int(plateau_time2*fs)]) # Normalized TA force to mean plateau value

# time_shaped = time[range_start:range_stop] #actual time line

# plt.rcParams['figure.dpi'] = 360
# plt.plot(time_shaped, norm_CC_exp[range_start:range_stop],  linewidth=2,linestyle='dotted', label ='common control  [0-4Hz] from the '+str(Nb_MN)+' recorded MNs')
# plt.plot(time_shaped, norm_CC_reconstructed[range_start:range_stop]*1.03, linewidth=2, color='red', label ='common control  [0-4Hz] from the 400 virtual MNs')
# plt.plot(time_shaped,norm_FM_exp[range_start:range_stop]*1.03,  '-k', linewidth=2, label='Experimental transducer force')#' Whole Muscle Force '+r'$F^{MT} (t)$' )
# plt.xlabel('Time [s]', color='k',fontsize=13)
# plt.ylabel('Normalized Common Control / Exp Force', color='k',fontsize=13)
# plt.xlim(t_start,end_force)
# plt.ylim(0,1.2)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.title('Figure 6')
# plt.legend()
# plt.show()

#------------------------------------------------------------------------------
# PLOTTING THE WHOLE MUSCLE FORCES AND GENERATING THE VALIDAITON METRICS (Figure 8)
metrics=np.zeros((3,8))
linest = np.array(['dotted', 'dashed', 'solid'])
col = np.array(['steelblue', 'green', 'red'])


for i in range(len(N_Nr_Input)): # Investigating the 3 types of neural controls  
    data_npy_files_list = dataname_func(N_Nr_Input, Trial, distrib_approach, Data, i)
    time_sim = np.load(path[i] / data_npy_files_list[0], allow_pickle=True) 
    FM_sim = np.load(path[i] / data_npy_files_list[9], allow_pickle=True) #Load simulated force
 
    if i==2: 
        # Plot the Experimental Force only once
        plt.plot(time_exp, FM_exp, color = 'black', linewidth = 2)  
        
        # In the case of completely reconstructed pool, let's remove in the following the MUs that fire at less than 7Hz (see manuscript for details)
        MUs_below_7Hz = find_MUmax_func(Firing_times_sim)
      
    # Plot simulated forces
    plt.plot(time_sim, FM_sim, color = col[i], linestyle = linest[i])
    
    # The metrics are d1, F_exp_d1, ME, RMS_total, RMS_ramp1, RMS_plateau, RMS_ramp2, r2
    metrics[i,:] = metrics_func(FM_exp, FM_sim, time_exp, time_sim, MVC, plateau_time1, plateau_time2)
    if i==2: 
        print(metrics)

plt.xlim(0,end_force)
plt.ylim(0,int(max(FM_exp)/100+1)*100)    
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (s)', fontsize=13)
plt.ylabel('TA Force (N)', fontsize=13)
plt.rcParams['figure.dpi'] = 360
plt.title('Figure 8')
plt.figure()


# /!\ The storage data for the distributions of MU active states and FMU are heavy (0.1-1.5 Go).
# Those files are not shared with the opens-source package but can be generated and saved with the MN_driven_model.py code by just re-running the code

# try:
# #------------------------------------------------------------------------------
# # PLOTTING THE DISTRIBUTION OF MU ACTIVE STATES (Figure 7A)
#     for i in range(0,3):
#         data_npy_files_list=dataname_func(N_Nr_Input, Trial, distrib_approach,Data, i)
#         a_plateau_mean = distribute_func(i, 7, path, data_npy_files_list, plateau_time1, plateau_time2, Trial)
#         if i<2:
#             plt.scatter(Real_MN_pop, a_plateau_mean, s=36, marker = '^', color = 'green')        
#         else:
#             a_plateau_mean=a_plateau_mean[0:MUs_below_7Hz]
#             plt.scatter(np.arange(0, len(a_plateau_mean)), a_plateau_mean, s=6, color = 'red',marker = 's')
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     plt.xlabel('MU population', fontsize=13)
#     plt.ylabel('Mean activation across plateau', fontsize=13)
#     plt.title('Figure 7A')
#     plt.figure()


# #------------------------------------------------------------------------------
#     # PLOTTING THE DISTRIBUTION OF MU F0MU and FMU (Figure 7B-D) 
#     for i in range(0,3):
#         data_npy_files_list=dataname_func(N_Nr_Input, Trial, distrib_approach, Data,i)
#         FMU_plateau_mean = distribute_func(i, 8, path, data_npy_files_list, plateau_time1, plateau_time2, Trial)
#         f0MU = np.load(path[i] / data_npy_files_list[-1], allow_pickle=True)
#         if i==0: # Nr MUs, homogeneous distribution of F0MU
#             last_recruited_MU = np.argwhere(Fth_distrib_func(np.arange(1,401), 400)<MVC)[-1][0]+1
#             MU_list_identified = np.arange(1,len(f0MU)+1)  * last_recruited_MU//len(f0MU)
#             plt.scatter(MU_list_identified, f0MU, s=6,  color = 'black')        
#             plt.scatter(MU_list_identified, FMU_plateau_mean, s=60, marker = '3', color = 'blue')
#             plt.title('Figure 7B')
#             plt.xlabel('MU population', fontsize=13)
#             plt.ylabel('Mean MU Force across plateau', fontsize=13)
#             for z in range(len(Real_MN_pop)):
#                 plt.plot([MU_list_identified[z],MU_list_identified[z]], [FMU_plateau_mean[z],f0MU[z]], linestyle='--', color='grey', linewidth=0.5)
#         elif i==1: # Nr MUs, located into the MU pool before f0MU assignment
#             plt.scatter(Real_MN_pop, f0MU, s=6,  color = 'black')        
#             plt.scatter(Real_MN_pop, FMU_plateau_mean, s=36, marker = '^', color = 'green') 
#             plt.title('Figure 7C')
#             plt.xlabel('MU population', fontsize=13)
#             plt.ylabel('Mean MU Force across plateau', fontsize=13)
#             for z in range(len(Real_MN_pop)):
#                 plt.plot([Real_MN_pop[z],Real_MN_pop[z]], [FMU_plateau_mean[z],f0MU[z]], linestyle='--', color='grey', linewidth=0.5)
#         else: # Completely reconstructed population
#             FMU_plateau_mean=FMU_plateau_mean[0:MUs_below_7Hz]
#             plt.scatter(np.arange(0, len(FMU_plateau_mean)), f0MU[0:MUs_below_7Hz], s=2,  color = 'black')        
#             plt.scatter(np.arange(0, len(FMU_plateau_mean)), FMU_plateau_mean, s=6, color = 'red')
#             plt.title('Figure 7D')
#             plt.xlabel('MU population', fontsize=13)
#             plt.ylabel('Mean MU Force across plateau', fontsize=13)
#         plt.xticks(fontsize=13)
#         plt.yticks(fontsize=13)
#         plt.figure()
# except:
#     pass


