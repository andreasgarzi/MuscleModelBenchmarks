"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBME
Created on Sun May 19 09:08:35 2024
___________________________________

Display Biological Benchmarks results
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import scipy as sp
sys.path.insert(0,'Modules')
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import os
from scipy.stats import ttest_ind
cwd = os.getcwd()
from pathlib import Path
root = Path(".")
path_to_res = root / "Results"
import matplotlib.pyplot as plt

#%%
#Choose Trial the results of which you wish to plot
#Trial = 'BBmax'
Trial = 'BBsub'

#%%

if Trial == 'BBmax':
    
    dir1 = "C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\maximalActivation"
    os.chdir(dir1) #max activation BB dir path
    disp_bb = np.genfromtxt('displacement.dat', delimiter='') #BB time & displacement data
    force_bb = []
    force_bb_Hatze = []
    force_bb_Millard = []
    
    fig, axs = plt.subplots(6, 1, figsize=(10, 12))
    
    dt = 0.0001 # time step (x-data)
    t_end = 2 # total seconds
    time_dt = np.arange(0, t_end, dt) # time for BB
    muscle_F0M = 1.36
    
    abs_err = []
    abs_err_Hatze = []
    abs_err_Millard = []
    mean_err = np.empty((6), dtype=object)
    max_err = np.empty((6), dtype=object)
    mean_err_Hatze = np.empty((6), dtype=object)
    max_err_Hatze = np.empty((6), dtype=object)
    mean_err_Millard = np.empty((6), dtype=object)
    max_err_Millard = np.empty((6), dtype=object)
    RMSE = np.empty((6), dtype=object)
    
    for i in range(6):
        
        force_bb.append(np.genfromtxt('force_trial'+str(i+1)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
        force_bb[i] = sp.interpolate.interp1d(force_bb[i][:,0], force_bb[i][:,1], kind='cubic')(np.arange(0,t_end,dt))
        
        R = np.load('R'+str(i)+'.npy', allow_pickle=True)
        
        #______________________________________________________________________
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\Benchmarks results + digitized calcium curves\\Benchmarks outputs\\Hatze\\Maximal-Activation")
        force_bb_Hatze.append(pd.read_csv('maximalActivationBenchmark_Trial'+str(i+1)+'.csv', delimiter = ','))
        os.chdir(dir1)
        #______________________________________________________________________
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\maximalActivation")
        force_bb_Millard.append(pd.read_csv('results_damped_equilibrium_trial'+str(i+1)+'.csv', delimiter = ','))
        force_bb_Millard[i] = sp.interpolate.interp1d(force_bb_Millard[i]['time'], force_bb_Millard[i]['force'], kind='cubic')(np.arange(0,1.999,dt))
        os.chdir(dir1)
        #______________________________________________________________________
        
        #Errors metrics
        abs_err.append((np.abs(force_bb[i]/muscle_F0M - R))*100)
        abs_err_Hatze.append((np.abs(force_bb[i]/1.234 - force_bb_Hatze[i]['F_TOT(t)']))*100)
        abs_err_Millard.append((np.abs(force_bb[i][0:19990]/1.27 - force_bb_Millard[i]/1.27))*100)
          
        mean_err[i] = (np.mean(abs_err[i]))
        max_err[i] = (np.max(abs_err[i]))
        mean_err_Hatze[i] = (np.mean(abs_err_Hatze[i]))
        max_err_Hatze[i] = (np.max(abs_err_Hatze[i]))
        mean_err_Millard[i] = (np.mean(abs_err_Millard[i]))
        max_err_Millard[i] = (np.max(abs_err_Millard[i]))
        #______________________________________________________________________
        
        plt.subplot(6,1,i+1)
        plt.rcParams['figure.dpi'] = 400
        plt.plot(time_dt, force_bb[i]/1.234, 'k', label = 'Experimental')
        plt.plot(force_bb_Hatze[i]['t'], force_bb_Hatze[i]['F_TOT(t)'], 'g', label = 'Hatze model')
        plt.plot(time_dt, R*muscle_F0M/1.234, 'r', label = 'MN-driven model')
        #plt.plot(time_dt[0:19990], force_bb_Millard[i], 'b', label = 'Millard model')
        plt.grid()
        plt.xlim((-0.03,2.03))
        plt.ylim((-0.05,1.6))
        
        if i == 0:
            plt.title(u"Max. displ. = \u00B1 0.05 mm", x = 0.16, y = 0.97, weight='bold')
            plt.legend(loc=(0.7,-0.18), fontsize=12)
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif i == 1:
            plt.title(u"Max. displ. = \u00B1 0.10 mm", x = 0.16, y = 0.97, weight='bold')
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif i == 2:
            plt.title(u"Max. displ. = \u00B1 0.25 mm", x = 0.16, y = 0.97, weight='bold')
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif i == 3:
            plt.title(u"Max. displ. = \u00B1 0.50 mm", x = 0.16, y = 0.97, weight='bold')
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif i == 4:
            plt.title(u"Max. displ. = \u00B1 1.00 mm", x = 0.16, y = 0.97, weight='bold')
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif i == 5:
            path_Theo = "MA_Benchmark_2mm_Hatze_final.csv"
            data_Theo = pd.read_csv(path_Theo, delimiter = ';')
            data_Theo = data_Theo.to_numpy() 
            #plt.plot(data_Theo[:,0], data_Theo[:,1], 'g')
            
            # path_mill = "results_damped_equilibrium_trial6.csv"
            # data_mill = pd.read_csv(path_mill, delimiter = ',')
            # data_mill = data_mill.to_numpy() 
            # plt.plot(data_mill[:,0], data_mill[:,1]/1.27, 'b--')
            
            #plt.ylabel(r'Muscle Force [$\frac{F}{F_{iso}}$]')
            plt.title(u"Max. displ. = \u00B1 2.00 mm", x = 0.16, y = 0.97, weight='bold')
            plt.xlabel('Time [s]', weight='bold', fontsize=14)
        
        
    #plt.suptitle('Maximal-Activation Biological Benchmark', weight='bold', y=0.92, fontsize=16)
    fig.supylabel('Normalized Force', x=0.06, weight='bold', fontsize=14)
    
    os.chdir(cwd)
 
    t1mean, p1mean = ttest_ind(list(mean_err), list(mean_err_Millard), equal_var=False)
    t2mean, p2mean = ttest_ind(list(mean_err_Hatze), list(mean_err_Millard))
    t1max, p1max = ttest_ind(list(max_err), list(max_err_Millard))
    t2max, p2max = ttest_ind(list(max_err_Hatze), list(max_err_Millard))
    
    x_err = [1,2,3,4,5,6]

    figure(figsize=(10, 3))
    plt.subplot(1,2,1)  
    plt.rcParams['figure.dpi'] = 400 
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, mean_err, 'r-o', label = 'MN-driven model')
    plt.plot(x_err, mean_err_Hatze, 'g-o', label='Hatze model')
    plt.plot(x_err, mean_err_Millard, 'b-o', label='Millard model')
    plt.ylabel('Mean abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.xlabel('Displ. amplitude [mm]', weight='bold', fontsize=15)
    #plt.xlabel('Trial', weight='bold', fontsize=15)
    plt.grid()
    plt.subplot(1,2,2)  
    plt.rcParams['figure.dpi'] = 400 
    plt.plot(x_err, max_err, 'r-o', label = 'MN-driven model')
    plt.plot(x_err, max_err_Hatze, 'g-o', label='Hatze model')
    plt.plot(x_err, max_err_Millard, 'b-o', label='Millard model')
    plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    #plt.xlabel('Displ. amplitude [mm]', weight='bold', fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.grid()
    
#%%

if Trial == 'BBsub':
    
    dt = 0.0001 # time step (x-data)
    t_end = 2 # total seconds
    time_dt = np.arange(0, t_end, dt) # time for BB
    muscle_F0M = 25.1
    
    #__________________________________________________________________________
    """ Fixed Frequency BB """
    
    dir1 = "C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\fixedfreq_yielding"
    os.chdir(dir1) #max activation BB dir path
    
    force_bb = []
    force_bb_Hatze_fixed = []
    force_bb_Millard_fixed = []

    fig, ax = plt.subplots(3, 2, figsize=(12, 11))
    
    abs_err_fixed = []
    abs_err_fixed_0 = []
    abs_err_Hatze_fixed = []
    abs_err_Millard_fixed = []
    mean_err_fixed = np.empty((6), dtype=object)
    max_err_fixed = np.empty((6), dtype=object)
    mean_err_Hatze_fixed = np.empty((6), dtype=object)
    max_err_Hatze_fixed = np.empty((6), dtype=object)
    mean_err_Millard_fixed = np.empty((6), dtype=object)
    max_err_Millard_fixed = np.empty((6), dtype=object)
    
    RMSE_fixed = np.empty((6), dtype=object)
    
    mean_err_fixed_0 = np.empty((6), dtype=object)
    max_err_fixed_0 = np.empty((6), dtype=object)
    
    RMSE_fixed_0 = np.empty((6), dtype=object)
    
    w = [1, 7, 2, 8, 3, 9]  # Hatze files order
    s = 0
    
    for i in [0, 3, 1, 4, 2, 5]:
        
        force_bb.append(np.genfromtxt('force_trial'+str(i)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
        force_bb[s] = sp.interpolate.interp1d(force_bb[s][:,0], force_bb[s][:,1], kind='cubic')(np.arange(0,t_end,dt))
        
        R = np.load('R'+str(i)+'.npy', allow_pickle=True)  # for visualization
        
        #______________________________________________________________________
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\Benchmarks results + digitized calcium curves\\Benchmarks outputs\\Hatze\\Submaximal-Activation")
        force_bb_Hatze_fixed.append(pd.read_csv('submaximalActivationBenchmark_Trial'+str(w[s])+'.csv', delimiter = ','))
        force_bb_Hatze_fixed[s] = sp.interpolate.interp1d(force_bb_Hatze_fixed[s]['t'], force_bb_Hatze_fixed[s]['F_TOT(t)'], kind='cubic')(np.arange(0,t_end,dt))
        force_bb_Hatze_fixed[s] = force_bb_Hatze_fixed[s]*muscle_F0M
        os.chdir(dir1)
        
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\fixedfreq_nocomp") #max activation BB dir path
        R_0 = np.load('R'+str(i)+'.npy', allow_pickle=True)
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\fixedfreq_yielding") #max activation BB dir path
        
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation")
        force_bb_Millard_fixed.append(pd.read_csv('results_damped_equilibrium_trial'+str(w[s]+6)+'.csv', delimiter = ','))
        force_bb_Millard_fixed[s] = sp.interpolate.interp1d(force_bb_Millard_fixed[s]['time'], force_bb_Millard_fixed[s]['force'], kind='cubic')(np.arange(0,t_end,dt))
        os.chdir(dir1)
        #______________________________________________________________________
        #Errors metrics
        #abs_err_fixed = np.empty((len(time_dt)), dtype=object)
        abs_err_fixed.append((np.abs(force_bb[s] - R)/muscle_F0M)*100)
        abs_err_Hatze_fixed.append((np.abs(force_bb[s] - force_bb_Hatze_fixed[s])/muscle_F0M)*100)
        abs_err_Millard_fixed.append((np.abs(force_bb[s] - force_bb_Millard_fixed[s])/muscle_F0M)*100)
        
        abs_err_fixed_0.append((np.abs(force_bb[s] - R_0)/muscle_F0M)*100)
                
        RMSE_fixed[s] = (np.sqrt(np.mean(np.square(((force_bb[s] - R))))) / np.abs((np.max(force_bb[s]) - np.min(force_bb[s])))) * 100
        RMSE_fixed_0[s] = (np.sqrt(np.mean(np.square(((force_bb[s] - R_0))))) / np.abs((np.max(force_bb[s]) - np.min(force_bb[s])))) * 100

        mean_err_fixed[s] = (np.mean(abs_err_fixed[s]))
        max_err_fixed[s] = (np.max(abs_err_fixed[s]))
        mean_err_Hatze_fixed[s] = (np.mean(abs_err_Hatze_fixed[s]))
        max_err_Hatze_fixed[s] = (np.max(abs_err_Hatze_fixed[s]))
        mean_err_Millard_fixed[s] = (np.mean(abs_err_Millard_fixed[s]))
        max_err_Millard_fixed[s] = (np.max(abs_err_Millard_fixed[s]))
        
        mean_err_fixed_0[s] = (np.mean(abs_err_fixed_0[s]))
        max_err_fixed_0[s] = (np.max(abs_err_fixed_0[s]))
        #______________________________________________________________________
        
        plt.subplot(3,2,s+1)
        plt.rcParams['figure.dpi'] = 400
        plt.plot(time_dt, force_bb[s], 'k', label = 'Experimental')
        plt.plot(time_dt, force_bb_Hatze_fixed[s], 'g', label = 'Hatze model')
        plt.plot(time_dt, R, 'r', label = 'MN-driven model')
        plt.plot(time_dt, force_bb_Millard_fixed[s], 'b')
        #plt.plot(time_dt, R_0, 'r--', label = 'Simulated (no yielding)')
        plt.grid()
        # plt.xlim((-0.03,2.03))
        # plt.ylim((-0.05,1.6))
        
        if s == 0:
            #plt.title(u"\u00B1 1 mm", x = 0.83, y = 0.85, fontsize=13)
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif s == 1:
            figg = plt.subplot(3,2,2)
            #plt.title(u"\u00B1 8 mm", x = 0.83, y = 0.85, fontsize=13)
            ax = figg
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif s == 2:
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
                
        elif s == 3:
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)  
                    
        elif s == 4:
            plt.xlabel('Time [s]', weight = 'bold', fontsize = 13)
            plt.legend(loc='lower center', fontsize=13)
            #plt.ylabel('Force [N]', weight = 'bold', fontsize = 13)
            
        elif s == 5:
            # path_Theo = "MA_Benchmark_2mm_Hatze_final.csv"
            # data_Theo = pd.read_csv(path_Theo, delimiter = ';')
            # data_Theo = data_Theo.to_numpy() 
            #plt.plot(data_Theo[:,0], data_Theo[:,1], 'g')
            
            # path_mill = "results_damped_equilibrium_trial6.csv"
            # data_mill = pd.read_csv(path_mill, delimiter = ',')
            # data_mill = data_mill.to_numpy() 
            # plt.plot(data_mill[:,0], data_mill[:,1]/1.27, 'b--')
            
            #plt.ylabel(r'Muscle Force [$\frac{F}{F_{iso}}$]')
            plt.xlabel('Time [s]', weight = 'bold', fontsize = 13)
        
        s = s+1
        
        
    plt.suptitle('Sub-maximal Activation Biological Benchmark (Constant d.r.)', weight='bold', y=0.92, fontsize=14)
    fig.supylabel('Force [N]', x=0.07, weight = 'bold', fontsize=13) 
    
    disp1 = [0, 2, 4]
    disp2 = [1, 3, 5]
    x_err = [10, 20, 30]
    
    figure(figsize=(10, 8))
    plt.subplot(2,2,1)  
    plt.rcParams['figure.dpi'] = 400 
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(mean_err_fixed[disp1], 'r-o', label = 'MN-driven model')
    plt.plot(mean_err_Hatze_fixed[disp1], 'g-o', label='Hatze model')
    plt.plot(mean_err_Millard_fixed[disp1], 'b-o', label='Millard damped eq. model')
    plt.ylabel('Mean abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.legend(loc=(0.2, -0.1), fontsize=15)
    #plt.xlabel('Trial', weight='bold', fontsize=15)
    plt.grid()
    plt.subplot(2,2,2)  
    
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, mean_err_fixed[disp2], 'r-o', label = 'MN-driven model')
    plt.plot(x_err, mean_err_Hatze_fixed[disp2], 'g-o', label='Hatze model')
    plt.plot(x_err, mean_err_Millard_fixed[disp2], 'b-o', label='Millard damped eq. model')
    #plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.xlabel('Frequency', weight='bold', fontsize=15)
    plt.legend(loc=(0.2, -0.1), fontsize=15)
    plt.grid()
    plt.subplot(2,2,3)  
     
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, max_err_fixed[disp1], 'r-o', label = 'MN-driven model')
    plt.plot(x_err, max_err_Hatze_fixed[disp1], 'g-o', label='Hatze model')
    plt.plot(x_err, max_err_Millard_fixed[disp1], 'b-o', label='Millard damped eq. model')
    plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.legend(loc='lower center', fontsize=15)
    #plt.xlabel('Trial', weight='bold', fontsize=15)
    plt.grid()
    plt.subplot(2,2,4)  
    
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, max_err_fixed[disp2], 'r-o', label = 'MN-driven model')
    plt.plot(x_err, max_err_Hatze_fixed[disp2], 'g-o', label='Hatze model')
    plt.plot(x_err, max_err_Millard_fixed[disp2], 'b-o', label='Millard damped eq. model')
    #plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.xlabel('Frequency', weight='bold', fontsize=15)
    
    plt.grid()
    
    #__________________________________________________________________________
    """ Variable Frequency BB """
    
    dir1 = "C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\variablefreq_yielding"
    os.chdir(dir1) #max activation BB dir path    
    
    force_bb_var = []
    force_bb_Hatze_var = []
    force_bb_Millard_var = []
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 11))
    
    abs_err_var = []
    abs_err_var_0 = []
    abs_err_Hatze_var = []
    abs_err_Millard_var = []
    mean_err_var = np.empty((6), dtype=object)
    max_err_var = np.empty((6), dtype=object)
    mean_err_var_0 = np.empty((6), dtype=object)
    max_err_var_0 = np.empty((6), dtype=object)
    mean_err_Hatze_var = np.empty((6), dtype=object)
    max_err_Hatze_var = np.empty((6), dtype=object)
    mean_err_Millard_var = np.empty((6), dtype=object)
    max_err_Millard_var = np.empty((6), dtype=object)
    
    RMSE_var = np.empty((6), dtype=object)
    RMSE_var_0 = np.empty((6), dtype=object)
    
    w = [4, 10, 5, 11, 6, 12]
    s = 0
    
    for i in [0, 3, 1, 4, 2, 5]:
        
        force_bb_var.append(np.genfromtxt('force_trial_var'+str(i)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
        force_bb_var[s] = sp.interpolate.interp1d(force_bb_var[s][:,0], force_bb_var[s][:,1], kind='cubic')(np.arange(0,t_end,dt))

        R_var = np.load('R_var'+str(i)+'.npy', allow_pickle=True) # for visualization
        
        #______________________________________________________________________
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\Benchmarks results + digitized calcium curves\\Benchmarks outputs\\Hatze\\Submaximal-Activation")
        force_bb_Hatze_var.append(pd.read_csv('submaximalActivationBenchmark_Trial'+str(w[s])+'.csv', delimiter = ','))
        force_bb_Hatze_var[s] = sp.interpolate.interp1d(force_bb_Hatze_var[s]['t'], force_bb_Hatze_var[s]['F_TOT(t)'], kind='cubic')(np.arange(0,t_end,dt))
        force_bb_Hatze_var[s] = force_bb_Hatze_var[s]*muscle_F0M
        os.chdir(dir1)
        
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\variablefreq_nocomp") #max activation BB dir path
        R_var0 = np.load('R_var'+str(i)+'.npy', allow_pickle=True)
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation\\variablefreq_yielding") #max activation BB dir path
                
        os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation")
        force_bb_Millard_var.append(pd.read_csv('results_damped_equilibrium_trial'+str(w[s]+6)+'.csv', delimiter = ','))
        force_bb_Millard_var[s] = sp.interpolate.interp1d(force_bb_Millard_var[s]['time'], force_bb_Millard_var[s]['force'], kind='cubic')(np.arange(0,t_end,dt))
        os.chdir(dir1)
        #______________________________________________________________________
        #Errors metrics
        abs_err_var.append((np.abs(force_bb_var[s] - R_var)/muscle_F0M)*100)
        abs_err_var_0.append((np.abs(force_bb_var[s] - R_var0)/muscle_F0M)*100)
        abs_err_Hatze_var.append((np.abs(force_bb_var[s] - force_bb_Hatze_var[s])/muscle_F0M)*100)
        abs_err_Millard_var.append((np.abs(force_bb_var[s] - force_bb_Millard_var[s])/muscle_F0M)*100)
         
        RMSE_var[s] = (np.sqrt(np.mean(np.square(((force_bb_var[s] - R_var))))) / np.abs((np.max(force_bb_var[s]) - np.min(force_bb_var[s])))) * 100
        RMSE_var_0[s] = (np.sqrt(np.mean(np.square(((force_bb_var[s] - R_var0))))) / np.abs((np.max(force_bb_var[s]) - np.min(force_bb_var[s])))) * 100

        mean_err_var[s] = (np.mean(abs_err_var[s]))
        max_err_var[s] = (np.max(abs_err_var[s]))
        mean_err_Hatze_var[s] = (np.mean(abs_err_Hatze_var[s]))
        max_err_Hatze_var[s] = (np.max(abs_err_Hatze_var[s]))
        mean_err_Millard_var[s] = (np.mean(abs_err_Millard_var[s]))
        max_err_Millard_var[s] = (np.max(abs_err_Millard_var[s]))
        
        mean_err_var_0[s] = (np.mean(abs_err_var_0[s]))
        max_err_var_0[s] = (np.max(abs_err_var_0[s]))
        #______________________________________________________________________
        
        plt.subplot(3,2,s+1)
        plt.rcParams['figure.dpi'] = 400
        plt.plot(time_dt, force_bb_var[s], 'k', label = 'Experimental') 
        plt.plot(time_dt, force_bb_Hatze_var[s], 'g', label = 'Hatze model')
        plt.plot(time_dt, R_var, 'r', label = 'MN-driven model')  
        plt.plot(time_dt, force_bb_Millard_var[s], 'b')
        #plt.plot(time_dt, R_var0, 'r--', label = 'Simulated (no yielding)')
        plt.grid()
        # plt.xlim((-0.03,2.03))
        # plt.ylim((-0.05,1.6))
        
        if s == 0:
            #plt.title(u"\u00B1 1 mm", x = 0.83, y = 0.85, fontsize=13)
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif s == 1:
            figg = plt.subplot(3,2,2)
            #plt.title(u"\u00B1 8 mm", x = 0.83, y = 0.85, fontsize=13)
            ax = figg
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif s == 2:
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif s == 3:
            plt.gca().tick_params(axis='x', which='both', labelbottom=False)
            
        elif s == 4:
            plt.xlabel('Time [s]', weight = 'bold', fontsize = 13)
            plt.legend(loc='lower center', fontsize=13)
            
        elif s == 5:
            # path_Theo = "MA_Benchmark_2mm_Hatze_final.csv"
            # data_Theo = pd.read_csv(path_Theo, delimiter = ';')
            # data_Theo = data_Theo.to_numpy() 
            #plt.plot(data_Theo[:,0], data_Theo[:,1], 'g')
            
            # path_mill = "results_damped_equilibrium_trial6.csv"
            # data_mill = pd.read_csv(path_mill, delimiter = ',')
            # data_mill = data_mill.to_numpy() 
            # plt.plot(data_mill[:,0], data_mill[:,1]/1.27, 'b--')
            
            #plt.ylabel(r'Muscle Force [$\frac{F}{F_{iso}}$]')
            plt.xlabel('Time [s]', weight = 'bold', fontsize = 13)
        
        s = s+1
        
    plt.suptitle('Sub-maximal Activation Biological Benchmark (Variable d.r.)', weight='bold', y=0.92, fontsize=14)
    fig.supylabel('Force [N]', x=0.07, weight = 'bold', fontsize=13) 
    
    os.chdir(cwd)
    
    disp1 = [0, 2, 4]
    disp2 = [1, 3, 5]
    x_err = [10, 20, 30]
    
    figure(figsize=(10, 8))
    plt.subplot(2,2,1)  
    plt.rcParams['figure.dpi'] = 400 
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, mean_err_var[disp1], 'r-o', label = 'MN-driven model')
    plt.plot(x_err, mean_err_Hatze_var[disp1], 'g-o', label='Hatze model')
    plt.plot(x_err, mean_err_Millard_var[disp1], 'b-o', label='Millard damped eq. model')
    plt.ylabel('Mean abs. err. [%F0M]', weight = 'bold', fontsize=15)
    
    #plt.xlabel('Trial', weight='bold', fontsize=15)
    plt.grid()
    plt.subplot(2,2,2)  
     
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, mean_err_var[disp2], 'r-o', label = 'MN-driven model')
    plt.plot(x_err, mean_err_Hatze_var[disp2], 'g-o', label='Hatze model')
    plt.plot(x_err, mean_err_Millard_var[disp2], 'b-o', label='Millard damped eq. model')
    plt.legend(loc=(0.3, -0.2), fontsize=15)
    #plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.xlabel('Frequency', weight='bold', fontsize=15)
    plt.grid()
    plt.subplot(2,2,3)  
     
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, max_err_var[disp1], 'r-o', label = 'MN-driven model')
    plt.plot(x_err, max_err_Hatze_var[disp1], 'g-o', label='Hatze model')
    plt.plot(x_err, max_err_Millard_var[disp1], 'b-o', label='Millard damped eq. model')
    plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.legend(loc='lower center', fontsize=15)
    #plt.xlabel('Trial', weight='bold', fontsize=15)
    plt.grid()
    plt.subplot(2,2,4)  
    
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(x_err, max_err_var[disp2], 'r-o', label = 'MN-driven model')
    plt.plot(x_err, max_err_Hatze_var[disp2], 'g-o', label='Hatze model')
    plt.plot(x_err, max_err_Millard_var[disp2], 'b-o', label='Millard damped eq. model')
    #plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.xlabel('Frequency', weight='bold', fontsize=15)
    
    plt.grid()

    t1fix, p1fix = ttest_ind(list(mean_err_fixed[disp1]), list(mean_err_Millard_fixed[disp1]))
    t8fix, p8fix = ttest_ind(list(mean_err_fixed[disp2]), list(mean_err_Millard_fixed[disp2]))
    t1var, p1var = ttest_ind(list(mean_err_var[disp1]), list(mean_err_Millard_var[disp1]))
    t8var, p8var = ttest_ind(list(mean_err_var[disp2]), list(max_err_Millard_var[disp2]))

    t1fixH, p1fixH = ttest_ind(list(mean_err_Hatze_fixed[disp1]), list(mean_err_Millard_fixed[disp1]))
    t8fixH, p8fixH = ttest_ind(list(mean_err_Hatze_fixed[disp2]), list(mean_err_Millard_fixed[disp2]))
    t1varH, p1varH = ttest_ind(list(mean_err_Hatze_var[disp1]), list(mean_err_Millard_var[disp1]))
    t8varH, p8varH = ttest_ind(list(mean_err_Hatze_var[disp2]), list(max_err_Millard_var[disp2]))

    #%%
    figure(figsize=(10, 9))
    plt.subplot(2,2,1)  
    plt.rcParams['figure.dpi'] = 400 
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(np.concatenate((mean_err_fixed[disp1], mean_err_var[disp1])), 'r-o', label = 'MN-driven model (yield.)')
    plt.plot(np.concatenate((mean_err_fixed_0[disp1], mean_err_var_0[disp1])), 'r--o', label = 'MN-driven model (no yield.)')
    plt.plot(np.concatenate((mean_err_Hatze_fixed[disp1], mean_err_Hatze_var[disp1])), 'g-o', label='Hatze model')
    plt.plot(np.concatenate((mean_err_Millard_fixed[disp1], mean_err_Millard_var[disp1])), 'b-o', label='Millard damped eq. model')
    plt.ylabel('Mean abs. err. [%F0M]', weight = 'bold', fontsize=15)
    plt.legend(loc='upper center', fontsize=15)
    #plt.legend(loc=(0.2, -0.1), fontsize=15)
    #plt.xlabel('Trial', weight='bold', fontsize=15)
    plt.ylim((1, 17))
    plt.grid()
    
    plt.subplot(2,2,2)  
    plt.gca().tick_params(axis='y', which='both', labelleft=False)
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(np.concatenate((mean_err_fixed[disp2], mean_err_var[disp2])), 'r-o', label = 'MN-driven model (yield.)')
    plt.plot(np.concatenate((mean_err_fixed_0[disp2], mean_err_var_0[disp2])), 'r--o', label = 'MN-driven model (no yield.)')
    plt.plot(np.concatenate((mean_err_Hatze_fixed[disp2], mean_err_Hatze_var[disp2])), 'g-o', label='Hatze model')
    plt.plot(np.concatenate((mean_err_Millard_fixed[disp2], mean_err_Millard_var[disp2])), 'b-o', label='Millard damped eq. model')
    #plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.xlabel('Frequency', weight='bold', fontsize=15)
    #plt.legend(loc=(0.2, -0.1), fontsize=15)
    plt.grid()
    plt.ylim((1, 17))
    plt.subplot(2,2,3)  
     
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(np.concatenate((max_err_fixed[disp1], max_err_var[disp1])), 'r-o', label = 'MN-driven model (yield.)')
    plt.plot(np.concatenate((max_err_fixed_0[disp1], max_err_var_0[disp1])), 'r--o', label = 'MN-driven model (no yield.)')
    plt.plot(np.concatenate((max_err_Hatze_fixed[disp1], max_err_Hatze_var[disp1])), 'g-o', label='Hatze model')
    plt.plot(np.concatenate((max_err_Millard_fixed[disp1], max_err_Millard_var[disp1])), 'b-o', label='Millard model')
    plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    
    #plt.xlabel('Trial', weight='bold', fontsize=15)
    plt.ylim((4, 80))
    plt.grid()
    
    plt.subplot(2,2,4)  
    plt.gca().tick_params(axis='y', which='both', labelleft=False)
    plt.gca().tick_params(axis='x', which='both', labelbottom=False)
    plt.plot(np.concatenate((max_err_fixed[disp2], max_err_var[disp2])), 'r-o', label = 'MN-driven model (yield.)')
    plt.plot(np.concatenate((max_err_fixed_0[disp2], max_err_var_0[disp2])), 'r--o', label = 'MN-driven model (no yield.)')
    plt.plot(np.concatenate((max_err_Hatze_fixed[disp2], max_err_Hatze_var[disp2])), 'g-o', label='Hatze model')
    plt.plot(np.concatenate((max_err_Millard_fixed[disp2], max_err_Millard_var[disp2])), 'b-o', label='Millard damped eq. model')
    #plt.ylabel('Max abs. err. [%F0M]', weight = 'bold', fontsize=15)
    #plt.xlabel('Frequency', weight='bold', fontsize=15)
    plt.ylim((4, 80))
    plt.grid()
