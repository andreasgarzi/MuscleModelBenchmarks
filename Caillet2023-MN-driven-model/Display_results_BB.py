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
sys.path.insert(0,'Modules')
import numpy as np
import pandas as pd
import os
cwd = os.getcwd()
from pathlib import Path
root = Path(".")
path_to_res = root / "Results"
import matplotlib.pyplot as plt

#______________________________________________________________________________
#Choose Trial the results of which you wish to plot
Trial = 'BBmax'
# Trial = 'BBsub'
#______________________________________________________________________________

if Trial == 'BBmax':
    
    os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\maximalActivation") #max activation BB dir path
    disp_bb = np.genfromtxt('displacement.dat', delimiter='') #BB time & displacement data
    force_bb = []
    fig, axs = plt.subplots(6, 1, figsize=(10, 12))
    
    dt = 0.0001 # time step (x-data)
    t_end = 2 # total seconds
    time_dt = np.arange(0, t_end, dt) # time for BB
    
    for i in range(6):
        
        force_bb.append(np.genfromtxt('force_trial'+str(i+1)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
        
        R = np.load('R'+str(i)+'.npy', allow_pickle=True)
        
        plt.subplot(6,1,i+1)
        plt.rcParams['figure.dpi'] = 400
        plt.plot(force_bb[i][:,0], force_bb[i][:,1]/1.17, 'k', label = 'Krylow & Sandercock (1997)')
        plt.plot(time_dt, R/1.17, 'r', label = 'MN-driven model')
        plt.grid()
        plt.xlim((-0.03,2.03))
        plt.ylim((-0.05,1.6))
        
        if i == 0:
            plt.title(u"Max. amp. = \u00B1 0.05 mm", x = 0.8, y = 0.7)
            plt.legend(loc='lower right')
            
        elif i == 1:
            plt.title(u"Max. amp. = \u00B1 0.10 mm", x = 0.8, y = 0.7)
            
        elif i == 2:
            plt.title(u"Max. amp. = \u00B1 0.25 mm", x = 0.8, y = 0.7)
            
        elif i == 3:
            plt.title(u"Max. amp. = \u00B1 0.50 mm", x = 0.8, y = 0.7)
            
        elif i == 4:
            plt.title(u"Max. amp. = \u00B1 1.00 mm", x = 0.8, y = 0.7)
            
        elif i == 5:
            path_Theo = "MA_Benchmark_2mm_Hatze_final.csv"
            data_Theo = pd.read_csv(path_Theo, delimiter = ';')
            data_Theo = data_Theo.to_numpy() 
            plt.plot(data_Theo[:,0], data_Theo[:,1], 'g')
            
            # path_mill = "results_damped_equilibrium_trial6.csv"
            # data_mill = pd.read_csv(path_mill, delimiter = ',')
            # data_mill = data_mill.to_numpy() 
            # plt.plot(data_mill[:,0], data_mill[:,1]/1.27, 'b--')
            
            plt.ylabel(r'Muscle Force [$\frac{F}{F_{iso}}$]')
            #plt.title(u"Max. amp. = \u00B1 2.00 mm", x = 0.8, y = 0.8)
        
        
        
    plt.suptitle('Maximal-Activation Biological Benchmark', weight='bold', y=0.9)
    plt.xlabel('Time [s]')
    fig.supylabel('Normalized Force', x=0.07)
    
    os.chdir(cwd)
    

    
    
if Trial == 'BBsub':
    os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation") #max activation BB dir path
    # to be written...
    os.chdir(cwd)


