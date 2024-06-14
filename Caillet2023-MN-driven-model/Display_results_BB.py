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
import os
cwd = os.getcwd()
from pathlib import Path
root = Path(".")
path_to_res = root / "Results"
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#______________________________________________________________________________
#Choose Trial the results of which you wish to plot
Trial = 'BBmax'
# Trial = 'BBsub'
#______________________________________________________________________________
plt.rcParams['figure.dpi'] = 360
figure(figsize=(12, 9))

if Trial == 'BBmax':
    os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\maximalActivation") #max activation BB dir path
    disp_bb = np.genfromtxt('displacement.dat', delimiter='') #BB time & displacement data
    force_bb = []
    for i in range(6):
        force_bb.append(np.genfromtxt('force_trial'+str(i+1)+'.dat', delimiter='')) #list of lists (6 BB time & forces data)
        fig = plt.subplot(6,1,i+1)
        plt.plot(force_bb[i][:,0], force_bb[i][:,1])
        plt.grid()
    plt.suptitle('BB-max.-activation vs. MN-driven model')
    #fig.supxlabel()
    #fig.supylabel()
    os.chdir(cwd)
    
    
if Trial == 'BBsub':
    os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\submaximalActivation") #max activation BB dir path
    # to be written...
    os.chdir(cwd)


