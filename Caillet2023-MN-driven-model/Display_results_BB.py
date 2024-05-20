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
path_to_res = root / "Results_BB" 
import matplotlib.pyplot as plt

#______________________________________________________________________________
#Choose Trial the results of which you wish to plot
Trial = 'BBmax'
# Trial = 'BBsub'
#______________________________________________________________________________

if Trial == 'BBmax':
    os.chdir("C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\biologicalBenchmark\\maximalActivation")
    disp_bb = np.genfromtxt('displacement.dat', delimiter='')
    force_bb = []
    for i in range(5):
        force_bb.append(np.genfromtxt('force_trial'+str(i+1)+'.dat', delimiter=''))
        
    os.chdir(cwd)
    


