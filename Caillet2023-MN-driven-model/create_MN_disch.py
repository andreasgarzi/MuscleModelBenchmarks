"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue Jun  4 10:09:55 2024
___________________________________

Create discharge times for N MNs to reach active state = 1 (50 Hz each one)
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

fd = 50 # 50 Hz d.r.
t_end = 2 # total seconds
dt = 0.0001 
T = 1/fd
Nr = 35

disch_t = np.empty((Nr, int(t_end/T)), dtype=float)
disch = np.arange(0,2,T)

for i in range(35):
    disch_t[i] = disch


