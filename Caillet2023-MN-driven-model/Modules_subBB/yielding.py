# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:05:46 2024

@author: z5517249
"""

import numpy as np

def Yield(dt, t, y, V):
    
    cy, Vy, Ty = 0.35, 0.1, 0.2
    Y = y[6]
    
    dY = (1 - cy*(1 - np.exp((-np.abs(V))/Vy)) - Y)/Ty
    
    return dY