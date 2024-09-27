# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:48:46 2024

@author: z5517249
"""

import numpy as np

def T_strain(Ft):
    
    #John 2013
    eps_0 = 0.055 # strain at max. isometric force in rat soleus (5-6% from Monti et al.2003)
    eps_toe = 0.609*eps_0
    #klin = 1.212/eps_0 #1.712
    klin = 1.212/eps_0 #1.712
    F_toe = 0.33
    k_toe = 2
    
    j = (Ft*np.exp(k_toe)-Ft+F_toe)/F_toe
    
    if Ft > F_toe:
        eps_T = (Ft-klin*eps_toe-F_toe)/klin
    elif Ft > 0 and Ft <= F_toe:
        eps_T = (np.exp(j)*eps_toe)/k_toe 
    else:
        eps_T = 0
  
    return eps_T