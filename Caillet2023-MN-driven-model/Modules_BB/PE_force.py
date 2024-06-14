"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue May 21 21:02:25 2024
___________________________________


"""

import numpy as np

def PEE_force(l_M_0):
    
    k1, k2 = 5, 0.6
    
    if l_M_0 < 1:
        normalized_PE_force_list = 0
    elif l_M_0 >= 1:
        normalized_PE_force_list = (np.exp((k1*(l_M_0-1))/k2)-1)/(np.exp(k1)-1)
        
    return normalized_PE_force_list