"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Mon May 27 09:02:41 2024
___________________________________


"""
import numpy as np

def penn_ang(l_MT, l_M, l_T, l_M_0, alpha0):

    alpha = alpha0
    w = l_M_0*np.sin(alpha0)
    cosalpha = 1/(np.sqrt(1 + (w/(l_MT-l_T))**2))
    alpha = np.arccos(cosalpha)
        
    if alpha > 1.4706289:  # np.arccos(0.1), 84.2608 degrees
        alpha = 1.4706289
    
    return alpha