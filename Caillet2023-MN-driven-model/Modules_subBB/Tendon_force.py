"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue May 21 20:46:53 2024
___________________________________

Tendon force equation given the strain (John 2013)
"""
import numpy as np

def T_force(eps):
    
    #John 2013
    #eps_0 = 0.055 # strain at max. isometric force in rat soleus (5-6% from Monti et al.2003)
    eps_0 = 0.033
    eps_toe = 0.609*eps_0
    #klin = 1.212/eps_0 #cat soleus
    klin = 1.212/eps_0 
    F_toe = 0.33
    k_toe = 3
        
    if eps > eps_toe:
        f_T = 0.001*(1+eps)+(klin*(eps-eps_toe)+F_toe)
    elif eps > 0 and eps <= eps_toe:
        f_T = 0.001*(1+eps)+(F_toe*((np.exp(k_toe*eps/eps_toe)-1)/(np.exp(k_toe)-1)))
    else:
        f_T = 0.001*(1+eps)   

    return f_T