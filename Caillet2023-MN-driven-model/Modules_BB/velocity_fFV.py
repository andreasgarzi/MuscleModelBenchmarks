"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBME
Created on Wed May 22 11:28:32 2024
____________________________________

Description: force-velocity relationship
"""


def velo_fFV(t, y, CE_force, FL_force, act, l_M, MU_type):
    
    FL_force = FL_force*act
    fmax = 1.33 #Hatze
    
    # Defining fFV involved parameters
    if MU_type == 'slow':
        kMU = 0.8
    elif MU_type == 'fast':
        kMU = 1
        
    fv = 0.8 + 0.2*act
    
    if l_M < 1:
        g = FL_force
    else:
        g = 1
    #g = 1
        
    if MU_type == 'slow':
        af = 1.2
    elif MU_type == 'fast':
        af = 1.4
        
    b = (fmax - 0.5)/(2 + 2/af)
    K = kMU*fv*g
    
    #fFV relationship inverted
    if CE_force >= 1:
        vel = b*((CE_force-1)/(fmax-CE_force)) # rat soleus (Krylow, Sandercock)
        vel = vel*K
    else:
        vel = (CE_force-1)/(CE_force/(af*K))
        
    
    return vel