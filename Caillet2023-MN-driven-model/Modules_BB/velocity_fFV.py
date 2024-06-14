"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBME
Created on Wed May 22 11:28:32 2024
____________________________________

Description: force-velocity relationship
"""

def velo_fFV(t, y, CE_force, FL_force, act, l_M, MU_type):
    
    # Defining fFV involved parameters
    if MU_type == 'slow':
        kMU = 0.5
    elif MU_type == 'fast':
        kMU = 1
        
    fv = 0.8 + 0.2*act
    
    if l_M < 1:
        g = FL_force
    else:
        g = 1
        
    if MU_type == 'slow':
        af = 0.2
    elif MU_type == 'fast':
        af = 0.4
        
    b = (1.1 - 1)/(2 + 2/af)
    K = kMU*fv*g
    
    #fFV relationship inverted
    if CE_force < 0:
        vel = CE_force*(1+1/(af*K))-1
    elif CE_force >= 0 and CE_force < 1:
        vel = (CE_force-1)/(CE_force/af+1)
    elif CE_force >= 1:
        #vel = b*((CE_force-1)/(1.4-CE_force)) # human TA
         vel = b*((CE_force-1)/(1.716-CE_force)) # rat soleus (Krylow, Sandercock)
         
    vel = vel*K
    
    return vel