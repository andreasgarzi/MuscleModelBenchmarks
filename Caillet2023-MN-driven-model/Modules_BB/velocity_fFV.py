"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBME
Created on Wed May 22 11:28:32 2024
____________________________________

Description: force-velocity relationship
"""

def velo_fFV(t, y, CE_force, FL_force, act, l_M, MU_type, vmax, af_par, fmax_par):
    
    #fmax = 1.33
    fmax = fmax_par
    
    # Defining fFV involved parameters
    if MU_type == 'slow':
        kMU = 0.5
    elif MU_type == 'fast':
        kMU = 1

    #fv = 0.2 + 0.8*act*FL_force
    
    fv = 0.8 + 0.2*act
    
    if l_M < 1:
        g = FL_force
    else:
        g = 1
    #g = 1
        
    if MU_type == 'slow':
        af = af_par # the lower the lower the inferior limit
    elif MU_type == 'fast':
        af = 0.8
        
    b = (fmax - 1)/(2 + 2/af)
    K = (kMU*fv*g)
    
    #fFV relationship inverted
    if CE_force >= 1:
        vel = b*((CE_force-1)/(fmax-CE_force)) # rat soleus (Krylow, Sandercock)
        vel = vel*K
    else:
        vel = (CE_force-1)/(CE_force/(af*K))
    
    return vel*vmax