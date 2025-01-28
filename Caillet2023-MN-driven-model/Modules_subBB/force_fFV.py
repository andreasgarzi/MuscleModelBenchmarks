"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Sat Jun  1 15:21:15 2024
___________________________________

Force-velocity relationship.
"""

def f_fFV(v_M, FL_force, act, l_M, MU_type):
    
    fmax = 1.2786
    #fmax = 1.4
    
    # Defining fFV involved parameters
    if MU_type == 'slow':
        kMU = 0.5
    elif MU_type == 'fast':
        kMU = 1

    #fv = 1
    
    fv = 0.8 + 0.2*act
    
    if l_M < 1:
        g = FL_force
    else:
        g = 1
    #g = 1
        
    if MU_type == 'slow':
        af = 0.17 # the lower the lower the inferior limit
    elif MU_type == 'fast':
        af = 0.8
        
    b = (fmax - 1)/(2 + 2/af)
    K = kMU*fv*g
    
    if v_M < -1:
        fvel = 1/(1 - (v_M/(af*K)))
        #fv = 0
    if v_M >= -1 and v_M < 0:
        fvel = 1/(1 - (v_M/(af*K)))
    elif v_M >= 0:
        fvel = (1 + fmax*(v_M/(K*b)))/(1 + v_M/(K*b))
    
    return fvel
        
       
       
       
    