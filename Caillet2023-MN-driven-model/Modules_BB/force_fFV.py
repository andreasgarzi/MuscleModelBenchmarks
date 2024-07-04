"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Sat Jun  1 15:21:15 2024
___________________________________

Force-velocity relationship.
"""

def f_fFV(v_M, FL_force, act, l_M, MU_type):
    
    FL_force = FL_force*act
    fmax = 1.33 #Hatze
    
    # Defining fFV involved parameters
    if MU_type == 'slow':
        kMU = 1
    elif MU_type == 'fast':
        kMU = 1
        
    fv = 0.8 + 0.6*act
    
    # if l_M < 1:
    #     g = FL_force
    # else:
    #     g = 1
    g = 1
        
    if MU_type == 'slow':
        af = 0.9
    elif MU_type == 'fast':
        af = 0.5
        
    b = (fmax - 1)/(2 + 2/af)
    K = kMU*fv*g
    
    if v_M < -1:
        fv = (1+v_M/K)/(1+(1/(af*K)))
    elif v_M >= -1 and v_M < 0:
        fv = (1+v_M/K)/(1-(v_M/(af*K)))
    elif v_M >= 0:
        fv = (1 + fmax*(v_M/(K*b)))/(1+v_M/(K*b))
    
    return fv
        
       
       
       
    