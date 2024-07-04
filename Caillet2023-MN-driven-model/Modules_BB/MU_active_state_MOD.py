""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2023
Imperial College London
Department of Civil Engineering
Function necessary to compute the results presented in the manuscript Caillet et al. 'Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy' (2023)
---------

ODE that defines the dynamics of MU active state, from an input concentration of CaTn (in Mols)

"""

#def MU_active_state_func(t, y, CaTn_concentration): 
def MU_active_state_func(t, y, Ca, MU_type): 
    
    #coefs = [1.00*10**5 , 0.021,  260] #works best against F-F steady-state curves. TTP = 60ms. Half Relaxation Time (HRT) = 75ms 
    # Need for a tuning to match the MU twitch
    #coefs = [1.5*10**5 , 0.015, 130] #  TTP = 35ms HRT = 43ms in rat soleus (Malak 2024)
    
    # a = y[5]
    # CaTn = CaTn_concentration     #from previously solved ODE ([Ca-Tn])
    # d1, d2, d3 = coefs
    # dadt = d1*CaTn - a/(d2+d3*CaTn)
    # return dadt

    if Ca < 0:
        Ca = 0

    if MU_type == 'slow':
        Ca_max = 4.5*10**-6
    elif MU_type == 'fast':
        Ca_max = 3.5*10**-6
        
    Ca = Ca/Ca_max
    
    a = y[4]
    k3, k4 = 22.5, 19.2
    dadt = -(k4*a - k3*Ca)*(1 - a)
    
    return dadt