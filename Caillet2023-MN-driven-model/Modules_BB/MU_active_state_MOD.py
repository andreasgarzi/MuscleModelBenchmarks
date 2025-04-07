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

def MU_active_state_func(t, Ca, act):

    Ca = Ca*2e5
    amin = 1*10**-9
    ac = (act - amin)/(1 - amin)
    
    k3, k4 = 15.5, 22.2  
    
    if Ca > ac:
        dadt = -(k4*ac - k3*Ca**2.5)*(1 - ac)
    else:
        dadt = -(k4*ac - k3*Ca)
    
    return dadt