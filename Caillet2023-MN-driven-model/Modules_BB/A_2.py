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


def active_state_2(t, y):

    a1, a2 = y[4], y[5]
    
    if a1 > a2:
        K = 15.79 
    else:
        K = 60.8329 
        
    dadt2 = (a1 - a2)*(a2 + K)
    
    return dadt2


