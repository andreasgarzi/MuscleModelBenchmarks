""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2023
Imperial College London
Department of Civil Engineering
Function necessary to compute the results presented in the manuscript Caillet et al. 'Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy' (2023)
---------

ODE that defines the dynamics of CaTn concentration in the MUs (in Mols), from an input concentration of free Calcium (in Mols)

"""

def active_state_1(t, y):     

    Ca, a1 = y[2], y[4]

    if Ca < 0:
        Ca = 0 
    
    Ca = Ca*2e5 # normalized to its maximum value
    
    if Ca > a1:
        K = 30.79 
        dadt1 = (Ca**2.5 - a1)*(a1 + K)*(1 - a1) # ascending phase limited to 1
    else:
        K = 60.8329 
        dadt1 = (Ca - a1)*(a1 + K) # decay following a non-normalized trend

    return dadt1

    
