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

def MU_active_state_func(t, Ca, act, Y):

    #Ca = Ca/(5.5877*10**-6)
    #Ca = Ca/(5*10**-5)
    #Ca = Ca/(9*10**-6) # cat soleus
    Ca = Ca/(2*10**-5)
    
    amin = 1*10**-9
    ac = (act - amin)/(1 - amin)
    
    n = 1  # species & exp conditions dependent parameter n + 2.5 (cat soleus)
    
    if Ca > ac:
        K = 15.79  # 15.79
        dadt = (Ca**n - ac)*(ac + K)*(1 - ac)
    else:
        K = 60.8329 # 60.8329
        dadt = (Ca - ac)*(ac + K) # decay following a non-normalized trend
    
    return dadt


