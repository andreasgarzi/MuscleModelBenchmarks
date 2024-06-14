""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2023
Imperial College London
Department of Civil Engineering
Function necessary to compute the results presented in the manuscript Caillet et al. 'Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy' (2023)
---------

Computation of the free Calcium concentration (in Mols) in the MUs from an input MU action potential (in V)
"""

from MU_free_Ca_ODE_MOD import MU_free_Ca_ODE_func

def MU_free_Ca_func(t, y, MU_AP_train, l_M, MU_type, Matrix_AP): 

    CA_delay = 2.1*10**-3
    #CA_delay = 0
        
    gamma = y[2] 
    dgammadt = y[3]
    #beta = MU_AP_train[max(int((t-CA_delay)/0.0001), 0)] #from previously solved ODE for MUAP
    beta = MU_AP_train  #from previously solved ODE for MUAP
    DDgammaDDt = MU_free_Ca_ODE_func(t-CA_delay, l_M, MU_type, beta, gamma, dgammadt)    
    return dgammadt, DDgammaDDt
  