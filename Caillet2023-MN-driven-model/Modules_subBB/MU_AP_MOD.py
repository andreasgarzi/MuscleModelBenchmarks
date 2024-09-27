""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2023
Imperial College London
Department of Civil Engineering
Function necessary to compute the results presented in the manuscript Caillet et al. 'Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy' (2023)
---------

Computation of the MU Action Potentials from input trains of MN Action Potentials
"""

from MU_AP_ODE_MOD import MU_AP_ODE_func

def MU_AP_func(t, y, Matrix_AP): 
    
    syn_delay, sarc_delay, tub_delay =  0.5*10**-3, 3.0*10**-3, 0.5*10**-3  #s
    MAP_delay = syn_delay + 4*sarc_delay + tub_delay 
    #MAP_delay = 0
     
    beta = y[0] 
    dbetadt = y[1]
    DDbetaDDt = MU_AP_ODE_func(t-MAP_delay, Matrix_AP, beta, dbetadt) #Actual 2nd ord. ODE
    return dbetadt, DDbetaDDt
 
