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

def CaTn(t, y, MU_type):     

    Ca, CaTn = y[2], y[4]
        
    if Ca < 0:
        Ca = 0 
        
    if MU_type == 'slow':
        k1, k2, T0 = 2e13, 12, 8.7e-5
            
    elif MU_type == 'fast':
        k1, k2, T0 = 5e12, 16, 2.4e-4
        
    dCaTndt = k1*(T0-CaTn)*Ca**2 - k2*CaTn

    return dCaTndt

    
