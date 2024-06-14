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

def MU_bound_calcium_func(t, y, free_Ca_concentration, l_M, MU_type, Matrix_AP): 

    if free_Ca_concentration < 0:   # avoid negative values
        free_Ca_concentration = 0

    if MU_type == 'fast':
        T0 = 3.8*10**-4 
        k1=0.1*10**13 
        k2 = 41 
        if l_M <= 0.7:
            f3 = 0.33
            f5 = 0.75
            f4 = 1.09
        elif l_M <= 0.97:
            f3 = 0.33 + 2.47*(l_M - 0.75)
            f5 = 0.75 + 0.94*(l_M - 0.75)
            f4 = 1.09 - 0.32*(l_M - 0.75)
        elif l_M >= 0.97 and l_M <= 1.02:
            f3 = 1
            f5 = 1
            f4 = 1
        elif l_M >= 1.02 and l_M <= 1.6:
            f3 = 8.9 - 11.73*l_M + 3.9*(l_M**2)
            f5 = 0.9 + 0.84*l_M - 0.75*(l_M**2)
            f4 = -4.4 + 8.24*l_M - 2.84*(l_M**2)
        elif l_M > 1.6:
            f3 = 0.125
            f5 = 0.34
            f4 = 1.46
        
    elif MU_type == 'slow':
        T0 = 17*10**-5 
        k1=0.6*10**13 
        k2 = 21 
        if l_M <= 0.7:
            f3 = 0.75
            f5 = 0.66
            f4 = 1.06
        elif l_M <= 0.97:
            f3 = 0.75 + 0.93*(l_M - 0.75)
            f5 = 0.66 + 1.27*(l_M - 0.75)
            f4 = 1.06 - 0.23*(l_M - 0.75)
        elif l_M >= 0.97 and l_M <= 1.02:
            f3 = 1
            f5 = 1
            f4 = 1
        elif l_M >= 1.02 and l_M <= 1.6:
            f3 = 1.2 - 0.01*l_M - 0.22*(l_M**2)
            f5 = 4.2 - 4.28*l_M - 1.11*(l_M**2)
            f4 = 4.5 - 5.09*l_M + 1.62*(l_M**2)
        elif l_M > 1.6:
            f3 = 0.67
            f5 = 0.21
            f4 = 0.52        
    
    gamma = free_Ca_concentration   # from previously solved ODE for free [Ca++]  
    delta = y[4]
    #ddeltadt = (k1/f3)*(T0/f4 - delta)*(gamma**2) - (k2/f5)*delta # modified Wexler to account for non-linear length dependency
    ddeltadt = k1*T0*gamma**2-(k1*gamma**2+k2)*delta #║Wexler 1997. It is quite the same but not entirely. 
    return ddeltadt
    
