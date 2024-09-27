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

def MU_active_state_func(t, Ca, act, tup, tdown, a, b):    
    
 # ARNAULT ODE 
    
    # coefs = [2*10**5 , 0.026, 260] #works best against F-F steady-state curves. TTP = 60ms. Half Relaxation Time (HRT) = 75ms 
    # # Need for a tuning to match the MU twitch
    # #coefs = [d1_tuned[int(t/dt)] , 0.041, d3_tuned[int(t/dt)]] #  TTP = 35ms HRT = 43ms in rat soleus (Malak 2024)
    
    # a = y[5]
    # CaTn = CaTn_concentration    #from previously solved ODE ([Ca-Tn])
    
    # d1, d2, d3 = coefs
    # dadt = (d1*CaTn - a/(d2+d3*CaTn))*(1-a)
    
    # return dadt

#______________________________________________________________________________
# Winters and Thelen ODE 
 
    # #Ca = Ca/(5*10**-6)
    # Ca = Ca/(150*10**-6)
    # #tup = 0.8
    # #tup = 0.01
    # #tdown = 0.04
    
    # amin = 1*10**-9
    # ac = (act-amin)/(1-amin)
    
    # if Ca > ac:
    #     tau = tup*(a + b*ac)  # for 20 Hz with Camax = 1*10**-6
    # else:
    #     tau = tdown/(a + b*ac)
    
    # dadt = ((Ca - ac)/tau)*(1-act)
    
    # return dadt

#______________________________________________________________________________
# Hussein 2022 ODE

    if Ca < 0:
        Ca = 0
        
    Ca = Ca/(6*10**-6)
    #Ca = Ca/(50*10**-7)
    
    #k1, k2 = 22.5, 19.2
    #k1, k2 = 22.5, 19.2
    dadt = -(b*act - a*Ca)*(1 - act)
    
    return dadt