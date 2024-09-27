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

def MU_active_state_func(t, Ca, a, Ca_max, t_up, t_down, dt): 
    
    # #coefs = [1.14*10**5 , 0.026, 260] #works best against F-F steady-state curves. TTP = 60ms. Half Relaxation Time (HRT) = 75ms 
    # # Need for a tuning to match the MU twitch
    # coefs = [d1_tuned[int(t/dt)] , d2_tuned[int(t/dt)], 260] #  TTP = 35ms HRT = 43ms in rat soleus (Malak 2024)
    
    # a = y[5]
    # CaTn = CaTn_concentration    #from previously solved ODE ([Ca-Tn])
    
    # d1, d2, d3 = coefs
    # dadt = (d1*CaTn - a/(d2+d3*CaTn))*(1-a)
    
    # return dadt

    Ca = Ca/Ca_max[int(t/dt)]
    t_up = t_up[int(t/dt)]
    t_down = t_down[int(t/dt)]
    
    amin = 1*10**-9
    ac = (a-amin)/(1-amin)
    
    # if Ca > ac:
    #     tau = t_up*(0.5 + 0.5*ac)  # for 20 Hz with Camax = 1*10**-6
    #     #tau = t_up*(0.5 + 0.5*ac)
    # else:
    #     tau = t_down/(1 + 0.5*ac)
    
    if Ca > ac:
        tau = t_up*(0.5 + 0.5*ac)  # for 20 Hz with Camax = 1*10**-6
        #tau = t_up*(0.5 + 0.5*ac)
    else:
        tau = t_down/(1 + 0.5*ac)
    
    dadt = ((Ca - ac)/tau)*(1-a)
    
    return dadt



    # if Ca < 0:
    #     Ca = 0

    # if MU_type == 'slow':
    #     Ca_max = 9.5*10**-6 
    #     #Ca_max = 6.8*10**-6
    #     #Ca_max = 1.5*10**-5
        
    # elif MU_type == 'fast':
    #     Ca_max = 3.5*10**-6
        
    # Ca = Ca/Ca_max
    
    # a = y[4]
    # #k1, k2 = 22.5, 19.2
    # k1, k2 = 10, 5

    # dadt = -((a)/(Ca*k1 + 0.02) - k2*Ca)*(1 - a)  # Modified Hussein et al. 2022 ODE
    #dadt = -(k2*a - k1*Ca)*(1 - a)
    
    #return dadt

