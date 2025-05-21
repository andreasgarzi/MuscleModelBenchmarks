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

def active_state(y, MU_type):

    # " Modified Hussein"
    # if MU_type == 'slow':
    #     Ca, a = y[2]*2.2e5, y[5]
    #     k1, k2 = 19.5, 22.2
        
    # elif MU_type == 'fast':
    #     Ca, a = y[2]*2.2e5, y[5]
    #     k1, k2 = 19.5, 22.2
    
    # # Ca, a = y[2]*2e5, y[4]      
    # # k1, k2 = 19.5, 21.2
    # #k1, k2 = 22.5, 23.2
            
    # if Ca > a:
    #     dadt = -(k2*a - k1*Ca)*(1 - a)
    # else:
    #     dadt = -(k2*a - k1*Ca) 

    
    #%%
    
    # Ca, a = y[2]*1e5, y[4]
    
    # if Ca > a:
    #     K = 20
    #     dadt = (Ca - a)*(1 + K)*(1-a)
    # else:
    #     K = 60
    #     dadt = (Ca - a)*(1 + K)
    
    #%%
    "Wexler 1997"
    
    # Ca, CaTn, a = y[2]*2.5e5, y[4], y[5]
    
    # tau1, tau2 = 0.06423, 234.69

    # if Ca > a:
    #     dadt = (3e4*CaTn - a/(tau1 + tau2*CaTn))*(1-a)
    # else:
    #     dadt = (3e4*CaTn - a/(tau1 + tau2*CaTn))
            
    #%%
    "Thelen 2003"
    
    # Ca, CaTn, a = y[2]*2e5, y[4], y[5]
    # amin = 1e-9
    # a_c = (a-amin)/(1-amin)
    
    # if Ca > a_c:
    #     tau = 0.10*(0.5 + 1.5*a_c)
    #     dadt = ((CaTn*3e3-a_c)/tau)*(1-a_c)
    # else:
    #     tau = 0.40/(0.5 + 1.5*a_c)
    #     dadt = (CaTn*3e3-a_c)/tau
    
    #%%
    "Wexler 1997"
    
    if MU_type == 'slow':
        CaTn, Ca, a = y[4], y[2], y[5]
        d1, tau1, tau2 = 1e5, 0.014, 1e5
        
        if Ca*6e5 > a:
            dadt = (CaTn*d1 - a/(tau1 + tau2*CaTn))*(1-a)
        else:
            dadt = (CaTn*d1 - a/(tau1 + tau2*CaTn))    
          
    elif MU_type == 'fast':
        Ca, a = y[2]*2.5e6, y[4]
        tau1, tau2 = 0.014, 0.04
        
        if Ca > a:
            dadt = (Ca - a/(tau1 + tau2*Ca))*(1-a)
        else:
            dadt = (Ca - a/(tau1 + tau2*Ca)) 
        
    # tau1, tau2 = 0.01423, 434.69
    # d1, tau1, tau2 = 2.5e6, 0.026, 270
    
    # dadt = (Ca - a/(tau1 + tau2*Ca*1e-5))
    
    return dadt


