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
  
def coef_CA(MU_type, i, f):  #c1, c2, c3
    if MU_type == 'fast' and i == 1 and f == 4: # 16°C
        c_1, c_2, c_3 = 1.3*10.**4,  4.3*10.**5, 1.8  # C2 = decay constant, C3 = pk2pk 0-5
    elif  MU_type == 'fast': # 35°C
        c_1, c_2, c_3 = 2.8*10.**3,  4.3*10.**5, 0.7  # 0.7
       
    if MU_type == 'slow' and i == 0 and f == 4: # 16°C
       c_1, c_2, c_3 = 8.5*10.**3,  1.8*10.**5, 0.4  # 1.8*10.**5
    elif MU_type == 'slow': 
       c_1, c_2, c_3 = 8*10.**3, 1.8*10.**5, 0.6 # 23°C
       
    return c_1, c_2, c_3

def Ca_l_amplitude_func(l): # F1 function
    p = [-0.4688, 3.7127, -11.0323, 14.063, -5.4709]
    return (l**4)*p[0] + (l**3)*p[1] + (l**2)*p[2] + l*p[3] + p[4]

def Ca_l_width_func(l): #F2 function
    p = [0.3783, -0.8320, 1.1885]
    return (l**2)*p[0] + l*p[1] + p[2]

def MU_free_Ca_ODE_func(t, l_M_norm, MU_type, beta, gamma,  dgammadt, i, f):
    
    c_1, c_2, c_3 = coef_CA(MU_type, i, f) 

    amp=Ca_l_amplitude_func(l_M_norm) #impact of l_M_norm on Ca amplitude
    width=Ca_l_width_func(l_M_norm) #impact of l_M_norm on Ca half-width
    
    if MU_type == 'slow':
        DDgammaDDt = c_3*beta - 1/amp*(c_1*dgammadt + width*c_2*gamma) #Actual 2nd ord. ODE
    elif MU_type == 'fast':
        DDgammaDDt = c_3*beta - 1/amp*(c_1*dgammadt + width*c_2*gamma*(gamma*10**5))
    return DDgammaDDt

def MU_free_Ca_func(t, y, MU_AP_train, l_M, MU_type, Matrix_AP, i, f): 

    #CA_delay = 2.1*10**-3
    CA_delay = 0  
    gamma = y[2] 
    dgammadt = y[3]
    beta = MU_AP_train  #from previously solved ODE for MUAP
    
    DDgammaDDt = MU_free_Ca_ODE_func(t-CA_delay, l_M, MU_type, beta, gamma, dgammadt, i, f)    
    return dgammadt, DDgammaDDt