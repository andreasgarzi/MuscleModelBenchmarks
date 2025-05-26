"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Created on Tue Jan 28 15:33:16 2025
___________________________________

MN-driven model based on Caillet et al. 2023 for simulating isometric and dynamic muscle contractions.

"""

import os
import scipy as sp
import numpy as np
import tkinter as tk
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from tkinter import simpledialog
import matplotlib.pyplot as plt
cwd = os.getcwd()

class MN_driven_model():

    def __init__(self, parameters, states, distimes):

        #######################################################################
        " Check and assign parameters "

        if parameters is None:
            raise ValueError('No parameters set!')
        elif not isinstance(parameters, dict):
            raise ValueError('Parameters are not in dict format!')
        else:
            self.P = parameters  # set parameters
        
        # Assign attributes from input dictionary
        self.time = self.P['time']
        
        if self.P['pool'] != 'y' and self.P['pool'] != 'n':
            raise ValueError("Expected either 'y' or 'n' for pool parameter")
        else:
            self.pool = self.P['pool']
            
        self.dt = self.P['dt']
        
        if self.P['muscle'] != 'TA' and self.P['muscle'] != 'GM' and self.P['muscle'] != 'SOL' and self.P['muscle'] != 'CF':
            raise ValueError("Expected either 'TA', 'GM', 'CF', 'SOL' for muscle parameter")
        else:
            self.muscle = self.P['muscle']
        
        if self.P['species'] != 'animal' and self.P['species'] != 'human':
            raise ValueError("Expected either 'human', or 'animal' for species parameter")
        else:
            self.species = self.P['species']
            
        if self.P['spread'] != 'evenly' and self.P['spread'] != 'identified':
            raise ValueError("Expected either 'evenly', or 'identified' for spread parameter")
        else:
            self.spread = self.P['spread']
        
        if self.P['yielding'] != 'y' and self.P['yielding'] != 'n':
            raise ValueError("Expected either 'y', or 'n' for yielding parameter")
        else:
            self.y = self.P['yielding']
           
        if self.P['sag'] != 'y' and self.P['sag'] != 'n':
            raise ValueError("Expected either 'y', or 'n' for yielding parameter")
        else:
            self.s = self.P['sag']
        
        
        self.MVC = self.P['MVC']
        self.Nr = self.P['Nr'] # Number of identified motor units
        self.MN_pool = self.P['MN_pool'] # Number of theoretical MNs in the muscle pool
        self.vmax = self.P['vmax']
        self.alpha_0 = self.P['alpha_0']
        self.l_MT = self.P['l_MT']
        self.l_M_opt = self.P['l_M_opt']
        self.l_T_slack = self.P['l_T_slack']
        self.distimes = distimes
        
        # self.path = self.P['path']    # only for experimental inputs...
        # self.l_MT = np.ones(len(self.time) + 1)*self.P['l_MT_0']  # Set MT lengths
        
        # self.Ca_max = self.P['Ca_max']  # only for optimization procedures...
        # self.c_1 = self.P['c_1'] 
        # self.c_2 = self.P['c_2'] 
        # self.c_3 = self.P['c_3'] 
        # self.vmax = self.P['vmax']
        # self.vmax = self.P['vmax']
    
        #######################################################################    
        " Check and assign states "
    
        if states is None:
            raise ValueError('No states set!')
        elif not isinstance(states, dict):
            raise ValueError('States are not in dict format!')
        else:
            self.S = states  # set parameters
        
        self.MUAP_0 = self.S['MUAP_0']
        self.Ca_0 = self.S['Ca_0']
        self.CaTn_0 = self.S['CaTn_0']
        self.act_0 = self.S['act_0']
        self.l_M_0 = self.S['l_M_0']
        self.y_0 = self.S['y_0']
        self.s_0 = self.S['s_0']
        
        #######################################################################
        " Check and assign discharge times input "
        
        if distimes is None:
            raise ValueError("Discharge times must be provided")
        else:
            self.distimes = distimes  # Use provided discharges
        
        self.fs = np.mean(1/np.diff(self.distimes))
        
        
        # Sort MUs according to recruitment   
        # sorted_indices = np.argsort([self.distimes[0, i][0, 0] for i in range(np.size(self.distimes, axis=1))])
        # self.distimes = self.distimes[:, sorted_indices]
            
        # Extract Fth recruitment in % of MVC (first discharge indeces)
        # self.F_thr = np.empty((self.Nr))
        # for f in range(np.size(self.distimes, axis=1)):
        #     self.F_thr[f] = self.path[0, self.distimes[0, f][0, 0]]*100 
        
        #######################################################################
        
        # Preallocate output arrays 
        self.initialize_arrays()
        

    def initialize_arrays(self):
        
        self.alpha = np.empty((self.Nr, len(self.time) + 1))
        self.l_T = np.empty((self.Nr, len(self.time)))
        self.eps_T = np.empty((self.Nr, len(self.time)))
        self.f_SE = np.empty((self.Nr, len(self.time)))
        self.f_CE = np.empty((self.Nr, len(self.time)))
        self.f_PE = np.empty((self.Nr, len(self.time)))
        self.f_FL = np.empty((self.Nr, len(self.time)))
        self.vel = np.empty((self.Nr, len(self.time)))
        self.f_M = np.empty((self.Nr, len(self.time)))
        self.l_M = np.empty((self.Nr, len(self.time)))
        self.MUAP = np.empty((self.Nr, len(self.time)))
        self.active_state = np.empty((self.Nr, len(self.time)))
        self.free_Ca = np.empty((self.Nr, len(self.time)))
        self.CaTn = np.empty((self.Nr, len(self.time)))
        self.yielding = np.empty((self.Nr, len(self.time)))
        self.sag = np.empty((self.Nr, len(self.time)))


#%%
    " MU type (so far for TA and GM) "    
    
    def MU_type_id_func(self, i):
    
        if self.muscle == 'TA' and self.species == 'human':
            if i/self.Nr < 0.9:
                MU_type='slow' # TA
            else: 
                MU_type='fast'
        elif self.muscle == 'GM' and self.species == 'human':
            if i/self.Nr < 0.5:
                MU_type='slow' # GM
            else: 
                MU_type='fast'
        elif self.muscle == 'SOL' and self.species == 'animal':
            MU_type = 'slow' # GM
        elif (self.muscle == 'CF' or self.muscle == 'GM') and self.species == 'animal':
            MU_type = 'fast' # CF
    
        return MU_type
    
    
    " Distribution of force with MU recruitment (TA) "
    
    def Fth_distrib_func(self, MN):  #Obtained from processed literature data, typical muscle-specific distributions of MU force recruitment thresholds (%MVC)
        if self.muscle == 'TA' or self.muscle == 'SOL':
            return 0.5052*(58.1*MN/self.MN_pool+120**((MN/self.MN_pool)**1.83)) 
        elif self.muscle == 'GM':
            return 0.6562*(46.7*MN/self.MN_pool+90**((MN/self.MN_pool)**1.79))

    def F0MU_norm_distrib_func(self, MN): 
        return 7.86*10**-4*(3.0*MN/self.MN_pool+8.20**((MN/self.MN_pool)**5.29))    
        
    def Ftw_norm_distrib_func(self, MN): 
        return 6.07*(4.52*MN/self.MN_pool+11.96**((MN/self.MN_pool)**4.66))  
    
    def F0MU_distrib_func(self):
        
        MU_list_identified = np.arange(1, self.Nr+1, 1) #list of identified MUs = 1:1:Nr
        MU_pool_list = np.arange(1, self.MN_pool+1, 1) #list of all TA MUs = 1:1:N(400)
        
        #### Let's scale the normalized distribution of MU F0MU (from literature) to Newtons
        # Let's compute the sum of the normalized F0MU(i)
        cumulative_all_MUs = sum(self.F0MU_norm_distrib_func(MU_pool_list))
        # Obtaining, using the subject-specific F0M, the N-% relationship
        scale_factor = self.MVC/cumulative_all_MUs

        # and scaling the normalized distribution of F0MU 
        F0MU_distribution_complete_MU_pool = self.F0MU_norm_distrib_func(MU_pool_list) * scale_factor
    
        if self.pool == 'y':
            F0MU_distribution = np.empty(self.Nr)
            last_recruited_MU = np.argwhere(self.Fth_distrib_func(MU_pool_list) < self.MVC)[-1][0]+1 #finding it with MVC value only. 'Blind' approach
        
            if self.spread == 'evenly': #easy approach: assuming the Nr identified MUs are evenly spread across the MU pool
                MU_list_identified = MU_list_identified * last_recruited_MU//self.Nr #Evenly spreading the Nr MUs across the MU pool
            elif self.spread == 'identified': #the identified MUs are here located into the MU pool according to their experimental Fth
                Real_MN_pop = np.zeros(self.Nr)
            
                for i in range (self.Nr):
                    Real_MN_pop[i] = (np.abs(self.Fth_distrib_func(MU_pool_list) - self.F_thr[i])).argmin() #looking for the closest match between typical and experimental %MVC recruitment threshold
             
                Real_MN_pop = Real_MN_pop.astype(int)
                MU_list_identified = Real_MN_pop # is the list of indices of location of each MU within the pool (from the Fth distribution with the MN_distribution_MOD.py function)       

            # Here, each of the Nr identified MUs is consisdered to be representative of a fraction of the MUs of the MU pool, and of their summed F0Mu
            # Basically, you are summing all F0MU in a average symmetric contour around the ith MU within the F0MU distribution (compensating for two-times-counted MUs, due to not always ascending indices of MUs within the whole pool)
            for i in range(0, len(MU_list_identified)):
                if i == 0 : 
                    index_min = 0
                else: 
                    index_min = int(MU_list_identified[i-1] + abs(MU_list_identified[i]-MU_list_identified[i-1])/2) +1
                if i == len(MU_list_identified)-1 : 
                    index_max = last_recruited_MU +1
                else: 
                    index_max = int(MU_list_identified[i] + abs(MU_list_identified[i+1]-MU_list_identified[i])/2)
           
                if index_max<index_min:   #because of this, some MUs may be counted two times. In such case, the sum of F0Ms must be scaled back to true value (see below)
                    index_max = index_min 
            
                F0MU_distribution[i] = sum(F0MU_distribution_complete_MU_pool[index_min : index_max+1])
        
            F0MU_distribution = F0MU_distribution / sum(F0MU_distribution) * sum(F0MU_distribution_complete_MU_pool[0:last_recruited_MU]) #in case some MUs have been counted twice, you normalize for the "affected cumulative value" and un-normalize back with respect to the "correct" cumulative value
        
            F0MU_distribution = np.reshape(F0MU_distribution, (self.Nr,1))
            
        elif self.pool == 'n':
            F0MU_distribution = np.reshape(F0MU_distribution_complete_MU_pool, (self.Nr,1))
              
        return F0MU_distribution

    
    " Tendon force given tendon strain (John 2013) "

    def T_force(self, eps):
        
        if self.species == 'animal':
            eps_0 = 0.06 # strain at max. isometric force in rat soleus (5-6% from Monti et al.2003)
            klin = 1.212/eps_0 
        elif self.species == 'human':
            eps_0 = 0.033
            klin = 1.712/eps_0 
            
        eps_toe = 0.609*eps_0
        F_toe = 0.33
        k_toe = 3
            
        if eps > eps_toe:
            f_T = 0.001*(1+eps)+(klin*(eps-eps_toe)+F_toe)
        elif eps > 0 and eps <= eps_toe:
            f_T = 0.001*(1+eps)+(F_toe*((np.exp(k_toe*eps/eps_toe)-1)/(np.exp(k_toe)-1)))
        else:
            f_T = 0.001*(1+eps)   

        return f_T

    
    " PE 1 & 2 (Virtual Muscle, Brown 1999) "

    def PE_force(self, l_M):
        
        k1, k2 = 5, 0.6  # Thelen 2003
        
        if l_M < 1:
            f_PE = 0
        elif l_M >= 1:
            f_PE = (np.exp((k1*(l_M-1))/k2)-1)/(np.exp(k1)-1)
            
        return f_PE
    
    
    " Calculate pennation angle "

    def penn_ang(self, l_MT, l_M, l_T, l_M_0, alpha0):

        alpha = alpha0
        w = l_M_0*np.sin(alpha0)
        cosalpha = 1/(np.sqrt(1 + (w/(l_MT-l_T))**2))
        alpha = np.arccos(cosalpha)
            
        if alpha > 1.4706289:  # np.arccos(0.1), 84.2608 degrees
            alpha = 1.4706289
        
        return alpha
    
    """ ODEs """
    " Motor unit action potential "
    
    def is_t_a_firing_time_func(self, t_round, Matrix_AP, chosen_precision): 

        t_int = int(t_round/chosen_precision+10**-4) # integer in the scale of the chosen precision
        Matrix_AP_newprecision = (Matrix_AP/chosen_precision).astype(int)  # list of integers in the scale of chosen precision
        if t_int in Matrix_AP_newprecision:
            binary = 1
        else:
            binary = 0
            
        return binary
    
    def MN_AP_func(self, t, Matrix_AP, V_N = 90):
        
        chosen_precision = 10**-3 # has to be higher than the duration of the AP (7*10**-4)
        t_round=int(t/chosen_precision+10**-4)*chosen_precision #input t [s] in chosen precision. +10**-4 is to avoid numerical imprecision that would be problematic with future 'int' function
        firing_time = self.is_t_a_firing_time_func(t_round, Matrix_AP, chosen_precision) #checks whether t_round is (1) or is not (0) a firing time
        
        if firing_time==0: #if not, the membrane potential remains zero
            alpha=0
        
        elif firing_time == 1: #if it is, returns the value of the sine wave calculated at time t
            sin_period=1.4*10**-3 # 2*0.7ms
            
            t_end_sine_wave = t_round + sin_period/2
            alpha=0
            if t<=t_end_sine_wave:
                alpha = np.sin(2*np.pi/sin_period*(t-t_round))   
                
        return V_N * alpha
    
    def MU_AP_ODE_func(self, t, Matrix_AP, beta, dbetadt):
        
        c_4, c_5, c_6 = 2*10**4, 5*10**7, 9*10**7
        DDbetaDDt = c_6*self.MN_AP_func(t, Matrix_AP)-c_5*beta-c_4*dbetadt+10**-50
        
        return  DDbetaDDt
    
    def MU_AP_func(self, t, y, Matrix_AP): 
        
        #syn_delay, sarc_delay, tub_delay =  0.5*10**-3, 3.0*10**-3, 0.5*10**-3  #s
        #MAP_delay = syn_delay + 4*sarc_delay + tub_delay 
        MAP_delay = 0
         
        beta = y[0] 
        dbetadt = y[1]
        DDbetaDDt = self.MU_AP_ODE_func(t-MAP_delay, Matrix_AP, beta, dbetadt) #Actual 2nd ord. ODE
        
        return dbetadt, DDbetaDDt


    " Free Ca2+ transients (exp. validated) "

    def MU_free_Ca_ODE_func(self, t, l, MU_type, beta, Ca, dCadt):
        
        if MU_type == 'fast':
            c_1, c_2, c_3 = 2.4*10.**3, 4.3*10.**5, 0.54  

        elif MU_type == 'slow': 
            c_1, c_2, c_3 = 6.037*10.**3, 1.8*10.**5, 0.54 
            
        #c_1, c_2, c_3 = self.c_1, 4.3*10.**5, self.c_3
        #c_1, c_2, c_3 = self.c_1, 1.8*10.**5, self.c_3
           
        p4 = [-0.4688, 3.7127, -11.0323, 14.063, -5.4709]
        amp = (l**4)*p4[0] + (l**3)*p4[1] + (l**2)*p4[2] + l*p4[3] + p4[4]
        
        if l < 1:
            amp = 0.1*l + 0.2
        #amp = 1
        
        p2 = [0.3783, -0.8320, 1.1885]
        width = (l**2)*p2[0] + l*p2[1] + p2[2]
        
        if l < 1.23:
            width = 0.73
        elif l > 2.04:
            width = 1.072
        
        if MU_type == 'slow':
            DDCaDDt = c_3*beta - 1/amp*(c_1*dCadt + width*c_2*Ca) #Actual 2nd ord. ODE
        elif MU_type == 'fast': 
            DDCaDDt = c_3*beta - 1/amp*(c_1*dCadt + width*c_2*Ca) #Actual 2nd ord. ODE

        return DDCaDDt
    
    def MU_free_Ca_func(self, t, y, l_M, MU_type): 

        #CA_delay = 2.1*10**-3
        CA_delay = 0
        MU_AP_train = y[0]
        Ca = y[2] 
        dCadt = y[3]
        beta = MU_AP_train  #from previously solved ODE for MUAP
        DDCaDDt = self.MU_free_Ca_ODE_func(t-CA_delay, l_M, MU_type, beta, Ca, dCadt)  
        
        return dCadt, DDCaDDt
    
    
    # " CaTn bounding (exp. validated) "
    
    # def Ca_Tn(self, t, y, MU_type):     

    #     Ca, CaTn = y[2], y[4]
        
    #     if Ca < 0:
    #         Ca = 0 
        
    #     if MU_type == 'slow':
    #         k1, k2, T0 = 2e13, 12, 8.7e-5
            
    #     elif MU_type == 'fast':
    #         k1, k2, T0 = 5e12, 16, 2.4e-4
        
    #     dCaTndt = k1*(T0-CaTn)*Ca**2 - k2*CaTn

    #     return dCaTndt


    " Active state "
    
    def act(self, y, MU_type, s):
        
        if MU_type == 'slow':
            k1, k2 = 11.5, 15.2
            Ca, a = y[2]*8e5, y[4] 
            
        elif MU_type == 'fast' and self.muscle == 'CF' and self.species == 'animal':  # MU 
            k1, k2 = 32.5, 72.2
            Ca, a = y[2]*1e6*s, y[4] 
            
        elif MU_type == 'fast' and self.muscle == 'GM' and self.species == 'animal':
            k1, k2 = 32.5, 72.2
            Ca, a = y[2]*8e5*s, y[4] 
                
        if Ca > a:
            dadt = -(k2*a - k1*Ca)*(1 - a)
        else:
            dadt = -(k2*a - k1*Ca) 
           
        return dadt
    
    " FL relationship (Lloyd-Besier 2003) "

    def Force_Length_func(self, act, l_M):
        
        X = l_M
        a = 0.45   
        b = (0.15*(1-act))+1
        f_FL = np.exp(-((X-b)/a)**2)  # Lloyd Besier
        
        return f_FL
    
    
    " FV relationship (adapted from Arnault Caillet PhD thesis) "
    
    def velo_fFV(self, act, l_M, f_CE, FL_force, MU_type, vmax):
        
        fmax = 1.4
        af = 0.17
        
        # Defining fFV involved parameters
        if MU_type == 'slow':
            kMU = 0.5
        elif MU_type == 'fast':
            kMU = 1
        
        fv = 0.9 + 0.1*act
        
        if l_M < 1:
            g = FL_force
        else:
            g = 1
            
        b = (fmax - 1)/(1 + 2/af)
        K = (kMU*fv*g)
        
        #fFV relationship inverted
        if f_CE >= 1:
            vel = b*((f_CE - 1)/(fmax - f_CE)) # rat soleus (Krylow, Sandercock)
            vel = vel*K
        else:
            vel = (f_CE - 1)/(f_CE/(af*K))
        
        return vel*vmax
    
    
    def f_fFV(self, v_M, FL_force, act, l_M, MU_type):
        
        fmax = 1.4
        af = 0.17
        
        # Defining fFV involved parameters
        if MU_type == 'slow':
            kMU = 0.5
        elif MU_type == 'fast':
            kMU = 1
        
        fv = 0.9 + 0.1*act
        
        if l_M < 1:
            g = FL_force
        else:
            g = 1
        
            
        b = (fmax - 1)/(2 + 2/af)
        K = kMU*fv*g
        
        if v_M < -1:
            fvel = 1/(1 - (v_M/(af*K)))
        if v_M >= -1 and v_M < 0:
            fvel = 1/(1 - (v_M/(af*K)))
        elif v_M >= 0:
            fvel = (1 + fmax*(v_M/(K*b)))/(1 + v_M/(K*b))
        
        return fvel


    " Muscle yield & sag (Brown 1999) "    

    def Yield(self, y, V):
        
        cy, Vy, Ty = 0.35, 0.1, 0.2
        Y = y
        
        dY = (1 - cy*(1 - np.exp((-np.abs(V))/Vy)) - Y)/Ty
        
        return dY

    def Sag(self, act, s, t):
        
        Ts = 0.043
        
        if t > 0 and t < (self.fs - 23.227)/82.426 and self.fs < 100:  
            As = (self.fs - 1.024)/22.891
        else:
            As = 0.8
            
        dS = (As - s)/Ts
        
        return dS
    
    
    """ ODE SYSTEMS """
    " Full MT dynamics "
    
    def ODE_system_MT(self, t, y, alpha, Matrix_AP, MU_type):
            
        if int(t/self.dt) == 0:  # initial pennation angle
            alpha[int(t/self.dt)] = self.alpha_0
            
        l_T = self.l_MT[int(t/self.dt)] - (y[5])*np.cos(alpha[int(t/self.dt)]) # tendon length
        eps_T = (l_T - self.l_T_slack)/self.l_T_slack # new tendon strain    
        f_SE = self.T_force(eps_T) # tendon force
        f_PE = self.PE_force(y[5]/self.l_M_opt) # passive el. force    
        f_CE = f_SE/np.cos(alpha[int(t/self.dt)]) - f_PE  # contractile el. force
        alpha[int((t + self.dt)/self.dt)] = self.penn_ang(self.l_MT[int((t + self.dt)/self.dt)], y[6], l_T, self.l_M_0*self.l_M_opt, self.alpha_0) # update pennation angle

        dbetadt, DDbetaDDt = self.MU_AP_func(t, y, Matrix_AP) # MU action potential

        dCadt, DDCaDDt = self.MU_free_Ca_func(t, y, y[5]/self.l_M_opt, MU_type) # Free Ca (remember to avoid negligible negative values)

        dadt = self.act(y, MU_type, y[7]) # Active state (intermediate step)
                
        FL = self.Force_Length_func(y[4], y[5]/self.l_M_opt) # F-L relationship    

        dldt = self.velo_fFV(y[4], y[5]/self.l_M_opt, f_CE/FL, FL, MU_type, self.vmax) # velocity

        dY = self.Yield(y[6], dldt/self.vmax) # yielding (Brown 1999)
                
        dS = self.Sag(y[4], y[7], t)

        return [dbetadt, DDbetaDDt, dCadt, DDCaDDt, dadt, dldt, dY, dS]
    
    
    def ODE_system_M(self, t, y, Matrix_AP, MU_type):
                        
        dbetadt, DDbetaDDt = self.MU_AP_func(t, y, Matrix_AP) # MU action potential

        dCadt, DDCaDDt = self.MU_free_Ca_func(t, y, self.l_MT[int(t/self.dt)], MU_type) # Free Ca (remember to avoid negligible negative values)

        dadt = self.act(y, MU_type, y[5]) # Active state (intermediate step)
        
        dS = self.Sag(y[4], y[5], t)

        return [dbetadt, DDbetaDDt, dCadt, DDCaDDt, dadt, dS]  
    
    
    
    " Solve ODE system and calculate new states (dyn/act cases) "
    " Muscle-tendon unit simulation "

    def run_MT_simulation(self):
        
        print('There are ', self.Nr, ' discharging MUs in this simulation')
        self.F0MU_distribution = self.F0MU_distrib_func() # isom. force distribution (muscle specific)
        
        "Run the muscle model simulation."
        for i in range(self.Nr):
            print('Computing Force for MU n°', str(i + 1))

            #sp_matrix = self.time[self.distimes[0,i]]
            
            #Matrix_AP = sp_matrix.astype(float)  # i-th MU discharge times [s] 
            MU_type = self.MU_type_id_func(i)  # i-th MU type identification (fast/slow)
            Matrix_AP = self.distimes.astype(float)
                    
            y0 = [self.MUAP_0, self.MUAP_0, self.Ca_0, self.Ca_0, self.act_0, self.l_M_opt, self.y_0, self.s_0]  # set initial states
            p = (self.alpha[i,:], Matrix_AP, MU_type)  # set ODE parameters
            sol = solve_ivp(self.ODE_system_MT, [self.time[0], self.time[-1]], y0, args=p, method='LSODA', t_eval=self.time, max_step=self.dt/4)  # solve IVP

            self.MUAP[i,:] = sol.y[0]  # get MUAP
            self.free_Ca[i,:] = sol.y[2]  # get free [Ca]
            self.active_state[i,:] = sol.y[4]  # get active state
            self.l_M[i,:] = sol.y[5]  # get l_M
            self.yielding[i,:] = sol.y[6]  # get yielding coeff.
            self.sag[i,:] = sol.y[7] # get sag

            # now recalculate data based on l_M values..
            self.alpha[i,0] = self.alpha_0
                
            for l in range(len(self.time)):
                self.l_T[i,l] = self.l_MT[l] - (self.l_M[i,l])*np.cos(self.alpha[i,l]) # tendon length
                self.eps_T[i,l] = (self.l_T[i,l] - self.l_T_slack)/self.l_T_slack # new tendon strain    
                self.f_SE[i,l] = self.T_force(self.eps_T[i,l]) # tendon force
                self.f_PE[i,l] = self.PE_force(self.l_M[i,l]/self.l_M_opt) # passive el. force 
                self.f_CE[i,l] = self.f_SE[i,l]/np.cos(self.alpha[i,l]) - self.f_PE[i,l] # contractile element force
                self.f_FL[i,l] = self.Force_Length_func(self.active_state[i,l], self.l_M[i,l]/self.l_M_opt)
                self.vel[i,l] = self.velo_fFV(self.active_state[i,l], self.l_M[i,l]/self.l_M_opt, self.f_CE[i,l]/self.f_FL[i,l], self.f_FL[i,l], MU_type, self.vmax)
                self.f_M[i,l] = self.f_fFV(self.vel[i,l]/self.vmax, self.f_FL[i,l], self.active_state[i,l], self.l_M[i,l]/self.l_M_opt, MU_type)
                
                self.alpha[i,l+1] = self.penn_ang(self.l_MT[l+1], self.l_M[i,l], self.l_T[i,l], self.l_M_opt, self.alpha_0) # update pennation angle
        
        if self.y == 'y' and MU_type == 'slow':   # yielding only for slow twitch MUs
            MU_Force_list = self.yielding * self.active_state * self.f_CE + self.f_PE
        elif self.s == 'y' and MU_type == 'fast': 
            MU_Force_list = self.active_state * self.f_M * self.f_FL + self.f_PE
        else:
            MU_Force_list = self.active_state * self.f_CE + self.f_PE
            
        F_MU_list = self.F0MU_distribution[0:self.Nr, :] * MU_Force_list  # from normalised MU forces (to F0 of each one) to N
        Tot_Muscle_force = F_MU_list.sum(axis=0)  # Total muscle force (in N)
        
        return Tot_Muscle_force, self.MUAP, self.free_Ca, self.active_state, self.l_M, self.yielding, self.sag  # Return force and solutions


    " Excitation-activation simulation (no tendon) "

    def run_M_simulation(self):
        
        print('There are ', self.Nr, ' discharging MUs in this simulation')
        self.F0MU_distribution = self.F0MU_distrib_func() # isom. force distribution (muscle specific)
        
        "Run the muscle model simulation."
        for i in range(self.Nr):
            print('Computing Force for MU n°', str(i + 1))

            #sp_matrix = self.time[self.distimes[0,i]]
            
            #Matrix_AP = sp_matrix.astype(float)  # i-th MU discharge times [s] 
            MU_type = self.MU_type_id_func(i)  # i-th MU type identification (fast/slow)
            Matrix_AP = self.distimes.astype(float)
                    
            y0 = [self.MUAP_0, self.MUAP_0, self.Ca_0, self.Ca_0, self.act_0, self.s_0]  # set initial states
            p = (Matrix_AP, MU_type)  # set ODE parameters
            sol = solve_ivp(self.ODE_system_M, [self.time[0], self.time[-1]], y0, args=p, method='LSODA', t_eval=self.time, max_step=self.dt/4)  # solve IVP

            self.MUAP[i,:] = sol.y[0]  # get MUAP
            self.free_Ca[i,:] = sol.y[2]  # get free [Ca]
            self.active_state[i,:] = sol.y[4]  # get active state
            self.sag[i,:] = sol.y[5] # get sag
                
            for l in range(len(self.time)):
                self.f_PE[i,l] = self.PE_force(self.l_MT[l]) # passive el. force 
                self.f_CE[i,l] = self.Force_Length_func(self.active_state[i,l], self.l_MT[l])
                
        if self.y == 'y' and MU_type == 'slow':   # yielding only for slow twitch MUs
            MU_Force_list = self.yielding * self.active_state * self.f_CE + self.f_PE
        elif self.s == 'y' and MU_type == 'fast': 
            MU_Force_list = self.active_state * self.f_CE + self.f_PE
        else:
            MU_Force_list = self.active_state * self.f_CE + self.f_PE
            
        F_MU_list = self.F0MU_distribution[0:self.Nr, :] * MU_Force_list  # from normalised MU forces (to F0 of each one) to N
        Tot_Muscle_force = F_MU_list.sum(axis=0)  # Total muscle force (in N)
        
        return Tot_Muscle_force, self.MUAP, self.free_Ca, self.active_state, self.sag  # Return force and solutions
    











#%%
###############################################################################
""" MODIFY parameters according to the benchmark """
###############################################################################

user = 'Andrea'
MN_pool = 1  # n. of theoretical MUs in the real pool 
Nr = 1 # n. of (exp.) MUs to represent in the pool
pool = 'n' # exp or pool
spread = 'evenly' # evenly or identified 
species = 'animal' # human/animal

dt = 1e-4 # time step (x-data)

input_folder = 'C:\\Users\\' + user + '\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\benchmark_input'

save = 'n' # save results 'y' or 'n'

#%%

root = tk.Tk() # Initialise input window
root.withdraw()  # Hide the root window

benchmark = simpledialog.askstring("Input", "Select benchmark ('max', 'sub', 'len', 'fast', 'Ca', 'CaTn'):") # max or sub

""" MAXIMAL benchmark (Sandercock 1997) """

if benchmark == 'max':
    
    scale = simpledialog.askstring("Input", "Amplitude displacement scale (e.g., 0.05, 0.1, 0.25, 0.5, 1, 2):")
    
    os.chdir(input_folder + '\\' + benchmark)
    t_end = 2
    muscle = 'SOL' # dorsi/plantar
    time_dt = np.arange(0, t_end, dt) # time for BB
    Distimes = np.arange(0, t_end, 1/70) # create array of dischare times at 70 Hz
    disp = np.genfromtxt('displacement.dat', delimiter='')
    disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # load displacement
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [1.36, 17.1, 17.1, 1, 6*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)-2 # Musculo-tendon length (mm)
    l_MT = l_MT_0 + disp*float(scale) # scaled MT length    
    exp_force = np.genfromtxt('force' + scale + '.dat', delimiter='') # load experimental force
    exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,2,dt)) # interpolate to have equal n of points
    yielding = 'n' # yielding not included
    sag = 'n'
    
    """ SUB-MAXIMAL benchmark (Perreault 2003) """
    
elif benchmark == 'sub':
    
    trial = simpledialog.askstring("Input", "Trial type? ('iso' or 'dyn'):")
    stim = simpledialog.askstring("Input", "Stimulation type? ('c' or 'v'):")
    fs = simpledialog.askstring("Input", "Stimulation frequency (Hz):")
    muscle = 'SOL' # dorsi/plantar
    yielding = 'y' # yielding included
    sag = 'y'
    
    os.chdir(input_folder + '\\' + benchmark) # load displacement, MVC and update MT lengths
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [26.8, 65, 30, 1, 7.5*np.pi/180] # MVC and M/T lengths
    l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0) - 4  # Musculo-tendon length (mm)
    
    t_end = 2
    time_dt = np.arange(0, t_end, dt) # time for BB
    os.chdir(input_folder + '\\' + benchmark + '\\' + stim + '_freq') 
    Distimes = np.load(fs + '_' + stim + '_times.npy') # exp. discharge times
    
    if trial == 'iso':
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # constant MT length
        exp_force = np.genfromtxt('force_isometric_' + stim + fs + '.dat', delimiter='')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))
    
    elif trial == 'dyn':
        
        d = simpledialog.askstring("Input", "Displacement amplitude (1 or 8mm):")
        
        disp = np.genfromtxt('displacement_' +  d + '.dat', delimiter='') # load displacement
        disp = sp.interpolate.interp1d(disp[:,0], disp[:,1], kind='cubic')(np.arange(0,2+dt,dt)) # interpolate displacement
        l_MT = l_MT_0 + (disp+8) # scaled MT length
        exp_force = np.genfromtxt('force_' + stim + fs + '_' + d + '.dat', delimiter='')
        exp_force = sp.interpolate.interp1d(exp_force[:,0], exp_force[:,1], kind='cubic')(np.arange(0,t_end,dt))


    """ TWITCH, SUB-TETANIC, TETANIC at different lengths (Kim 2015) """

elif benchmark == 'len':
    
    l = simpledialog.askstring("Input", "Length? ('0', '8', or '16'):")
    fs = simpledialog.askstring("Input", '"Stimulation frequency (Hz):", or "twitch"')
    yielding = 'y'
    sag = 'n'
    
    os.chdir(input_folder + '\\' + benchmark)
    MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [27.1, 65, 30, 1, 7.5*np.pi/180] # MVC and M/T lengths
    t_end = 1.4
    muscle = 'SOL' # dorsi/plantar
    time_dt = np.arange(0, t_end, dt) # time for BB
    
    if l == '0':
        d = 8
    elif l == '8':
        d = 0 
    elif l == '16':
        d = -8
        
    l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0) - 4    # Musculo-tendon length (mm)
    l_MT_0 = l_MT_0 + d  # apply displacement
    l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array

    exp_force = np.load(l + '_' + fs + '_interp.npy')
    if fs == 'twitch':
        Distimes = np.round(np.array(np.load(l + '_' + fs + '_times.npy')), 3)
    else:
        Distimes = np.load(l + '_' + fs + '_times.npy') # exp. discharge times
    
    
    """ Test fast fibres (Brown 1999 - Cat CF, Brown 1999) """

elif benchmark == 'fast':
    
    benchmark2 = simpledialog.askstring("Input", "muscle or MU: ")
    
    if benchmark2 == 'muscle':
        
        trial = simpledialog.askstring("Input", "Trial type? ('iso_f', 'iso_l', 'dyn'):")
        yielding = 'n'
        sag = 'y'
    
        if trial == 'iso_f':
            
            os.chdir(input_folder + '\\' + benchmark + '\\' + benchmark2 + '\\' + 'iso')
            fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (15, 30, 40, 50, 120):")
            exp_force = np.load(trial + '_' + fs + '_interp.npy')
        
            if fs == '15':   # define end time for setting discharges
                t_dis = 0.5
            elif fs == '30':
                t_dis = 0.25
            elif fs == '40':
                t_dis = 0.18
            elif fs == '50':
                t_dis = 0.13
            elif fs == '120':
                t_dis = 0.09
            
            Distimes = np.arange(0, t_dis, 1/float(fs)) # exp. discharge times
            t_end = 0.6
            muscle = 'CF' # dorsi/plantar
            time_dt = np.arange(0, t_end, dt)
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [15.4, 17, 0.37*56, 1, 0] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
        
        elif trial == 'iso_l':
            
            os.chdir(input_folder + '\\' + benchmark + '\\' + benchmark2 + '\\' + 'iso')
            l_M = simpledialog.askstring("Input", "Muscle length (0.8, 0.9, 1.0, 1.1, 1.2):")
            exp_force = np.load(trial + '_' + l_M + '_interp.npy')
            
            Distimes = np.arange(0, 0.25, 1/30) # exp. discharge times
            t_end = 0.36
            muscle = 'CF' # dorsi/plantar
            time_dt = np.arange(0, t_end, dt)
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [15.4, 17, 0.37*56, float(l_M), 0] # MVC and M/T lengths
            l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)
            l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
            
        elif trial == 'dyn':
            
            os.chdir(input_folder + '\\' + benchmark + '\\' + benchmark2 + '\\' + 'dyn')
            d = simpledialog.askstring("Input", "Displacement amplitude ('short' or 'length'):")
            
            t_end = 0.16
            Distimes = np.arange(0, t_end, 1/120) # exp. discharge times
            muscle = 'CF' # dorsi/plantar
            time_dt = np.arange(0, t_end, dt)
    
            MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [15.4, 17, 0.37*56, 0.95, 0] # MVC and M/T lengths
            exp_force = np.load('120_0.95_' + d + '_interp.npy')
            disp = np.load('disp_' +  d + '_interp.npy') # load displacement
            l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)
            l_MT = np.empty((len(time_dt)+1), dtype=object) # full MT length array
            l_MT[0:-1] = np.ones((len(time_dt)), dtype=object)*l_MT_0 + (disp*l_M_opt)*np.cos(alpha_0)
            l_MT[-1] = l_MT[-2]
    
    
    elif benchmark2 == 'MU':
        
        os.chdir(input_folder + '\\' + benchmark + '\\' + benchmark2)
        trial = 'iso'
        yielding = 'n'
        sag = 'y'
        
        fs = simpledialog.askstring("Input", "Stimulation frequency in Hz (25, 30, 35, 40, 150):")
        exp_force = np.load(trial + '_' + fs + '_interp.npy')
        
        Distimes = np.load(trial + '_' + fs + '_times.npy') # exp. discharge times
        t_end = 0.7
        muscle = 'GM' # dorsi/plantar
        time_dt = np.arange(0, t_end, dt)
    
        MVC, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0.078, 24.8, 13.5, 1, 0] # MVC and M/T lengths
        l_MT_0 = l_T_slack + (l_M_opt*l_M_0)*np.cos(alpha_0)  # Musculo-tendon length (mm)
        l_MT = np.ones((len(time_dt)+1), dtype=object)*l_MT_0 # full MT length array
        
    
    """ Test Ca dynamics (Hollingworth, Rincon exp. data) """
    
elif benchmark == 'Ca':
    
    fibre = simpledialog.askstring("Input", "Fibre type? ('slow', 'fast'):")
    yielding = 'n'
    sag = 'n'
    
    os.chdir(input_folder + '\\' + benchmark)
    
    Ca_slow_23 = "Ca_slow_23_100Hz.csv" # load digitized slow fibre data
    Ca_slow_23 = pd.read_csv(Ca_slow_23, delimiter = ' ')
    Ca_slow_23 = Ca_slow_23.to_numpy()  
    
    Ca_fast_35 = "Ca_fast_35_125Hz.csv" # load digitized fast fibre data
    Ca_fast_35 = pd.read_csv(Ca_fast_35, delimiter = ' ')
    Ca_fast_35 = Ca_fast_35.to_numpy()

    t_end = 1.4
    time_dt = np.arange(0, t_end, dt)
    
    if fibre == 'slow':
        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1, 0, 30, 1, 0] # at optimal sarcomere length (assumption)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array
        T = 1/102 # frequency = 100 Hz (corrected)
        Distimes = np.arange(0, 0.04, T)
        muscle = 'SOL' # slow fibre muscle
    elif fibre == 'fast':
        MVC, l_MT, l_T_slack, l_M_opt, l_M_0, alpha_0 = [0, 1.6, 0, 30, 1.6, 0] # at longer sarcomere length (see article)
        l_MT = np.ones((len(time_dt)), dtype=object)*l_MT # full MT length array
        T = 1/125 # frequency = 125 Hz
        Distimes = np.arange(0, 0.08, T)
        muscle = 'CF' # fast fibre muscle
    
    
os.chdir(cwd)

#%%
" Create dictionary as model's input "
    
parameters = {
    'muscle': muscle, 
    'time': time_dt, 
    'pool': pool,
    'dt': dt,
    'MVC': MVC,
    'spread': spread,
    'MN_pool': MN_pool,
    'yielding': yielding,
    'sag': sag,
    'Nr': Nr,
    'l_T_slack': l_T_slack, # scaled with respect to Rajagopal model
    'l_M_opt': l_M_opt, # scaled with respect to Rajagopal model
    'alpha_0': alpha_0,
    'l_MT': l_MT,
    'species': species,
    'vmax': 10.5428 *l_M_opt,
    # 'c_1': 0,  # for Ca ODE optimization
    # 'c_3': 0
}

states = {
    'MUAP_0': 0,
    'Ca_0': 0,
    'CaTn_0': 0,
    'act_0': 1e-9,
    'l_M_0': l_M_0,
    'y_0': 1,
    's_0': 1
}

#%%
###############################################################################
""" PARAMETERS OPTIMIZATION (Nelder-Mead-lstsq)"""
###############################################################################

""" 1) MAXIMAL benchmark [MVC, Vmax] estimation """

# def obj(x, parameters, states, Distimes, exp_force): 
    
#     parameters['MVC'] =  x[0]
#     parameters['vmax'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
#     force_sim, _, _, _, _ = model.run_MT_simulation() # Run the simulation
    
#     residuals = force_sim - exp_force  # assuming both are in same units
#     return np.sum(residuals**2)  

# x0 = [1.2, 10]
# bnds = [(1, 1.5), (8, 12)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})


"""  2) SUB-MAXIMAL benchmark [MVC, Ca_max] estimation """

# def obj(x, parameters, states, Distimes, exp_force): 
    
#     parameters['MVC'] =  x[0]
#     parameters['Ca_max'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
#     force_sim, _, Ca, a, l_M, _ = model.run_MT_simulation() # Run the simulation
    
#     residuals = force_sim - exp_force  # assuming both are in same units
#     return np.sum(residuals**2)  

# x0 = [26, 2.3e5]
# bnds = [(25, 29), (1e5, 5e5)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_force), 
#                x0, method='Nelder-Mead', bounds=bnds, options={'disp': True})


""" 3) Ca transient ODE [c1, c2, c3] parameters estimation """

# def obj(x, parameters, states, Distimes, exp_data): 
    
#     parameters['c_1'] = x[0]
#     parameters['c_3'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
#     _, _, Ca, _, _ = model.run_M_simulation() # Run the simulation

#     idx = np.isin(np.round(time_dt,4), np.round((exp_data[:,0]-exp_data[0,0])*1e-3, 4)).nonzero()[0]

#     residuals = Ca[0,idx]*10**6 - exp_data[:,1]  # assuming both are in same units
#     return np.sum(residuals**2)  

# if fibre == 'slow':
#     exp_data = Ca_slow_23
#     x0 = [8*10.**3, 0.6]
#     bnds = [(1e2, 1e5), (0.1, 2)]
# elif fibre == 'fast':
#     exp_data = Ca_fast_35
#     x0 = [2.8*10.**3, 0.7]
#     bnds = [(1e2, 1e5), (0.1, 2)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_data), 
#                 x0, method='Nelder-Mead', bounds=bnds, options={'disp': True, 'maxiter': 500})

""" 4) CaTn bounding ODE parameters estimation """

# def obj(x, parameters, states, Distimes, exp_data): 
    
#     parameters['c_1'] = x[0]
#     parameters['c_3'] = x[1]
    
#     model = MN_driven_model(parameters, states, Distimes) # Create an model class instance
#     _, _, Ca, _, _, _ = model.run_M_simulation() # Run the simulation

#     idx = np.isin(np.round(time_dt,4), np.round((exp_data[:,0]-exp_data[0,0])*1e-3, 4)).nonzero()[0]

#     residuals = CaTn[0,idx]*10**6 - exp_data[:,1]  # assuming both are in same units
#     return np.sum(residuals**2)  

# if fibre == 'slow':
#     exp_data = CaTn_slow
#     x0 = [8*10.**3, 0.6]
#     bnds = [(1e2, 1e5), (0.1, 2)]
# elif fibre == 'fast':
#     exp_data = CaTn_fast
#     x0 = [2.8*10.**3, 0.7]
#     bnds = [(1e2, 1e5), (0.1, 2)]

# res = minimize(lambda x: obj(x, parameters, states, Distimes, exp_data), 
#                 x0, method='Nelder-Mead', bounds=bnds, options={'disp': True, 'maxiter': 500})

#%% 
###############################################################################
""" Running simulations and plot solutions """
###############################################################################

model = MN_driven_model(parameters, states, Distimes) # Create an model class instance


if benchmark == 'max' or benchmark == 'sub' or benchmark == 'len' or benchmark == 'fast':
    
    force_sim, _, Ca, a, l_M, _, _ = model.run_MT_simulation() # Run the simulation

    # Plot the force profiles
    plt.rcParams['figure.dpi'] = 400
    if 'benchmark2' in globals() and benchmark2 == 'muscle':
        plt.plot(time_dt, force_sim/MVC, 'r', label='Simulated Force')
    else:
        plt.plot(time_dt, force_sim, 'r', label='Simulated Force')
    plt.plot(time_dt, exp_force, 'k', label='Exp. Force')
    #plt.plot(time_dt, a[0,:])
    #plt.ylabel('Active state', weight='bold', fontsize=12)
    plt.ylabel('Force [N]', weight='bold', fontsize=12)
    plt.xlabel('Time [s]', weight='bold', fontsize=12)
    plt.title('Reconstructed ' + muscle + ' force for ' + str(Nr) + ' MUs', weight='bold')
    plt.legend(loc='lower right')
    plt.grid()
    

elif benchmark == 'Ca':
    
    _, _, Ca, _, _ = model.run_M_simulation()
    
    plt.rcParams['figure.dpi'] = 400

    # Plot excitation/activation quantities
    if fibre == 'slow':
        plt.plot(time_dt, Ca[0,:]*10**6, 'g', label='Simulated (23°C)')
        plt.plot((Ca_slow_23[:,0] - Ca_slow_23[0,0])*10**-3, Ca_slow_23[:,1], 'k--', label = 'Rincon et al. 2021 (23°C)')
        plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]', weight='bold', fontsize=12) 
        plt.xlabel('Time [s]', weight='bold', fontsize=12)
        plt.title('Free [$Ca^{2+}$] for slow mouse fibres', weight='bold', fontsize=15)
        plt.legend(loc='upper right')
        plt.xlim((0, 0.12))
        plt.grid()
    elif fibre == 'fast':
        plt.plot(time_dt, Ca[0,:]*10**6, 'g', label='Simulated (35°C)')
        plt.plot((Ca_fast_35[:,0] - Ca_fast_35[0,0])*10**-3, Ca_fast_35[:,1], 'k--', label = 'Hollingworth 1996 (35°C)')
        plt.ylabel('Free [$Ca^{2+}$] [$\mu$M]', weight='bold', fontsize=12) 
        plt.xlabel('Time [s]', weight='bold', fontsize=12)
        plt.title('Free [$Ca^{2+}$] for fast mouse fibres', weight='bold', fontsize=15)
        plt.legend(loc='upper right')
        plt.xlim((0, 0.12))
        plt.grid()   



#%%
# # Error metrics (%mAE, %MAE)
# mean_abs_error = np.mean((np.abs(force_sim - exp_force)/MVC)*100)
# max_abs_error = np.max((np.abs(force_sim - exp_force)/MVC)*100)

#%%
# Save the figure if needed

os.chdir('C:\\Users\\Andrea\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Code\\Python_Scripts\\BB_tests\\fastCFcat')

if save == 'y':
    print("Saving results...")
    np.save('120_0.95_length', force_sim, allow_pickle=True)

