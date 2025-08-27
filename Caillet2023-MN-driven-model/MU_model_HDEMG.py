"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Created on Tue Jan 28 15:33:16 2025
___________________________________

MN-driven model based on Caillet et al. 2023 for simulating isometric and dynamic muscle contractions.

"""

import os
import numpy as np
import tkinter as tk
from scipy import signal
import scipy.io as sio
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
        self.l_M_opt = self.P['l_M_opt']
        self.l_T_slack = self.P['l_T_slack']
        self.distimes = distimes
        
        self.path = self.P['path']    # only for experimental inputs...
        self.l_MT = np.ones(len(self.time) + 1)*self.P['l_MT_0']  # Set MT lengths
        
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
            
        # Extract Fth recruitment in % of MVC (first discharge indeces)
        self.F_thr = np.empty((self.Nr))
        for f in range(np.size(self.distimes, axis=1)):
            self.F_thr[f] = self.path[0, self.distimes[0, f][0, 0]]*100 
        
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
            if self.Nr > 30: # use MU types if more than 5 MU was identified
                if i/self.Nr < 0.9:
                    MU_type='slow' # TA
                else: 
                   MU_type='fast'
            else:
                MU_type='slow'
                
        elif self.muscle == 'GM' and self.species == 'human':
            if self.Nr > 1: # use MU types if more than 1 MU was identified
                if i/self.Nr < 0.5:
                    MU_type='slow' # GM
                else: 
                    MU_type='fast'
            else:
                MU_type = 'slow'
                
        elif self.muscle == 'SOL' and self.species == 'animal':
            MU_type = 'slow' # GM
        elif (self.muscle == 'CF' or self.muscle == 'GM') and self.species == 'animal':
            MU_type = 'fast' # CF
    
        return MU_type
    
    
    " Distribution of force with MU recruitment (TA) "
    
    def Fth_distrib_func(self, MN):  #Obtained from processed literature data, typical muscle-specific distributions of MU force recruitment thresholds (%MVC)
        if self.muscle == 'TA' or self.muscle == 'SOL' or self.muscle == 'GM':
            return 0.5052*(58.1*MN/self.MN_pool+120**((MN/self.MN_pool)**1.83)) 
        elif self.muscle == 'ND':
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


    " Active state "
    
    def act(self, y, MU_type, s):
        
        if MU_type == 'slow' and self.species == 'animal':
            k1, k2 = 11.5, 15.2
            Ca, a = y[2]*8e5, y[4] 
            
        elif MU_type == 'fast' and self.muscle == 'CF' and self.species == 'animal':  # MU 
            k1, k2 = 32.5, 72.2
            Ca, a = y[2]*1e6*s, y[4] 
            
        elif MU_type == 'fast' and self.muscle == 'GM' and self.species == 'animal':
            k1, k2 = 32.5, 72.2
            Ca, a = y[2]*8e5*s, y[4] 
            
        if MU_type == 'slow' and self.species == 'human' and self.muscle == 'TA':
            k1, k2 = 11.5, 15.2
            Ca, a = y[2]*8e3, y[4] 
            
        if MU_type == 'slow' and self.species == 'human' and self.muscle == 'GM':
            k1, k2 = 11.5, 15.2
            Ca, a = y[2]*4e4, y[4] 
                
        elif MU_type == 'fast' and self.species == 'human' and self.muscle == 'GM':  # MU 
            k1, k2 = 32.5, 72.2
            if s == 'y':
                Ca, a = y[2]*1e5*s, y[4]
            else:
                Ca, a = y[2]*8e4, y[4]
                
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

            sp_matrix = self.time[self.distimes[0,i]-1] # index correction from matlab
            
            Matrix_AP = sp_matrix.astype(float)  # i-th MU discharge times [s] 
            MU_type = self.MU_type_id_func(i)  # i-th MU type identification (fast/slow)
            
            self.fs = np.mean(1/np.diff(Matrix_AP))
                    
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
""" INSERT filename, musclename and contraction intensity """
###############################################################################

user = 'Andrea' # z5517249 or Andrea

test = simpledialog.askstring("Input", "Test subject ID? (e.g. S01_TD, from 1 to 14):")
pool = simpledialog.askstring("Input", "Consider pool functionality? (y/n):")
muscle = simpledialog.askstring("Input", "Simulated muscle (TA/GM):")
intensity = simpledialog.askstring("Input", "Contraction intensity? (10/30/50/70):")
spread = simpledialog.askstring("Input", "MU distribution? (identified/evenly):")
species = simpledialog.askstring("Input", "Species? (human/animal):")

#test = 'S09_TD' # participant ID
MN_pool = 400
# pool = 'y'
# muscle = 'GM'
# intensity = '30'
# spread = 'identified' # evenly or identified
# yielding = 'n'
# sag = 'n'
# species = 'human' # human/animal
save = 'n' # save results 'y' or 'n'

#%%
" PARAMETERS SETTING FROM INPUT TEST CASE "

test_cases= np.array([['S01_TD', 'TA', 1, 0.317, 23.83, 11.47, 2048, 0.08, 0.085, 57.65],
                      ['S01_TD', 'GM', 1, 0.381, 19.83, 22.08, 2048, 0.08, 0.085, 70.12], 
                      ['S02_TD', 'TA', 1, 0.295, 17.21, 10.63, 2048, 0.085, 0.092, 43.86],
                      ['S02_TD', 'GM', 1, 0.336, 24.06, 21.83, 2048, 0.085, 0.092, 61.26], 
                      ['S03_TD', 'TA', 1, 0.363, 25.62, 14.12, 2048, 0.1, 0.11, 92.58],
                      ['S03_TD', 'GM', 1, 0.426, 22.12, 22.19, 2048, 0.1, 0.11, 134.64], 
                      ['S04_TD', 'TA', 1, 0.302, 12.90, 10.09, 2048, 0.08, 0.085, 52.56],
                      ['S04_TD', 'GM', 1, 0.357, 23.01, 27.49, 2048, 0.08, 0.085, 91.17], 
                      ['S06_TD', 'TA', 1, 0.238, 12.59, 10.88, 2048, 0.07, 0.075, 39.91],
                      ['S06_TD', 'GM', 1, 0.302, 17.53, 25.22, 2048, 0.07, 0.075, 67.00], 
                      ['S07_TD', 'TA', 1, 0.359, 15.93, 10.67, 2048, 0.09, 0.09, 80.00],
                      ['S07_TD', 'GM', 1, 0.418, 11.54, 22.49, 2048, 0.09, 0.09, 134.02], 
                      ['S09_TD', 'TA', 1, 0.323, 11.13, 12.38, 2048, 0.075, 0.08, 57.52],
                      ['S09_TD', 'GM', 1, 0.377, 12.83, 23.05, 2048, 0.075, 0.08, 150.63], 
                      ['S10_TD', 'TA', 1, 0.373, 18.85, 9.91, 2048, 0.08, 0.09, 85.24],
                      ['S10_TD', 'GM', 1, 0.438, 7.15, 24.11, 2048, 0.08, 0.09, 206.07], 
                      ['S12_TD', 'TA', 1, 0.347, 14.96, 13.70, 2048, 0.085, 0.09, 74.03],
                      ['S12_TD', 'GM', 1, 0.395, 20.69, 22.15, 2048, 0.085, 0.09, 151.56], 
                      ['S13_TD', 'TA', 1, 0.375, 23.91, 13.91, 2048, 0.1, 0.1, 119.12],
                      ['S13_TD', 'GM', 1, 0.423, 30.70, 23.37, 2048, 0.1, 0.1, 217.53], 
                      ['S14_TD', 'TA', 1, 0.375, 39.67, 9.13, 2048, 0.09, 0.095, 111.52],
                      ['S14_TD', 'GM', 1, 0.450, 12.10, 28.62, 2048, 0.09, 0.095, 194.06], 
                      ], dtype=object)  # lengths in m, freq in Hz, angles in degrees, PCSA in cm2

[name, muscle, l_M_0, l_MT_0, MVC_rec, alpha_0, fs, a, h, vol] = test_cases[np.where((test_cases[:, 0] == test) & (test_cases[:, 1] == muscle))[0][0]]

#%%
" Load discharge times indeces "

os.chdir('C:\\Users\\' + user + '\\Dropbox\\UNSW - Andrea - Luca [PhD]\\Data\\HDsEMG_study\\TD_group\\' + name + '\\data_4model\\')

if muscle == 'TA':
    task = 'dorsi'
elif muscle == 'GM':
    task = 'plantar'

data = sio.loadmat(name + '_' + intensity + 'MVC_' + task + '_4model.mat')
d = {key: data[key] for key in data.keys()}
Distimes = d['Distimeclean'][0,0]  # extract exp. edited discharge times
path = d['path'] # extract dynamometer recorded force [% MVC]
ngrid = d['ngrid'][0][0]

if ngrid == 2:  # check the second grid if there's one
    Distimes2 = d['Distimeclean'][0,1] # extract data from second grid
    if np.size(Distimes, axis=1) == 0 and np.size(Distimes2, axis=1) != 0:  # check for empty data
        Distimes = Distimes2
    elif np.size(Distimes2, axis=1) == 0 and np.size(Distimes, axis=1) != 0:
        Distimes = Distimes
    else:
        Distimes = np.concatenate((Distimes, Distimes2), axis=1) # concatenate grids if grids == 2


sorted_indices = np.argsort([Distimes[0, i][0, 0] for i in range(np.size(Distimes, axis=1))]) # sort MUs based on recruitment order
Distimes = Distimes[:, sorted_indices]

os.chdir(cwd)

#%%
" Time, MVC and number of MUs "

T = 1/fs  # fd = 2048 Hz
time_real = np.arange(0, np.size(path, axis=1)*T, T) # actual time vector
dt = 0.0001  # dt for solution and arrays
time_dt = np.arange(0, np.size(path, axis=1)*dt, dt)  # time for solving ODE system
MVC_rec = MVC_rec * 9.81  # muscle max. isom. force [N]    
Nr = np.size(Distimes, axis=1)  # n. of MNs in the pool

x =  0.15 # OT Bioelettronica dynamometer heel to force-cell lenght [m]

#%%
" Scale model and extract single muscles contribution "

if muscle == 'TA':
    l_MT_Raj = 0.32642 # Rajagopal generic model [m]
    l_T_slack_Raj = 0.241 
    l_M_opt_Raj = 0.068
    r = 0.034  # moment arm
    M0_Raj = 41.03 # max. moment [Nm]
    Mtot = 68.26  # tot. moment from all dorsiflexors
elif muscle == 'GM':
    l_MT_Raj = 0.43418
    l_T_slack_Raj = 0.399
    l_M_opt_Raj = 0.051
    r = 0.04 
    M0_Raj = 52.6
    Mtot = 228

F0_Raj = M0_Raj/r
scale = l_MT_0/l_MT_Raj # scale factor from MT length
l_T_slack = l_T_slack_Raj*scale
l_M_opt = l_M_opt_Raj*scale

T_ankle = MVC_rec * ((np.sqrt((x-np.sqrt(h**2-a**2))**2 + a**2)) / np.cos(30*np.pi / 180)) # from dynamomter force to total ankle moment [Nm]

MVC = ((vol*np.cos(alpha_0*np.pi / 180))/(l_M_opt*(10**2))) * 60 # muscle MVC from vol (PCSA) and scaled opt fibre length, 60 = specific tetanic tension [N/cm2]
r_muscle = r * scale # scale the moment arm
M_ratio = M0_Raj / Mtot  # es. 52.6 / 228 ≈ 0.23
T_muscle = T_ankle * M_ratio  # torque specifico del GM


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
    'path': path,
    'sag': sag,
    'Nr': Nr,
    'l_T_slack': l_T_slack*(10**3), # scaled with respect to Rajagopal model
    'l_M_opt': l_M_opt*(10**3), # scaled with respect to Rajagopal model
    'alpha_0': alpha_0*np.pi / 180,
    'l_MT_0': l_MT_0*(10**3),
    'species': species,
    'vmax': 10.5428 *l_M_opt*(10**3),
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
# Create an instance of the MN_driven_model class
model = MN_driven_model(parameters, states, Distimes)

# Run the simulation
force_sim, _, _, _, _, _, _ = model.run_MT_simulation()

# Calculate muscle force contribution
exp_force = (T_muscle / r_muscle)*path[0,:]

#%%
" Filter simulated force "

b, a = signal.butter(4, 5/(2048/2), 'low') # LPF 
force_sim_filt = signal.filtfilt(b, a, force_sim) 
    
#%%
" Visual validation and result saving "  

plt.rcParams['figure.dpi'] = 400
fig, ax = plt.subplots()

# Plot the force profile
ax.plot(time_real, force_sim_filt[0:len(time_real)], 'r', label='Simulated Force')
ax.plot(time_real, exp_force, 'k', label='Exp. Force')
ax.set_ylabel('Force [N]', weight='bold', fontsize=12)
ax.set_xlabel('Time [s]', weight='bold', fontsize=12)
ax.set_title('Reconstructed ' + muscle + ' force for ' + str(Nr) + ' MUs', weight='bold')
ax.legend(loc='upper right')
ax.grid()

ax2 = ax.twinx()  
cmap = plt.get_cmap("Greens")  
color = 'green'

MU_range = Nr + 5  # Height for the motor units axis 

for i in range(Nr):
    color = cmap(np.clip(i / Nr, 1, 1))  # Assign a unique color to each motor unit
    ax2.eventplot(Distimes[0, i]/fs, lineoffsets=i + 1, colors='green', linelengths=0.5, 
              linewidth=0.8, alpha=0.7)

ax2.set_ylabel('n.MU', color='green', weight='bold', fontsize=12)
ax2.set_ylim([0, MU_range])  # Place the raster plot at the bottom of the main plot

ax2.spines['right'].set_color('green') 
ax2.spines['right'].set_linewidth(2)   
ax2.tick_params(axis='y', colors='green')

plt.tight_layout()
plt.show()

#%%
# Error metrics (%mAE, %MAE)
indices = force_sim_filt > 1 # calculate MAE only where MU activity is detected
mean_abs_error = np.mean((np.abs(force_sim_filt[indices] - exp_force[indices])/exp_force[indices])*100)
max_abs_error = np.max((np.abs(force_sim_filt[indices] - exp_force[indices])/exp_force[indices])*100)

#%%
# Save the figure if needed
if save == 'y':
    print("Saving results...")
    np.save('Predicted_force', force_sim, allow_pickle=True)


    
    