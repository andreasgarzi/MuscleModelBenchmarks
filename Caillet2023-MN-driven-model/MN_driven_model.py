"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Created on Tue Jan 28 15:33:16 2025
___________________________________

MN-driven model based on Caillet et al. 2023 for simulating isometric and dynamic muscle contractions.

"""

import numpy as np
from scipy.integrate import solve_ivp

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
        
        if self.P['muscle'] != 'TA' and self.P['muscle'] != 'GM' and self.P['muscle'] != 'SOL':
            raise ValueError("Expected either 'TA', 'GM', or 'SOL' for muscle parameter")
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
            
        if self.P['sim'] != 'MT' and self.P['sim'] != 'act':
            raise ValueError("Expected either 'MT', or 'act' for sim parameter")
        else:  
            self.sim = self.P['sim']
        
        self.MVC = self.P['MVC']
        #self.path = self.P['path']
        self.Nr = self.P['Nr'] # Number of identified motor units
        self.MN_pool = self.P['MN_pool'] # Number of theoretical MNs in the muscle pool
        self.vmax = self.P['vmax']
        self.alpha_0 = self.P['alpha_0']
        #self.l_MT = np.ones(len(self.time) + 1)*self.P['l_MT_0']  # Set MT lengths
        self.l_MT = self.P['l_MT']
        self.l_M_opt = self.P['l_M_opt']
        self.l_T_slack = self.P['l_T_slack']
        self.distimes = distimes
    
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
        self.act_0 = self.S['act_0']
        self.l_M_0 = self.S['l_M_0']
        self.y_0 = self.S['y_0']
        
        #######################################################################
        " Check and assign discharge times input "
        
        if distimes is None:
            raise ValueError("Discharge times must be provided")
        else:
            self.distimes = distimes  # Use provided discharges
        
        # Sort MUs according to recruitment   
        #sorted_indices = np.argsort([self.distimes[0, i][0, 0] for i in range(np.size(self.distimes, axis=1))])
        #self.distimes = self.distimes[:, sorted_indices]
            
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
        self.SE_force = np.empty((self.Nr, len(self.time)))
        self.CE_force = np.empty((self.Nr, len(self.time)))
        self.PE_force = np.empty((self.Nr, len(self.time)))
        self.FL_force = np.empty((self.Nr, len(self.time)))
        self.vel = np.empty((self.Nr, len(self.time)))
        self.F_M = np.empty((self.Nr, len(self.time)))
        self.l_M = np.empty((self.Nr, len(self.time)))
        self.MUAP = np.empty((self.Nr, len(self.time)))
        self.active_state = np.empty((self.Nr, len(self.time)))
        self.free_Ca = np.empty((self.Nr, len(self.time)))
        self.yielding = np.empty((self.Nr, len(self.time)))


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
            MU_type = 'slow'
    
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
            eps_0 = 0.055 # strain at max. isometric force in rat soleus (5-6% from Monti et al.2003)
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

    
    " PEE (Thelen 2003) "

    def PEE_force(self, l_M):
        
        k1, k2 = 5, 0.6
        
        if l_M < 1:
            normalized_PE_force_list = 0
        elif l_M >= 1:
            normalized_PE_force_list = (np.exp((k1*(l_M-1))/k2)-1)/(np.exp(k1)-1)
            
        return normalized_PE_force_list

    
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

    def MU_free_Ca_ODE_func(self, t, l_M_norm, MU_type, beta, gamma,  dgammadt):
        
        if MU_type == 'fast':
           c_1, c_2, c_3 = 2.5*10.**3,  4.3*10.**5, 0.8  # 0.7

        elif MU_type == 'slow':
           c_1, c_2, c_3 = 9.5*10.**3, 1.8*10.**5, 0.8 
           
        if l_M_norm <= 1.0:
            amp = 0.8
        elif l_M_norm <= 1.15:
            amp = 0.8 + 1.33*(l_M_norm - 1.0)
        elif l_M_norm <= 1.3:
            amp = 1.0
        else:
            amp = 1.0 - 0.6*(l_M_norm - 1.3)
        
        if l_M_norm <= 1.15:
            width = 1.0
        else:
            width = (1.0 - 0.4*(l_M_norm - 1.15))
        
        if MU_type == 'slow':
            DDgammaDDt = c_3*beta - 1/amp*(c_1*dgammadt + width*c_2*gamma)   # #Actual 2nd ord. ODE
        elif MU_type == 'fast':
            c_2 = c_2*(gamma*10**5)
            DDgammaDDt = c_3*beta - 1/amp*(c_1*dgammadt + width*c_2*gamma) 
            
        return DDgammaDDt
    
    def MU_free_Ca_func(self, t, y, l_M, MU_type): 

        #CA_delay = 2.1*10**-3
        CA_delay = 0
        MU_AP_train = y[0]
        gamma = y[2] 
        dgammadt = y[3]
        beta = MU_AP_train  #from previously solved ODE for MUAP
        DDgammaDDt = self.MU_free_Ca_ODE_func(t-CA_delay, l_M, MU_type, beta, gamma, dgammadt)  
        
        return dgammadt, DDgammaDDt


    " Active state "
    
    def MU_active_state_func(self, y):
        
        Ca, act = y[2], y[4]
        amin = 1*10**-9
        ac = (act - amin)/(1 - amin)
    
        if self.species == 'human' and self.muscle == 'TA':
            Ca = Ca/(4*10**-5) # Ca normalisation
            n = 3  # species & exp conditions dependent parameter 
        elif self.species == 'human' and self.muscle == 'GM':
            Ca = Ca/(1.4*10**-5) # Ca normalisation
            n = 3  # species & exp conditions dependent parameter 
        elif self.species == 'animal':
            Ca = Ca*2e5
            n = 2.5
            
        # if Ca > ac:
        #     K = 15.79 # 25.79
        #     dadt = (Ca**n - ac)*(ac + K)*(1 - ac) # ascending phase limited to 1
        # else:
        #     K = 60.8329 
        #     dadt = (Ca - ac)*(ac + K) # decay following a non-normalized trend
    
        k3, k4 = 15.5, 22.2  
        
        if Ca > ac:
            dadt = -(k4*ac - k3*Ca**n)*(1 - ac)
        else:
            dadt = -(k4*ac - k3*Ca)
        
        return dadt

    
    " FL relationship (Lloyd-Besier 2003) "

    def Force_Length_func(self, y, l_M_opt):
        
        X, act = y[5]/l_M_opt, y[4]
        a = 0.45
        b = (0.15*(1-act))+1
        
        return np.exp(-((X-b)/a)**2)
    
    
    " FV relationship (adapted from Arnault caillet PhD thesis) "
    
    def velo_fFV(self, y, CE_force, FL_force, l_M_opt, MU_type, vmax):
        
        act, l_M = y[4], y[5]/l_M_opt
        fmax = 1.4
        
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
            
        if MU_type == 'slow':
            af = 0.17 # the lower the lower the inferior limit
        elif MU_type == 'fast':
            af = 0.34
            
        b = (fmax - 1)/(2 + 2/af)
        K = (kMU*fv*g)
        
        #fFV relationship inverted
        if CE_force >= 1:
            vel = b*((CE_force-1)/(fmax-CE_force)) # rat soleus (Krylow, Sandercock)
            vel = vel*K
        else:
            vel = (CE_force-1)/(CE_force/(af*K))
        
        return vel*vmax


    " Muscle yielding (Brown 1999) "    

    def Yield(self, y, V):
        
        cy, Vy, Ty = 0.35, 0.1, 0.2
        Y = y[6]
        
        dY = (1 - cy*(1 - np.exp((-np.abs(V))/Vy)) - Y)/Ty
        
        return dY

    
    " Solve ODE system and calculate new states (dyn/act cases) "

    def run_simulation(self):
        
        print('There are ', self.Nr, ' discharging MUs in this simulation')
        self.F0MU_distribution = self.F0MU_distrib_func() # isom. force distribution (muscle specific)
        
        "Run the muscle model simulation."
        for i in range(self.Nr):
            print('Computing Force for MU n°', str(i + 1))

            #sp_matrix = self.time[self.distimes[0,i]]
            
            MU_type = self.MU_type_id_func(i)  # i-th MU type identification (fast/slow)
            Matrix_AP = self.distimes.astype(float)
            #Matrix_AP = sp_matrix.astype(float)  # i-th MU discharge times [s] 

            def ODE_system_MT(t, y, l_MT, l_M_0, l_M_opt, l_T_slack, Matrix_AP, MU_type, alpha_0, dt, alpha, vmax):
            
                if int(t/dt) == 0:  # initial pennation angle
                    alpha[int(t/dt)] = alpha_0
                l_T = l_MT[int(t/dt)] - (y[5])*np.cos(alpha[int(t/dt)]) # tendon length
                eps_T = (l_T-l_T_slack)/l_T_slack # new tendon strain    
                SE_force = self.T_force(eps_T) # tendon force
                PE_force = self.PEE_force(y[5]/l_M_opt) # passive el. force    
                CE_force = SE_force/np.cos(alpha[int(t/dt)]) - PE_force # contractile el. force
                alpha[int((t+dt)/dt)] = self.penn_ang(l_MT[int((t+dt)/dt)], y[5], l_T, l_M_0*l_M_opt, alpha_0) # update pennation angle

                dbetadt, DDbetaDDt = self.MU_AP_func(t, y, Matrix_AP) # MU action potential

                dgammadt, DDgammaDDt = self.MU_free_Ca_func(t, y, y[5]/l_M_opt, MU_type) # Free Ca (remember to avoid negligible negative values)

                dadt = self.MU_active_state_func(y) # Active state

                FL = self.Force_Length_func(y, l_M_opt) # F-L relationship    

                dldt = self.velo_fFV(y, CE_force/FL, FL, l_M_opt, MU_type, vmax) # velocity

                dY = self.Yield(y, dldt/vmax) # yielding (Brown 1999)

                return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dadt, dldt, dY]


            def ODE_system_act(t, y, l_M_0, Matrix_AP, MU_type):
                        
                dbetadt, DDbetaDDt = self.MU_AP_func(t, y, Matrix_AP) # MU action potential

                dgammadt, DDgammaDDt = self.MU_free_Ca_func(t, y, l_M_0, MU_type) # Free Ca (remember to avoid negligible negative values)

                dadt = self.MU_active_state_func(y) # Active state

                return [dbetadt, DDbetaDDt, dgammadt, DDgammaDDt, dadt]            


            if self.sim == 'MT':  # full MT dynamic case
                
                y0 = [self.MUAP_0, self.MUAP_0, self.Ca_0, self.Ca_0, self.act_0, self.l_M_opt, self.y_0]  # set initial states
                p = (self.l_MT, self.l_M_0, self.l_M_opt, self.l_T_slack, Matrix_AP, MU_type, self.alpha_0, self.dt, self.alpha[i,:], self.vmax)  # set ODE parameters
                sol = solve_ivp(ODE_system_MT, [self.time[0], self.time[-1]], y0, args=p, method='LSODA', t_eval=self.time, max_step=self.dt/4)  # solve IVP

                self.MUAP[i,:] = sol.y[0]  # get MUAP
                self.free_Ca[i,:] = sol.y[2]  # get free [Ca]
                self.active_state[i,:] = sol.y[4]  # get active state
                self.l_M[i,:] = sol.y[5]  # get l_M
                self.yielding[i,:] = sol.y[6]  # get yielding coeff.

                # now recalculate data based on l_M values..
                self.alpha[i,0] = self.alpha_0
                
                for l in range(len(self.time)):
                    self.l_T[i,l] = self.l_MT[l] - (self.l_M[i,l])*np.cos(self.alpha[i,l]) # tendon length
                    self.eps_T[i,l] = (self.l_T[i,l] - self.l_T_slack)/self.l_T_slack # new tendon strain    
                    self.SE_force[i,l] = self.T_force(self.eps_T[i,l]) # tendon force
                    self.PE_force[i,l] = self.PEE_force(self.l_M[i,l]/self.l_M_opt) # passive el. force 
                    self.CE_force[i,l] = self.SE_force[i,l]/np.cos(self.alpha[i,l]) - self.PE_force[i,l] # contractile element force
                    
                    self.alpha[i,l+1] = self.penn_ang(self.l_MT[l+1], self.l_M[i,l], self.l_T[i,l], self.l_M_opt, self.alpha_0) # update pennation angle
        
                if self.y == 'y' and MU_type == 'slow': # yielding only for slow twitch MUs
                    MU_Force_list = self.yielding * self.active_state * self.CE_force + self.PE_force
                else:
                    MU_Force_list = self.active_state * self.CE_force + self.PE_force
            
                F_MU_list = self.F0MU_distribution[0:self.Nr, :] * MU_Force_list  # from normalised MU forces (to F0 of each one) to N
                Tot_Muscle_force = F_MU_list.sum(axis=0)  # Total muscle force (in N)
        
                return Tot_Muscle_force, self.MUAP, self.free_Ca, self.active_state, self.l_M, self.yielding  # Return force and solutions
            
            
            elif self.sim == 'act':   # excitation-activation only case
                
                y0 = [self.MUAP_0, self.MUAP_0, self.Ca_0, self.Ca_0, self.act_0]  # set initial states
                p = (self.l_M_0, Matrix_AP, MU_type)  # set ODE parameters
                sol = solve_ivp(ODE_system_act, [self.time[0], self.time[-1]], y0, args=p, method='LSODA', t_eval=self.time, max_step=self.dt/4)  # solve IVP

                self.MUAP[i,:] = sol.y[0]  # get MUAP
                self.free_Ca[i,:] = sol.y[2]  # get free [Ca]
                self.active_state[i,:] = sol.y[4]  # get active state
 
                return self.MUAP, self.free_Ca, self.active_state  # Return solutions
            

