"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Created on Tue Jan 28 15:33:16 2025
___________________________________

MN-driven model based on Caillet et al. 2023 for simulating isometric and dynamic muscle contractions
given the experimental MN (or MU) discharge times as inputs. The muscle is modelled as an ensamble of N
in-parallel 3-elements Hill-type models, each one representing an identified active MU. The single normalized
MU forces are scaled based on an experimentally-derived force distribution from the human TA and summed up
to simulate the whole muscle force.

The model is benchmarked against animal experimental data from slow (SOL) and fast (EDL) muscles.
It can also be used with experimental data from human MN spike trains.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
cwd = os.getcwd()

class MU_model():

    def __init__(self, parameters, states, distimes):

        # Check and assign parameters 

        if parameters is None:
            raise ValueError('No parameters set!')
        elif not isinstance(parameters, dict):
            raise ValueError('Parameters are not in dict format!')
        else:
            self.P = parameters  
        
        # Assign attributes from input dictionary

        self.time = self.P['time']
        self.pool = self.P['pool'] 
        self.dt = self.P['dt']
        self.muscle = self.P['muscle']
        self.species = self.P['species']
        self.spread = self.P['spread']
        self.y = self.P['yielding']
        self.s = self.P['sag']    
        
        self.MVC = self.P['MVC'] # max tetanic force
        self.Nr = self.P['Nr']  # n. of identified motor units
        self.MN_pool = self.P['MN_pool']  # n. of theoretical MNs in the muscle pool
        self.vmax = self.P['vmax'] # max velocity
        self.alpha_0 = self.P['alpha_0'] # pennation angle
        self.l_MT = self.P['l_MT'] # muscle-tendon length
        self.l_M_opt = self.P['l_M_opt'] # optimal length (at which tetanic force is produced)
        self.l_T_slack = self.P['l_T_slack'] # tendon slack length
        
        # self.Ca_max = self.P['Ca_max']  # only for optimization procedures...
        # self.c_1 = self.P['c_1'] 
        # self.c_3 = self.P['c_3']
        # self.vmax = self.P['vmax']
        # self.k1 = self.P['k1']
        # self.k2 = self.P['k2']

        # Check and assign states 
    
        if states is None:
            raise ValueError('No states set!')
        elif not isinstance(states, dict):
            raise ValueError('States are not in dict format!')
        else:
            self.S = states 
        
        self.MUAP_0 = self.S['MUAP_0']
        self.Ca_0 = self.S['Ca_0']
        self.act_0 = self.S['act_0']
        self.l_M_0 = self.S['l_M_0']
        self.y_0 = self.S['y_0']
        self.s_0 = self.S['s_0']
        
        
        # self.F_thr = np.empty((self.Nr)) # only for HDsEMG input...
        # for f in range(np.size(self.distimes, axis=1)):
        #     self.F_thr[f] = self.path[0, self.distimes[0, f][0, 0]]*100 # Extract Fth recruitment in % of MVC (first discharge indeces)
        

        " Check and assign discharge times input "
        
        if distimes is None:
            raise ValueError("Discharge times must be provided")
        else:
            self.distimes = distimes  
        
        self.distimes = np.atleast_1d(np.asarray(distimes, dtype=float)).ravel()
        if self.distimes.size >= 2: # if more than 1 discharge
            self.fs = float(np.mean(1.0 / np.diff(self.distimes))) # calculate instantaneouos freq. for sag calculation
        else:
            self.fs = float(self.P['stim_freq'])  # otherwise take input from user
        
        self.initialize_arrays() # Preallocate output arrays 
        

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
        self.yielding = np.empty((self.Nr, len(self.time)))
        self.sag = np.empty((self.Nr, len(self.time)))


    " MU type (so far for TA and GM) "    
    
    def MU_type_id_func(self, i):
    
        if self.muscle == 'SOL': # Soleus
            MU_type = 'slow' 
        elif self.muscle == 'GM' or self.muscle == 'EDL': # GasMed or Ext. Digitorum longus
            MU_type = 'fast' 
        elif self.Nr > 1 and self.species == 'human' and self.spread == 'evenly': # example of human Tibialis anterior (still to be fully implemented...)
            if i/self.Nr < 0.9: 
                MU_type='slow' 
            else: 
                MU_type='fast'
        elif self.Nr > 1 and self.species == 'human' and self.spread == 'identified':
            if i/self.Nr < 0.9: 
                MU_type='slow' 
            else: 
                MU_type='fast'
    
        return MU_type
    
    
    " Distribution of force with MU recruitment (TA) "
    
    def fth_percent(self, mn_idx):

        x = mn_idx / self.MN_pool  # normalized MN index in the pool
        return 0.5052 * (58.1 * x + 120 ** (x ** 1.83))  # Recruitment threshold (in %MVC) for MU index in the pool

    def f0mu_rel(self, mn_idx):
        
        x = mn_idx / self.MN_pool  # normalized MN index in the pool
        return 7.86e-4 * (3.0 * x + 8.20 ** (x ** 5.29)) # Relative distribution (unitless) of each MU F0

    def map_identified_positions(self, last_rec):
        """
        Map the Nr identified MUs in the pool [1..last_rec].
        - 'evenly': equispaced positions
        - 'identified': use self.F_thr (%MVC) if available, otherwise fallback to 'evenly'
        Return ordered integers of shape (Nr,)
        """
        Nr = self.Nr
        if self.spread == 'evenly': 
            pos = np.rint(np.linspace(1, last_rec, Nr)).astype(int) # Distribute Nr points rounded to the closest integer between 1 and last_rec
            return pos

        pool_idx = np.arange(1, self.MN_pool + 1) # if spread == 'identified' ( with Fth %MVC)
        fth_pool = self.fth_percent(pool_idx) # recruitment threshold (%MVC) for each MU
        pos = np.zeros(Nr, dtype=int)
        for i in range(Nr):
            pos[i] = int(np.argmin(np.abs(fth_pool - self.F_thr[i])) + 1) # pool index with threshold %MVC closest to the exp. Fth 
        
        pos.sort() # ascending order to have continuous bins

        pos = np.unique(pos) # remove duplicates mantaining Nr
        if pos.size < Nr:
            extra = np.setdiff1d(
                np.rint(np.linspace(1, last_rec, Nr)).astype(int), pos # add uniformely distributed indeces if pos.size < Nr
            )
            need = Nr - pos.size
            pos = np.sort(np.concatenate([pos, extra[:need]]))
        else:
            pos = pos[:Nr]
        return pos

    def compute_bins(self, positions, last_rec):
        """
        Boundaries based on integer indeces.
        Returns (left_edges, right_edges) of shape (Nr,) inclusive.
        """
        pos = positions.astype(float)
        
        mids = np.rint((pos[:-1] + pos[1:]) / 2.0).astype(int) # mid-point between adjacent MUs
       
        left_edges = np.empty_like(positions) # left/right boundaries (inclusive)
        right_edges = np.empty_like(positions)
        left_edges[0] = 1 # the first cluster starts from 1
        right_edges[-1] = last_rec # last cluster ends to last_rec
        if positions.size > 1:
            left_edges[1:] = mids # each i>0 cluster starts from the previous cluster' mid point
            right_edges[:-1] = mids - 1 # each cluster i<Nr-1 ends to (mid point)-1 of the next one
        else:
            right_edges[0] = last_rec # special case of single MU covering the whole [1..last rec]
       
        bad = right_edges < left_edges # verify left <= right
        right_edges[bad] = left_edges[bad]
        return left_edges, right_edges

    def f0mu_weights(self):
        """
        F0MU weights for each of the Nr MUs identified
        - Scale the distribution relative to the subject MVC
        - If pool == 'y':each MU is a recruitable cluster (< 100%MVC).
        - If pool == 'n': each Mu inherits only the corresponding mapped index 
        Return an array shaped (Nr, 1).
        """
        pool_idx = np.arange(1, self.MN_pool + 1)

        fth_pool = self.fth_percent(pool_idx)  # get MU % threshold within 100 %MVC    

        valid = np.where(fth_pool <= 100.0)[0] # last recruitable MU
        last_rec = (valid[-1] + 1) if valid.size else 1

        
        rel = self.f0mu_rel(pool_idx) # Relative distribution and scaling to MVC                 
        rel_sum = rel.sum()
        if rel_sum <= 0:
            raise ValueError("F0MU relative distribution sums to zero.")
        scale = self.MVC / rel_sum
        f0mu_pool_N = rel * scale  # N for each MU in the pool

        pos = self.map_identified_positions(last_rec) # Mapping of the Nr intentified MUs: (Nr,), 1..last_rec

        if self.pool == 'y':
            
            L, R = self.compute_bins(pos, last_rec) # Bins left & right extremes: each MU is a cluster of MUs
            w = np.zeros(self.Nr)
            for i in range(self.Nr):
                w[i] = f0mu_pool_N[L[i]-1:R[i]].sum()  # sum F0 contributions (N) of the MUs of the pool in the i-th bin 

            target = f0mu_pool_N[:last_rec].sum() # numerical alignment
            s = w.sum()
            if s > 0:
                w *= (target / s)
            return w.reshape(self.Nr, 1)

        else:  # No clustering: single MU value in the pool
            
            w = f0mu_pool_N[pos - 1]
            return w.reshape(self.Nr, 1)
    
    " SE (Thelen 2003, John 2013) "

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

    
    " PE (Thelen 2003, John 2013) "

    def PE_force(self, l_M):
        
        k1, k2 = 5, 0.6  # Thelen 2003
        
        if l_M < 1:
            f_PE = 0
        elif l_M >= 1:
            f_PE = (np.exp((k1*(l_M-1))/k2)-1)/(np.exp(k1)-1)
            
        return f_PE
    
    
    " Calculate pennation angle "

    def penn_ang(self, l_MT, l_T, l_M_0, alpha0):

        alpha = alpha0
        w = l_M_0*np.sin(alpha0)
        cosalpha = 1/(np.sqrt(1 + (w/(l_MT-l_T))**2))
        alpha = np.arccos(cosalpha)
            
        if alpha > 1.4706289:  # np.arccos(0.1), 84.2608 degrees
            alpha = 1.4706289
        
        return alpha
    
    " ODEs "
    " Convert discharge times to binary "
    
    def is_t_a_firing_time_func(self, t_round, Matrix_AP, chosen_precision): 

        t_int = int(t_round/chosen_precision+10**-4) # integer in the scale of the chosen precision
        Matrix_AP_newprecision = (Matrix_AP/chosen_precision).astype(int)  # list of integers in the scale of chosen precision
        if t_int in Matrix_AP_newprecision:
            binary = 1
        else:
            binary = 0
            
        return binary
    
    """ MNAP """

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
    
    """ MUAP """

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
        
        if MU_type == 'slow': 
            c_1, c_2, c_3 = 6.029*10.**3, 1.8*10.**5, 0.54
            # c_1, c_2, c_3 = self.c_1, 1.8*10.**5, self.c_3
        
        elif MU_type == 'fast':
            c_1, c_2, c_3 = 2.4*10.**3, 4.3*10.**5, 0.6
            #c_1, c_2, c_3 = self.c_1, 4.3*10.**5, self.c_3
        
        if l < 0.9892:
            amp = 0.7926 # horizontal line for missing data
        elif l >= 0.9892 and l <= 1.1379:
            amp = 1.3947 * l - 0.5871 # left line
        elif l > 1.1379 and l < 1.239:
            amp = 1 # plateau
        else:
            amp = 1 - 0.4623 * (l - 1.239) # right line

        p2 = [0.3783, -0.8320, 1.1885] # Fitted quadratic (Konishi 1991)
        width = (l**2)*p2[0] + l*p2[1] + p2[2]
        
        if l < 1.23: # assume extreme values out of the exp. length interval
            width = 0.73
        elif l > 2.04:
            width = 1.072
        
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

    " Active state (Adapted from Hussein 2022)"
    
    def act(self, y, MU_type, s):
        
        if MU_type == 'slow':
            k1, k2 = 10.8, 14.7
            #k1, k2 = self.k1, self.k2
            #Ca, a = y[2]*2.51772e5, y[4]
            Ca, a = y[2]*285916, y[4] 
            #Ca, a = y[2]*self.Ca_max, y[4] 
            
        elif MU_type == 'fast' and self.muscle == 'EDL' and self.species == 'animal':  # MU 
            k1, k2 = 11.8, 14.7
            if self.s == 'y':
               Ca, a = y[2]*2.91772e5*s, y[4] 
            else:
               Ca, a = y[2]*2.91772e5*s, y[4] 
            
        elif MU_type == 'fast' and self.muscle == 'GM' and self.species == 'animal':
            k1, k2 = 32.5, 72.2
            if self.s == 'y':
                Ca, a = y[2]*(1.99249e5)*s, y[4] 
                # Ca, a = y[2]*self.Ca_max*s, y[4] 
            else:
                Ca, a = y[2]*(1.99249e5), y[4] 
                # Ca, a = y[2]*self.Ca_max*s, y[4] 
                
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
        
        if t > 0 and t < (self.fs - 23.227)/82.426 and self.fs < 100:  # time dependent below 100 Hz (exp. fitting, Chelikowski)
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
        alpha[int((t + self.dt)/self.dt)] = self.penn_ang(self.l_MT[int((t + self.dt)/self.dt)], l_T, self.l_M_0*self.l_M_opt, self.alpha_0) # update pennation angle

        dbetadt, DDbetaDDt = self.MU_AP_func(t, y, Matrix_AP) # MU action potential

        dCadt, DDCaDDt = self.MU_free_Ca_func(t, y, y[5]/self.l_M_opt, MU_type) # Free Ca (remember to avoid negligible negative values)

        dadt = self.act(y, MU_type, y[7]) # Active state 
                
        FL = self.Force_Length_func(y[4], y[5]/self.l_M_opt) # F-L relationship    

        dldt = self.velo_fFV(y[4], y[5]/self.l_M_opt, f_CE/FL, FL, MU_type, self.vmax) # F-V relationship

        dY = self.Yield(y[6], dldt/self.vmax) # Yielding 
                
        dS = self.Sag(y[4], y[7], t) # Sag 

        return [dbetadt, DDbetaDDt, dCadt, DDCaDDt, dadt, dldt, dY, dS]
    
    
    def ODE_system_M(self, t, y, Matrix_AP, MU_type):
                        
        dbetadt, DDbetaDDt = self.MU_AP_func(t, y, Matrix_AP) # MU action potential

        dCadt, DDCaDDt = self.MU_free_Ca_func(t, y, self.l_MT[int(t/self.dt)], MU_type) # Free Ca (remember to avoid negligible negative values)

        dadt = self.act(y, MU_type, y[5]) # Active state
        
        dS = self.Sag(y[4], y[5], t) # Sag 

        return [dbetadt, DDbetaDDt, dCadt, DDCaDDt, dadt, dS]  
    
    
    
    " Solve ODE system and calculate new states (dyn/act cases) "
    " Muscle-tendon unit simulation "

    def run_MT_simulation(self):
        
        print('There are ', self.Nr, ' discharging MUs in this simulation')
        self.F0MU_distribution = self.f0mu_weights() # isom. force distribution (muscle specific)
        
        "Run the muscle model simulation."
        for i in range(self.Nr):
            print('Computing Force for MU n°', str(i + 1))

            MU_type = self.MU_type_id_func(i)  # i-th MU type identification (fast/slow)
            Matrix_AP = self.distimes.astype(float)
                    
            y0 = [self.MUAP_0, self.MUAP_0, self.Ca_0, self.Ca_0, self.act_0, self.l_M_opt, self.y_0, self.s_0]  # set initial states
            p = (self.alpha[i,:], Matrix_AP, MU_type)  # set ODE parameters
            sol = solve_ivp(self.ODE_system_MT, [self.time[0], self.time[-1]], y0, args=p, method='LSODA', t_eval=self.time, max_step=self.dt/4)  # solve IVP

            self.MUAP[i,:] = sol.y[0]  # get MUAP
            self.free_Ca[i,:] = sol.y[2]  # get free [Ca]
            self.active_state[i,:] = sol.y[4]  # get active state
            self.l_M[i,:] = sol.y[5]  # get l_M
            self.yielding[i,:] = sol.y[6]  # get yielding 
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
                
                self.alpha[i,l+1] = self.penn_ang(self.l_MT[l+1], self.l_T[i,l], self.l_M_opt, self.alpha_0) # update pennation angle
        
        if self.y == 'y' and MU_type == 'slow':   # yielding only for slow MUs
            MU_Force_list = self.yielding * self.active_state * self.f_M * self.f_FL + self.f_PE
        else:
            MU_Force_list = self.active_state * self.f_M * self.f_FL + self.f_PE
            
        F_MU_list = self.F0MU_distribution[0:self.Nr, :] * MU_Force_list  # from normalised MU forces (to F0 of each one) to N
        Tot_Muscle_force = F_MU_list.sum(axis=0)  # Total muscle force (in N)
        
        return Tot_Muscle_force, self.MUAP, self.free_Ca, self.active_state, self.l_M, self.yielding, self.sag  


    " Muscle simulation (no SE) "

    def run_M_simulation(self):
        
        print('There are ', self.Nr, ' discharging MUs in this simulation')
        self.F0MU_distribution = self.f0mu_weights() # isom. force distribution (muscle specific)
        
        "Run the muscle model simulation."
        for i in range(self.Nr):
            print('Computing Force for MU n°', str(i + 1))

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
    
        MU_Force_list = self.active_state * self.f_CE + self.f_PE
            
        F_MU_list = self.F0MU_distribution[0:self.Nr, :] * MU_Force_list  # from normalised MU forces (to F0 of each one) to N
        Tot_Muscle_force = F_MU_list.sum(axis=0)  # Total muscle force (in N)
        
        return Tot_Muscle_force, self.MUAP, self.free_Ca, self.active_state, self.sag  # Return force and solutions
    




 