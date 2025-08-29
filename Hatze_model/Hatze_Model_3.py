
"""
Hatze, 1981 - “Myocybernetic control models of skeletal muscle: Characteristics and applications”

MODEL 3 (Muscle and Muscle-tendon)

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['lines.markersize'] = 4
matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rc('axes', grid=True, labelsize=12, titlesize=13, ymargin=0.01)
matplotlib.rc('legend', numpoints=1, fontsize=10)

# plt.close('all')



class Hatze_Muscle:
    
    def __init__(self, name, l_opt, fibre_type, G=1, lbda_T0=0, theta=np.pi/2, l0=0):
        
        """
        Whole muscle object following Hatze's model 3.
    
        Parameters:
        -----------
        
        name : str
            name of the muscle
    
        l_opt : float
            muscle optimal length
        
        fibre_type : str
            either 'slow' or 'fast' string that define the fibres type
        
        G : float (optional)
            muscle mass in kg, default: 1 (quantities are mass specific)
        
        lbda_T0 : float (optional)
            tendon rest length, default: 0 if the lumped model doesn't include a tendon element
        
        theta : float (optional)
            pennation angle when l = l_opt, default: pi/2 for a fusiform muscle
        
        l0 : float (optional)
            rest length of the muscle
    
        """
        
        self.name = name
        
        self.A0         = 0.372
        self.A0_p       = 3.6
        self.a2_p       = 0.14
        self.a3         = 3.2
        self.a3_p       = 0.105
        self.a6_p       = 6.97
        self.alpha_     = 0.08
        self.B          = 0.297
        self.c          = 1.373e-4
        self.cbar       = 3.7
        self.delta_b    = 1e-4
        self.d1         = 5.0
        self.d2         = 6.8
        self.enco       = 14.3
        self.fCf_ratio  = 1.33
        self.fibre_type = fibre_type
        self.k2         = 0.0001
        self.kappa_     = 0.0306
        self.lbda_T0    = lbda_T0
        self.l0         = l0
        self.q0         = 0.005
        self.s          = 1.0
        self.sigma      = 1.531
        self.sk         = 0.46
        self.theta      = theta
        self.xi_hat     = 2.90
        
        self.c_PE   = 4.17789
        self.k_PE   = 0.081371
        self.k_PE_p = 0.015
        
        self.max_stim_rate_set = {'fast': 100.0, 'slow': 28.0}
        self.max_stim_rate     = self.max_stim_rate_set[self.fibre_type]
        
        self.a7_set = {'fast': 5.94, 'slow': 2.70}
        self.a7     = self.a7_set[self.fibre_type]

        self.A2     = self.a2_p / self.A0
        self.A3     = self.a3_p / self.A0
        self.b1     = self.q0 / (1 - self.q0)
        
        self.set_l_opt(l_opt)
        self.set_contraction_params(self.a3)
        
        
        
        # ===================================== #
        #   Metabolic cost model constants      #
        # ===================================== #
        
        self.gh_set = {'fast': 150, 'slow': 24.4}
        self.phi_b_set = {'fast': 0.35, 'slow': 0.45}
        self.aF_ratio_set = {'fast': 0.28, 'slow': 0.16}
        
        self.G = G
        self.cv = 0
        
        self.aF_ratio = self.aF_ratio_set[self.fibre_type]
        self.f_b = 1e6
        self.gh = self.gh_set[self.fibre_type]
        self.kappa7 = 18.2
        self.kappa8 = 0.25
        self.phi_b = self.phi_b_set[self.fibre_type]


    
# =============================================================================
#     Parameters setting
# =============================================================================
    
        
    def set_l_opt(self, l_opt):
        
        """
        Method that updates muscle optimal length and updates associated parameters (lbda_S0, d, theta_b, lbda_opt).
        
        Parameters:
        -----------

        l_opt : float
            muscle optimal length
            
        """
        
        self.l_opt = l_opt

        # Update parameters that depend on l_opt
            
        if self.theta == np.pi/2:
            if self.lbda_T0 != 0:
                self.lbda_S0 = 0.98 * self.lbda_T0 + 0.02 * self.l_opt
            else:
                self.lbda_S0 = 0.6 * self.l_opt / (1 + 0.6 * self.alpha_)
            self.d = 0
            self.theta_b = self.theta
            self.lbda_opt = (self.l_opt - (1 + self.alpha_) * self.lbda_S0) / (1 + self.kappa_)
        else:
            self.d = (self.l_opt - self.lbda_T0) / np.tan(self.theta)
            self.theta_b = np.arctan(np.tan(self.theta) - self.alpha_ * self.lbda_T0 / self.d)
            self.lbda_opt = ((0.98 - 0.02 * self.alpha_) * self.l_opt - 0.98 * (1 + self.alpha_) * self.lbda_T0) / (1 + self.kappa_)
            self.lbda_S0 = 0.07 * self.lbda_opt


    def set_contraction_params(self, a3=3.2):
        
        """
        Method that updates the parameters of the contractions dynamics (a1, a2, a3, b1, b2, nu_dot_star and w_bar) by updating the value of the a3 parameter.
        
        Parameters:
        -----------

        a3 : float
            value of the a3 parameter for the contraction dynamics
            
        """
        
        self.a3 = a3
        
        # Update parameters that depend on a3
        self.a2 = - 0.5 * np.log(self.q0 * (1 - 1/self.fCf_ratio)) / np.sinh(self.a3 / 2)
        self.a1 = (self.q0 * np.exp(self.a2 * np.sinh(self.a3 / 2)) - np.exp(-self.a2 * np.sinh(self.a3 / 2))) / (1 - self.q0)
        self.b2 = (1 - self.q0) / (self.a1 + np.exp(-self.a2 * np.sinh(self.a3 / 2)))
        
        self.nu_dot_star = -0.5 + np.arcsinh(-np.log(-self.a1 + 1 / (self.b2 * (self.b1 + 0.999 * (1 / (self.a1 * self.b2) - self.b1)))) / self.a2) / self.a3
        self.w_bar = 0.01 * (1 / (self.a1 * self.b2) - self.b1 - 1)
        
    
    
    def set_contraction_fCf_ratio(self, fCf_=1.33):
        
        """
        Method that updates the parameters of the contractions dynamics (a1, a2, a3, b1, b2, nu_dot_star and w_bar) by updating the value of the fC/f ratio parameter.
        
        Parameters:
        -----------

        fCf_ : float
            value of the relative maximum stretching force
            
        """
        
        self.fCf_ratio = fCf_
        
        # Update parameters that depend on fCf_
        self.a2 = - 0.5 * np.log(self.q0 * (1 - 1/self.fCf_ratio)) / np.sinh(self.a3 / 2)
        self.a1 = (self.q0 * np.exp(self.a2 * np.sinh(self.a3 / 2)) - np.exp(-self.a2 * np.sinh(self.a3 / 2))) / (1 - self.q0)
        self.b2 = (1 - self.q0) / (self.a1 + np.exp(-self.a2 * np.sinh(self.a3 / 2)))
        
        self.nu_dot_star = -0.5 + np.arcsinh(-np.log(-self.a1 + 1 / (self.b2 * (self.b1 + 0.999 * (1 / (self.a1 * self.b2) - self.b1)))) / self.a2) / self.a3
        self.w_bar = 0.01 * (1 / (self.a1 * self.b2) - self.b1 - 1)
    


    def set_q0(self, q0=0.005):
        
        """
        Method that updates the initial muscle active state and updates associated parameters (a1, a2, b1 and b2).
        
        Parameters:
        -----------

        q0 : float
            initial muscle active state
        
        """
        
        self.q0 = q0
        
        # Update parameters that depend on q0
        self.a2 = - 0.5 * np.log(self.q0 * (1 - 1/self.fCf_ratio)) / np.sinh(self.a3 / 2)
        self.a1 = (self.q0 * np.exp(self.a2 * np.sinh(self.a3 / 2)) - np.exp(-self.a2 * np.sinh(self.a3 / 2))) / (1 - self.q0)
        self.b1 = self.q0 / (1 - self.q0)
        self.b2 = (1 - self.q0) / (self.a1 + np.exp(-self.a2 * np.sinh(self.a3 / 2)))
    
    
    
    
    
    
# =============================================================================
#     Auxiliary functions
# =============================================================================
    
    
    
    def compute_eps_1(self, y):
        
        """
        Polynomial auxiliary function used to compute the m function appearing in the excitation dynamics.
        See Eq (A1.38) in Appendix A1 from Hatze's book.
        
        Parameters:
        -----------

        y : float
            polynomial variable
            
        
        Output:
        -----------
    
        eps_1(y) : float
        
        """
        
        eps_1 = (y**2 + 2.334733 * y + 0.250621) / (y**2 + 3.330657 * y + 1.681534)
        
        return eps_1
    
    
    def compute_rho0(self, xi):
        
        """
        Method that computes the normalized Ca density (length depency factor in the Ca ion concentration Gamma). 
        See Eq (A1.15) in Appendix A1 from Hatze's book.
        
        Parameters:
        -----------

        xi : float
            normalized length of the CE (state variable)
            
        
        Output:
        -----------
    
        rho0 : float
            normalized Ca density
        
        """
        
        rho = 53300 * (self.xi_hat ** self.s - 1) / ((self.xi_hat / xi) ** self.s - 1)

        return rho
    
    
    def compute_k(self, xi):
        
        """
        Method that computes length-tension relationship for the muscle model.
        See Eq (3.34) in Hatze's book.
        
        Parameters:
        -----------

        xi : float
            normalized length of the CE (state variable)
            
        
        Output:
        -----------
    
        k(xi) : float
            length-tension relationship
        
        """
        
        k = np.exp(-((xi - 1) / self.sk) ** 2)
        
        return k
    
    
    def compute_k1(self, xi):
        
        """
        Method that computes internal fibre extension force for the muscle model.
        See Eq (4.17) in Hatze's book.
        
        Parameters:
        -----------

        xi : float
            normalized length of the CE (state variable)
            
        
        Output:
        -----------
    
        k1(xi) : float
            internal fibre extension force
        
        """
        
        k1 = 2 - np.exp(self.a6_p * (xi - 1))
        
        return k1
    
    
    def compute_S(self, n, r, psi, phi, xi, delta_eps):
        
        """
        Method that computes the auxiliary S function, that denotes the dependency on the distribution of active and semi-active MUs of the contraction velocity.
        See Eq (A1.27), (A1.46), (A1.69), (A1.70)-(A1.72) and (A1.77) in Hatze's book.
        
        Parameters:
        -----------

        n : float
            normalized population of active MUs (state variable)
        
        r : float
            normalized population of semi-active MUs (state variable)
        
        psi : float
            pseudo Ca ion concentration of the stimulated population of MUs (state variable)
        
        phi : float
            pseudo Ca ion concentration of the semi-active population of MUs (state variable)

        xi : float
            normalized length of the CE (state variable)
        
        delta_eps : float
            relative stretch potentiation
            
        
        Output:
        -----------
    
        S : float
            factor that denotes the dependency on the distribution of stimulated and semi-active MUs of the contraction velocity
        
        eps : float
            excitation function
        
        """
        
        Sn = (self.a2_p / self.B) - (self.a3_p / self.B) * n * (0.568 + 0.2307 * n)
        Snr = (self.a2_p / self.B) - (self.a3_p / self.B) * (n + 0.568 * r + 0.2307 * r**2)
        S0 = (self.a2_p / self.B) - (self.a3_p / self.B) * (n + r + 0.568 * (1 - n - r) + 0.2307 * (1 - n - r)**2)
        
        q_psi = self.compute_q(xi, psi)
        q_phi = self.compute_q(xi, phi)
        
        eps_n = q_psi * (np.exp(self.cbar * n) - 1) / (np.exp(self.cbar) - 1)
        eps_r = q_phi * (np.exp(self.cbar * (n + r)) - np.exp(self.cbar * n)) / (np.exp(self.cbar) - 1)
        eps_0 = self.q0 * (np.exp(self.cbar) - np.exp(self.cbar * (n + r))) / (np.exp(self.cbar) - 1)
        
        eps = (eps_n + eps_r + eps_0) * (1 + delta_eps)
        
        S = (eps_n * Sn + eps_r * Snr + eps_0 * S0) / eps
        
        return S, eps
    
    
    def compute_q(self, xi, Ca_con):
        
        """
        Method that computes the muscle active state.
        See Eq (3.29) in Hatze's book.
        
        Parameters:
        -----------

        xi : float
            normalized length of the CE (state variable)
        
        Ca_con : float
            pseudo Ca ion concentration, i.e. psi or phi (state variable)
            
        
        Output:
        -----------
    
        q : float
            muscle active state
        
        """
        
        if self.lbda_opt_shift:
            rho0 = self.compute_rho0(1) * xi
        else:
            rho0 = self.compute_rho0(xi)
        
        q = (self.q0 + (rho0 * Ca_con)**2) / (1 + (rho0 * Ca_con)**2)
        
        return q
    
    
    def compute_mn(self, n):
        
        """
        Function m(n) used to estimate the peudo Ca ion concentrations in the concentration dynamics. 
        See Eq (A1.39) in Appendix A1 from Hatze's book
        
        Parameters:
        -----------

        n : float
            normalized number of recruited MUs (state variable)
            
        
        Output:
        -----------
    
        m(n) : float
            weighting parameter in the excitation dynamics equations
        
        """
        
        y1 = self.cbar * self.A2 / self.A3 - self.cbar * (n + self.delta_b)
        y2 = self.cbar * self.A2 / self.A3
        
        eps_11 = self.compute_eps_1(y1)
        eps_12 = self.compute_eps_1(y2)
    
        mn = (np.exp(self.cbar * (n + self.delta_b)) * eps_11 / (self.A2 - self.A3 * (n + self.delta_b)) - eps_12 / self.A2) / (np.exp(self.cbar * (n + self.delta_b)) - 1)
        
        return mn
    
    
    def compute_mnr(self, n, r):
        
        """
        Function m(n,r) used to estimate the pseudo Ca ion concentrations in the concentration dynamics. 
        See Eq (A1.56) in Appendix A1 from Hatze's book
        
        Parameters:
        -----------

        n : float
            normalized number of recruited MUs (state variable)
            
        r : float
            normalized number of semi-active MUs (state variable)
            
        
        Output:
        -----------
    
        m(n,r) : float
            weighting parameter in the recruitment and excitation dynamic equations
        
        """
        
        y1 = self.cbar * self.A2 / self.A3 - self.cbar * (n + self.delta_b)
        y3 = self.cbar * self.A2 / self.A3 - self.cbar * (n + r + 2 * self.delta_b)
        
        eps_11 = self.compute_eps_1(y1)
        eps_13 = self.compute_eps_1(y3)
        
        mnr = (eps_13 / (self.A2 - self.A3 * (n + r + 2 * self.delta_b)) - eps_11 * np.exp(-self.cbar * (r + self.delta_b)) / (self.A2 - self.A3 * (n + self.delta_b))) / (1 - np.exp(-self.cbar * (r + self.delta_b)))
        
        return mnr
    
    
    
    def compute_norm_f_PE(self, l, l_dot):
        
        """
        Method that computes the passive force from the parallel element.
        See Eq (3.5) in Hatze's book.
        
        Parameters are not provided in Hatze's book, the ones used here were calculated by fitting the curves for the passsive force from Winters (2011) and Gollapudi (2009), used in Millard (2012).
        
        Parameters:
        -----------
        
        l : float
            muscle length
        
        l_dot : float
            muscle velocity
            
        
        Output:
        -----------
    
        norm_f_PE : float
            passive force of the parallel element
        
        """
        
        zeta = (l - self.l0) / self.l0
        zeta_dot = l_dot / self.l0
        
        norm_f_PE = self.k_PE * (np.exp(self.c_PE * zeta) - 1) + self.k_PE_p * zeta_dot
        
        return np.maximum(norm_f_PE, 0)
    
    
    def compute_delta(self, l, xi, lbda_T=0):
        
        """
        Method that computes the strain of the SE.
        See Eq (4.42) in Hatze's book.
        
        Parameters:
        -----------
        
        l : float
            muscle length
        
        xi : float
            normalized length of the CE (state variable)
            
        lbda_T : float (optional)
            tendon length
            
        
        Output:
        -----------
    
        delta : float
            strain of the SE
        
        """
        
        l_theta = np.sqrt(self.d ** 2 + (l - lbda_T) ** 2)
        delta = (l_theta - self.lbda_opt * (xi + self.kappa_) - self.lbda_S0) / (self.alpha_ * self.lbda_S0)
        
        return delta
    
    
    def compute_norm_f_SE(self, delta):
        
        """
        Method that computes the relative force output of the muscle model.
        See Eq (4.42) in Hatze's book.
        
        Parameters:
        -----------

        delta : float
            strain of the SE
            
        
        Output:
        -----------
    
        norm_f_SE : float
            relative force output of the SE
        
        """
        
        norm_f_SE = (np.exp(self.sigma * delta) - 1) / (np.exp(self.sigma) - 1)
        
        if self.lbda_T0 == 0:
            norm_f_SE = np.maximum(norm_f_SE, 0)
        
        return norm_f_SE
    
    
    def compute_norm_f_ST(self, lbda_T):
        
        """
        Method that computes the relative force output of the muscle model.
        See Eq (4.54) in Hatze's book.
        
        Parameters:
        -----------

        lbda_T : float
            tendon length
        
        Output:
        -----------
    
        norm_f_ST : float
            force output of the muscle model, relatively to the maximum isometric force in the direction of F_SE
        
        """
            
        norm_f_ST = np.sin(self.theta_b) * (np.exp(self.sigma * (lbda_T / self.lbda_T0 - 1) / self.alpha_) - 1) / (np.exp(self.sigma) - 1)
            
        return np.maximum(norm_f_ST, 0)
    
    
    
    def evaluate_xi0(self, l0, err=1e-6):
        
        """
        Method that runs the Newton method applied on f = norm_f_SE - b1 * (k - k1) to calculate the initial normalized length xi_0 of the CE.
        See Eq (4.43) in Hatze's book.
        
        Parameters:
        -----------

        l0 : float
            initial muscle length
            
        
        Output:
        -----------
    
        xi_0 : float
            initial normalized length of the CE
        
        """
        
        xi_0 = 0
        xi_1 = 1
    
        n = 0

        while np.abs(xi_1 - xi_0) > err:
            
            xi_0 = xi_1

            delta_0 = self.compute_delta(l0, xi_0, self.lbda_T0)
            norm_f_SE_0 = self.compute_norm_f_SE(delta_0)
    
            k = self.compute_k(xi_0)
            k1 = self.compute_k1(xi_0)
    
            f = norm_f_SE_0 - self.b1 * (k - k1)
    
            d_norm_f_SE = - self.sigma * self.lbda_opt * np.exp(delta_0 * self.sigma) / (self.alpha_ * self.lbda_S0 * (np.exp(self.sigma) - 1))
            
            d_k = 0.71 * (-1.112 * np.exp(-1.112 * (xi_0 - 1)) * np.sin(3.722 * (xi_0 - 0.656)) + 3.722 * np.exp(-1.112 * (xi_0 - 1)) * np.cos(3.722 * (xi_0 - 0.656)))
            d_k1 = - self.a6_p * np.exp(self.a6_p * (xi_0 - 1))

            d_f = d_norm_f_SE - self.b1 * (d_k - d_k1)
    
            xi_1 = xi_0 - f / d_f
    
            n += 1

            if n > 200:
                print("Break While loop for xi_0's evaluation : over 200 loops")
                return xi_1
            
        return  xi_1
    
    
    def evaluate_xi0_bis(self, l0, err=1e-6, a=0.8, b=1.2):
        
        """
        Method that runs the bissection method applied on f = norm_f_SE - b1 * (k - k1) to calculate the initial normalized length xi_0 of the CE.
        See Eq (4.43) in Hatze's book.
        
        Parameters:
        -----------

        l0 : float
            initial muscle length
        
        
        Output:
        -----------
    
        xi_0 : float
            initial normalized length of the CE
        
        """

        if self.lbda_T0 != 0:
            
            def f_root(xi_0):
        
                k = self.compute_k(xi_0)
                k1 = self.compute_k1(xi_0)
                
                if self.ignore_SE:
                    lbda_T = l0 - np.sqrt((self.lbda_opt * xi_0)**2 - self.d**2)
                    sin_theta = np.sqrt(1 - (self.d / (self.lbda_opt * xi_0))**2)
                    norm_f_0 = self.compute_norm_f_ST(lbda_T) / sin_theta
                else:
                    delta_0 = self.compute_delta(l0, xi_0, self.evaluate_lbda_T_bis(l0, xi_0))
                    norm_f_0 = self.compute_norm_f_SE(delta_0)
                
                if self.n_0 == 1:
                    eps = self.compute_S(self.n_0, 0, self.psi_0, 0, xi_0)[1]
                    return norm_f_0  - (eps * k - self.q0 * k1) / (1 - self.q0)
                else:
                    return norm_f_0  - self.b1 * (k - k1)
            
        else:
            
            def f_root(xi_0):

                k = self.compute_k(xi_0)
                k1 = self.compute_k1(xi_0)
                
                delta_0 = self.compute_delta(l0, xi_0, 0)
                norm_f_0 = self.compute_norm_f_SE(delta_0)
        
                return norm_f_0  - self.b1 * (k - k1)
            
        c = sp.optimize.bisect(f_root, 0.58, 1.8)
        
        return c
    
    
    
    def evaluate_lbda_T(self, l, xi, t, err=1e-6):
        
        """
        Method that runs the Newton method applied on f = norm_f_ST(lbda_T) - norm_f_SE(l, xi, lbda_T) * sin_theta(l, lbda_T) to calculate the tendon length lbda_T at date t.
        See Eq (4.54) and (4.55) in Hatze's book.
        
        Parameters:
        -----------

        l : float
            muscle length
        
        xi : float
            normalized length of the CE
        
        
        Output:
        -----------
    
        lbda_T : float
            tendon length
        
        """
        
        n = 0
        lbda_T_0 = self.lbda_T0
        lbda_T_1 = 1.05 * lbda_T_0
        
        while np.abs((lbda_T_1 - lbda_T_0) / lbda_T_0) > err:
            
            lbda_T_0 = lbda_T_1
            
            norm_f_ST_0 = self.compute_norm_f_ST(lbda_T_0)
            delta_0 = self.compute_delta(l, xi, lbda_T_0)
            norm_f_SE_0 = self.compute_norm_f_SE(delta_0)
            sin_theta_0 = (l - lbda_T_0) / np.sqrt(self.d**2 + (l - lbda_T_0)**2)
            
            f = norm_f_ST_0 - norm_f_SE_0 * sin_theta_0
            
            d_norm_f_ST_0 = np.sin(self.theta_b) * self.sigma * np.exp(self.sigma * (lbda_T_0 / self.lbda_T0 - 1) / self.alpha_) / (self.alpha_ * self.lbda_T0 * (np.exp(self.sigma) - 1))
            d_norm_f_SE_0 = - (self.sigma / (self.alpha_ * self.lbda_T0)) * (l - lbda_T_0) * np.exp(self.sigma * delta_0) / ((np.exp(self.sigma) - 1) * np.sqrt(self.d**2 + (l - lbda_T_0)**2))            
            d_sin_theta_0 = - self.d**2 / (self.d**2 + (l - lbda_T_0)**2)**1.5
            
            d_f = d_norm_f_ST_0 - (d_norm_f_SE_0 * sin_theta_0 + norm_f_SE_0 * d_sin_theta_0)
            
            lbda_T_1 = lbda_T_0 - f / d_f
            
            n += 1
            
            if n > 100:
                print("Break While loop for lbda_T's evaluation : over 100 loops", lbda_T_1, np.abs((lbda_T_1 - lbda_T_0) / lbda_T_0))
                return lbda_T_1
            
        return lbda_T_1


    def evaluate_lbda_T_bis(self, l, xi, err=1e-6, a=0.9, b=1.1):
        
        """
        Method that runs the bissection method applied on f = norm_f_ST(lbda_T) - norm_f_SE(l, xi, lbda_T) * sin_theta(l, lbda_T) to calculate the tendon length lbda_T at date t.
        See Eq (4.54) and (4.55) in Hatze's book.
        
        Parameters:
        -----------

        l : float
            muscle length
        
        xi : float
            normalized length of the CE
        
        
        Output:
        -----------
    
        lbda_T : float
            tendon length
        
        """
            
        def f_root(lbda_T):
            
            sin_theta = (l - lbda_T) / np.sqrt(self.d**2 + (l - lbda_T)**2)
            norm_f_ST = self.compute_norm_f_ST(lbda_T)
            delta = self.compute_delta(l, xi, lbda_T)
            norm_f_SE = self.compute_norm_f_SE(delta)
                
            return norm_f_ST - sin_theta * norm_f_SE
        
        
        a *= self.lbda_T0
        b *= self.lbda_T0
        
        try:
            c = sp.optimize.bisect(f_root, 0.6 * self.lbda_T0, 1.4 * self.lbda_T0)
        except ValueError:
            print('Dichotomy for lbda_T failed')
            c = self.lbda_T0
        
        return c
        
            
            
            



# =============================================================================
#     Dynamics computations
# =============================================================================
    
    
    def compute_n_dot(self, z):
        
        """
        Method that computes the right hand side of the recruitment dynamics equation for n (normalized population of stimulated MUs).
        See Eq (4.37) in Hatze's book.
        
        Parameters:
        -----------

        z : float
            normalized rate of motor unit recruitment (control parameter)
            
        
        Output:
        -----------
    
        n_dot : float
            derivative of the normalized population of stimulated MUs
        
        """
        
        n_dot = self.enco * z
        
        return n_dot


    def compute_r_dot(self, n, r, phi, z, w_p, w_m, mnr):
        
        """
        Method that computes the right hand side of the recruitment dynamics equation for r (normalized population of semi-active MUs).
        See Eq (4.37) in Hatze's book.
        
        Parameters:
        -----------
        
        n : float
            normalized population of stimulated MUs (state variable)
        
        r : float
            normalized population of semi-active MUs (state variable)
        
        phi : float
            pseudo Ca ion concentration of the semi-active population of MUs (state variable)

        z : float
            normalized rate of motor unit recruitment (control parameter)
        
        w_p, w_m : int
            switching parameters depending on the value of the control parameter z 
        
        mnr : float
            weighting parameter in the recruitment and excitation equations
            
        
        Output:
        -----------
    
        r_dot : float
            derivative of the normalized population of semi-active MUs
        
        """
        
        r_dot = -self.enco * z * (r - w_m * self.delta_b) / (r + self.delta_b) - (1 + w_m) * mnr * r / (0.001 * mnr + (phi / (self.k2 * self.c)) ** 2)
        
        return r_dot


    def compute_psi_dot(self, n, r, psi, phi, v, z, w_p, w_m, mn, mnr, rho0):
        
        """
        Method that computes the right hand side of the excitation dynamics equation for psi (pseudo Ca ion concentration of the stimulated population of MUs).
        See Eq (4.37) in Hatze's book.
        
        Parameters:
        -----------
        
        n : float
            normalized population of stimulated MUs (state variable)
        
        r : float
            normalized population of semi-active MUs (state variable)
        
        psi : float
            pseudo Ca ion concentration of the stimulated population of MUs (state variable)
        
        phi : float
            pseudo Ca ion concentration of the semi-active population of MUs (state variable)
        
        v : float
            normalized average stimulation rate of the population of stimulated MUs (control parameter)

        z : float
            normalized rate of motor unit recruitment (control parameter)
        
        w_p, w_m : int
            switching parameters depending on the value of the control parameter z 
        
        mn, mnr : float
            weighting parameters in the recruitment and excitation equations
        
        rho0 : float
            Ca density, accounting for the length dependency of the Ca ion concentration
            
        
        Output:
        -----------
    
        psi_dot : float
            derivative of the pseudo Ca ion concentration of the stimulated population of MUs
        
        """
        
        psi_dot = mn * (self.c * v - psi) + w_p * z * self.cbar * self.enco * (1 - np.exp(rho0 * (psi - phi))) / (rho0 * (1 - np.exp(-self.cbar * n - self.delta_b))) - (1 + w_m) * mnr * phi

        return psi_dot
    
    
    def compute_phi_dot(self, n, r, psi, phi, v, z, w_p, w_m, mn, mnr, rho0):
        
        """
        Method that computes the right hand side of the excitation dynamics equation for phi (pseudo Ca ion concentration of the semi-active population of MUs).
        See Eq (4.37) in Hatze's book.
        
        Parameters:
        -----------
        
        n : float
            normalized population of stimulated MUs (state variable)
        
        r : float
            normalized population of semi-active MUs (state variable)
        
        psi : float
            pseudo Ca ion concentration of the stimulated population of MUs (state variable)
        
        phi : float
            pseudo Ca ion concentration of the semi-active population of MUs (state variable)
        
        v : float
            normalized average stimulation rate of the population of stimulated MUs (control parameter)

        z : float
            normalized rate of motor unit recruitment (control parameter)
        
        w_p, w_m : int
            switching parameters depending on the value of the control parameter z 
        
        mn, mnr : float
            weighting parameters in the recruitment and excitation equations
        
        rho0 : float
            Ca density, accounting for the length dependency of the Ca ion concentration
            
        
        Output:
        -----------
    
        phi_dot : float
            derivative of the pseudo Ca ion concentration of the semi-active population of MUs
        
        """
        
        phi_dot = -mnr * phi - w_m * (mn * phi * ((self.c * v) / (psi + self.delta_b) - 1) - z * self.cbar * self.enco * (1 - np.exp(rho0 * (phi - psi))) / (rho0 * (np.exp(self.cbar * r + self.delta_b) - 1)))
        
        return phi_dot
    
    
    
    def compute_delta_eps_dot(self, delta_eps, xi, xi_dot, S):
        
        """
        Method that computes the right hand side of the differential equation defining stretch potentiation.
        See Eq (6.22) in Hatze's book.
        
        Parameters:
        -----------
        
        delta_eps : float
            stretch potentiation
            
        xi : float
            normalized length of the CE (state variable)
    
        xi_dot : float
            derivative of normalized length of the CE (normalized velocity of the CE)
        
        S : float
            factor that denotes the dependency on the distribution of stimulated and semi-active MUs of the contraction velocity
        
        
        Output:
        -----------
    
        delta_eps_dot : float
            temporal derivative of the relative stretch potentiation
        
        
        """
        
        if xi > 1:
            g = 1 - self.compute_k(xi) ** 1
        else:
            g = 0
        
        if xi_dot > 0:
            h = 1 / (self.b2 * (self.a1 + np.exp(-self.a2 * np.sinh(self.a3 * (xi_dot * S + 1/2))))) - 1 / (1 - self.q0)
        else:
            h = 0
        
        delta_eps_dot = self.d1 * (self.d2 * g * h - delta_eps)
        
        return delta_eps_dot
    
    
    
    def compute_xi_dot(self, xi, eps, S, l, l_pre, lbda_T, dt):
        
        """
        Method that computes the right hand side of the contraction dynamics equation for xi (normalized length of the CE).
        See Eq (4.37) in Hatze's book and MYOSIM report for computational exception cases.
        
        Parameters:
        -----------
        
        xi : float
            normalized length of the CE (state variable)
        
        eps : float
            excitation function
        
        S : float
            factor that denotes the dependency on the distribution of stimulated and semi-active MUs of the contraction velocity
        
        l : float
            muscle length
        
        lbda_T : float
            tendon length
        
        dt : float
            integrator step size
            
        
        Output:
        -----------
    
        xi_dot : float
            derivative of normalized length of the CE
        
        """
        
        k = self.compute_k(xi)
        k1 = self.compute_k1(xi)
        
        if self.lbda_T0 < lbda_T:
            self.slack_init = 0
        
        # if self.slack_init:
            
        #     # Millard initialization
        #     sin_theta = np.sqrt(1 - (self.d / (self.lbda_opt * xi))**2)
        #     d_FS = sin_theta * (self.sigma * (l - lbda_T) * np.exp(self.sigma * (np.sqrt(self.d**2 + (l - lbda_T)**2) - xi * self.lbda_opt - self.lbda_S0) / (self.lbda_S0 * self.alpha_))) / (self.alpha_ * self.lbda_S0 * np.sqrt(self.d**2 + (l - lbda_T)**2) * (np.exp(self.sigma) - 1))
        #     d_FT = np.sin(self.theta_b) * self.sigma * np.exp(self.sigma * (lbda_T / self.lbda_T0 - 1) / self.alpha_) / (self.alpha_ * self.lbda_T0 * (np.exp(self.sigma) - 1))      
            
        #     if lbda_T < self.lbda_T0 or np.abs(d_FS + d_FT) < 1e-6:
        #         xi_dot = 0
        #     else:
        #         l_dot = (l - l_pre) / dt
        #         xi_dot = d_FT * l_dot / (d_FT + d_FS)
        #     self.slack_init = 0
        #     print('millard')
        #     return xi_dot
        
        
        if self.slack_init:
            norm_f = 0
            self.slack_init = 0
        
        else:
            
            if self.ignore_SE:
                sin_theta = np.sqrt(1 - (self.d / (self.lbda_opt * xi))**2)
                norm_f = self.compute_norm_f_ST(lbda_T) / sin_theta
            else:
                norm_f = self.compute_norm_f_SE(self.compute_delta(l, xi, lbda_T))
        

        critical_norm_f_SE = max(0.999 * (eps * k / (self.a1 * self.b2) - self.b1 * k1), 0)
        
        if norm_f > critical_norm_f_SE:

            xi_dot = self.a7 * (self.nu_dot_star + (1 / self.w_bar) * ((norm_f + self.b1 * k1) / (eps * k) - 0.999 / (self.a1 * self.b2) - 0.001 * self.b1))
        
        elif norm_f + self.b1 * k1 < 0.00005:
            
            if eps < self.q0 + 0.001:

                xi_dot = 0
                
            else:

                xi_dot = 0.1 * self.a7 * eps
                
        elif -self.a1 + k * eps / (self.b2 * (norm_f + self.b1 * k1)) < 0:
            
            if xi < 1:

                xi_dot = max((1 - xi), 0.01) / dt
                
            else:

                xi_dot = - 0.01 / dt
                
        else:

            xi_dot = (1/S) * (-0.5 + np.arcsinh(-np.log(-self.a1 + (eps * k) / (self.b2 * (norm_f + self.b1 * k1))) / self.a2) / self.a3)

        return xi_dot
        
    
    
    
    
    
    # =============================================================================
    #     Metabolic cost dynamics
    # =============================================================================
    
    def compute_E_dot(self, xi_dot, n, xi, eps, v, l, l_dot):
        
        """
        Method that computes the right hand side of the metabolic energy expense equation.
        See Eq (7.45) in Hatze's book.
        
        Parameters:
        -----------
        
        xi_dot : float
            normalized contraction velocity of the CE
        
        n : float
            normalized population of stimulated MUs (state variable)
        
        xi : float
            normalized length of the CE (state variable)
        
        eps : float
            excitation function
        
        v : float
            normalized average stimulation rate of the population of stimulated MUs
        
        l : float
            muscle length
        
        l_dot : float
            muscle length variation
            
        
        Output:
        -----------
    
        E_dot : float
            total energy rate of the muscular contraction
        
        """
        
        norm_f = self.compute_norm_f_SE(self.compute_delta(l, xi))
        k = max(self.compute_k(xi), self.phi_b)
        
        E_dot = self.G * (self.gh * (eps * (k - self.phi_b) + (np.exp(self.cbar * n) - 1) * self.phi_b * v * (1 - np.exp(-self.kappa8 - self.kappa7 / (self.max_stim_rate * v))) / ((np.exp(self.cbar) - 1) * (1 - np.exp(-self.kappa8 - self.kappa7 / self.max_stim_rate)))) - xi_dot * 0.001 * self.f_b * (self.aF_ratio * eps * k + norm_f)) + self.cv * l_dot**2
        
        return E_dot 
    
    
    
    
    def compute_dynamics(self, dt, get_l, get_v, get_z):
        
        """
        Method that computes the right-hand side of the dynamics system of model 3 in the format required for the scipy.integrate.solve_ivp function.
        
        Parameters:
        -----------
        
        dt : float
            integrator step size
        
        get_l : callable (input: t, ouput: l(t))
            muscle length function
        
        get_v : callable (input: t, ouput: v(t))
            normalized average stimulation rate function (control parameter)
        
        get_z : callable (input: t, ouput: z(t))
            normalizd average recruitment rate function (control parameter)
        

        Output:
        -----------
    
        dynamics_system : callable
            function that computes the dynamics system in the format required to use scipy.integrate.solve_ivp
                input: (t,y)
                output: [n_dot, r_dot, psi_dot, phi_dot, xi_dot]
        
        """
        
        def dynamics_system(t, y, dt, get_l, get_v, get_z):

            n, r, psi, phi, xi, delta_eps = y
            
            n = min(n, 1)
            r = min(r, 1)
            
            l, l_pre = get_l(t), get_l(t-dt)
            z = get_z(t)
            if n >= 1 and z > 0:
                z = 0
            if n <= 0 and z < 0:
                z = 0
            v = get_v(t)

            w_p = 0
            w_m = 0
            if z > 0:
                w_p = 1
            if z < 0:
                w_m = -1

            mn  = self.compute_mn(n)
            mnr = self.compute_mnr(n, r)    
            if self.lbda_opt_shift:
                rho0 = self.compute_rho0(1) * xi
            else:
                rho0 = self.compute_rho0(xi)
            S, eps = self.compute_S(n, r, psi, phi, xi, delta_eps)
            
            if self.lbda_T0 != 0:
                if self.ignore_SE:                    
                    lbda_T = l - np.sqrt((self.lbda_opt * xi)**2 - self.d**2)
                else:
                    lbda_T  = self.evaluate_lbda_T_bis(l, xi)
            else:
                lbda_T = 0
            
            if n >= 1 and z > 0:
                n_dot = 0
                r_dot = 0
            elif n <= 0 and z < 0:
                n_dot = 0
                r_dot = 0
            else:
                n_dot     = self.compute_n_dot(z)
                r_dot     = self.compute_r_dot(n, r, phi, z, w_p, w_m, mnr)
            psi_dot       = self.compute_psi_dot(n, r, psi, phi, v, z, w_p, w_m, mn, mnr, rho0)
            phi_dot       = self.compute_phi_dot(n, r, psi, phi, v, z, w_p, w_m, mn, mnr, rho0)
            xi_dot        = self.compute_xi_dot(xi, eps, S, l, l_pre, lbda_T, dt)
            if self.stretch_potentiation:
                delta_eps_dot = self.compute_delta_eps_dot(delta_eps, xi, xi_dot, S)
            else:
                delta_eps_dot = 0

            return [n_dot, r_dot, psi_dot, phi_dot, xi_dot, delta_eps_dot]
        
        return dynamics_system
        
    
    
    
    def integrate(self, dydt, t, y0, dt, params=None, events=None):
        
        """
        Integration function using scipy.integrate.solve_ivp method
        
        Parameters:
        -----------

        dydt : function
            right side of the differiential equation (scipy.integrate.solve_ivp format)
        
        t : Numpy array of shape (N,)
            time vector
        
        y0 : Numpy array with the same shape as the ouput of dydt
            initial state to solve the differential equation
        
        dt : float
            maximal integrator step size
        
        params : tuple (optional)
            additional parameters for dydt
        
        events : tuple of functions (optional)
            events to process during integration
            
        
        Output:
        -----------
    
        output : object
            output of integration with scipy.integrate.solve_ivp
        
        """
        
        t_span = (t[0], t[-1])
        options = {'max_step': dt}

        output = sp.integrate.solve_ivp(dydt, t_span, y0, t_eval=t, events=events, args=params, **options)
        
        return output
    
    
    
    
    
    
    def simulate(self, filename, t, get_l, get_v, get_z, l_init, N=10000, write_output=0, ignore_SE=0, icbel=0, n_0=0, psi_0=0, lbda_opt_shift=0, potentiation=0):
        
        """
        Simulation of a stimulation test
        
        Parameters:
        -----------
        
        filename : str
            path of the file to write the model output
        
        t : Numpy array of shape (N,)
            time vector
        
        get_l : callable (input: t, ouput: l(t))
            muscle length function
        
        get_v : callable (input: t, ouput: v(t))
            normalized average stimulation rate function (control parameter)
        
        get_z : callable (input: t, ouput: z(t))
            normalizd average recruitment rate function (control parameter)
        
        l_init : float
            length of the muscle before the beginning of the simulation
        
        N : int (optional, default is 10000)
            number of integration steps
        
        write_output : bool (optional, default is 0)
            if 1 the output of the simulation is exported in a .csv file with the name filename, if 0 no file is created
        
        ignore_SE : bool (optional, default is 0)
            if 1 the series elastic element is ignored according to pages 69-70 (chapter 5)
        
        icbel : bool (optional, deault is 0)
            if 1 the independence of interfilamentary velocity of the number of cross-bridges attached to the actin complex is assumed
        
        n_0 : float (optional, default is 0)
            initial proportion of recruited motor units
            
        psi_0 : float (optional, default is 0)
            initial pseudo Ca concentration of the population of active motor units
        
        lbda_opt_shit : bool (optional, default is 0)
            if 1 the linear rho0 function from Rockenfeller (2018) is used
        
        potentiation : bool (optional, default is 0)
            if 1 the stretch potentiation ODE enhancing the active state is used
        
        
        Output:
        -----------
    
        res : Pandas DataFrame of shape (13, N)
            "t"         : time vector (DataFrame index)
            "n(t)"      : normalized population of stimulated MUs
            "r(t)"      : normalized population of semi-active MUs
            "xi(t)"     : normalized length of the CE
            "psi(t)"    : pseudo Ca ion concentration of stimulated populations
            "phi(t)"    : pseudo Ca ion concentration of semi-active populations
            "q(n)(t)"   : active state of the population of stimulated MUs
            "q(r)(t)"   : active state of the population of semi-active MUs
            "epsilon(t)": muscle excitation
            "F_TOT(t)"  : total normalized force output
            "F_ST(t)"   : normalized force output of the tendon element
            "l(t)"      : length of the muscle-tendon unit
            "E_rate(t)" : rate of energy expenditure of the muscle
        
        """
        
        self.lbda_opt_shift = lbda_opt_shift
        self.ignore_SE = ignore_SE
        self.slack_init = 0
        self.stretch_potentiation = potentiation
        
        self.n_0 = n_0
        self.psi_0 = psi_0
        
        self.icbel = icbel
        if self.icbel == 0:
            self.kappa_ = 0
            if self.l0 != 0 and l_init < self.l0:    # the muscle is always considered inactive initially
                self.xi_0 = 1
                self.slack_init = 1
            else:
                self.xi_0 = self.evaluate_xi0_bis(l_init)
        else:
            self.lbda_BE = np.log(1 + (np.exp(self.sigma) - 1) / (1 - self.q0)) / (self.sigma / (self.kappa_ * self.lbda_opt))
            if l_init < 0.97061 * self.lbda_opt + self.lbda_BE + self.lbda_S0:
                self.xi_0 = 0.97061
            else:
                self.xi_0 = self.evaluate_xi0_bis(l_init)
                
        
        dt = (t[-1] - t[0]) / N

        dydt = self.compute_dynamics(dt, get_l, get_v, get_z)
        
        y0 = [self.n_0, 0, psi_0, 0, self.xi_0, 0]
        # print('n(0), r(0), psi(0), phi(0), xi(0), d_eps(0)', y0)
        
        if self.lbda_T0 != 0:
            lbda_T_prec = self.evaluate_lbda_T_bis(l_init, self.xi_0)
            # print('lbda_T_0', lbda_T_prec / self.lbda_T0)
        
        params = [dt, get_l, get_v, get_z]
        sol = self.integrate(dydt, t, y0, dt, params=params)        

        n, r, psi, phi, xi, delta_eps = sol.y

        l = np.array([get_l(tt) for tt in t])

        if self.lbda_T0 != 0:
            if self.ignore_SE:
                lbda_T = np.array([l[i] - np.sqrt((self.lbda_opt * xi[i])**2 - self.d**2) for i in range(t.shape[0])])
                sin_theta = np.sqrt(1 - (self.d / (self.lbda_opt * xi))**2)
                norm_f_SE = self.compute_norm_f_ST(lbda_T) / sin_theta
            else:
                lbda_T = np.array([self.evaluate_lbda_T_bis(l[i], xi[i]) for i in range(t.shape[0])])
                norm_f_SE = self.compute_norm_f_ST(lbda_T)
        else:
            norm_f_SE = self.compute_norm_f_SE(self.compute_delta(l, xi))
        
        if self.l0 != 0:
            l_dot = np.hstack((0, np.diff(l))) / dt
            norm_f_PE = self.compute_norm_f_PE(l, l_dot)
        else:
            norm_f_PE = 0
            
        norm_F = norm_f_SE + norm_f_PE
        
        eps = self.compute_S(n, r, psi, phi, xi, delta_eps)[1]
        q_n = self.compute_q(xi, psi)
        q_r = self.compute_q(xi, phi)
        
        l_dot = np.hstack((0, np.diff(l))) / dt
        xi_dot = np.hstack((0, np.diff(xi))) / dt
        v = np.array([get_v(t_i) for t_i in t])
        E_rate = np.array([self.compute_E_dot(xi_dot[i], n[i], xi[i], eps[i], v[i], l[i], l_dot[i]) for i in range(t.shape[0])])
        
        data = {'n(t)': n, 'r(t)': r, 'xi(t)': xi, 'psi(t)': psi, 'phi(t)': phi, 'q(n)(t)': q_n, 'q(r)(t)': q_r, 'epsilon(t)': eps, 'F_TOT(t)': norm_F, 'F_ST(t)': norm_f_SE, 'l(t)': l, 'E_rate(t)': E_rate}
        res = pd.DataFrame(data, index=t)
        res.index.name = 't'
        
        if write_output:
            res.to_csv(path_or_buf=filename, sep=',', index=True, float_format='%.5f')
        
        return res
        

            

# =============================================================================
#   Execution
# =============================================================================
    
if __name__ == '__main__':
    
    pass
    
# =============================================================================
#   Maximal activation Benchmark Trial
# =============================================================================
    
    # # Muscle parameters
    
    # name = 'Rat soleus'
    # alpha0 = 6
    # L0_M = 0.0171
    # LT_slack = L0_M
    # LT_opt = LT_slack * 1.08
    # F0_max = 1.17
    # fibre_type = 'slow'
    
    # l_opt = L0_M * np.cos(alpha0 * np.pi / 180) + LT_opt
    # theta = np.pi/2 - alpha0 * np.pi / 180
    # l0 = L0_M * np.cos(alpha0 * np.pi / 180) + LT_slack
    
    # muscle = Hatze_Muscle(name, l_opt, fibre_type, theta=theta, lbda_T0=LT_slack, l0=l0)
    
    # n_0 = 1
                         
    # # Define control parameters profiles
    
    # def get_v(t):
    #     return 2.5
    
    # if n_0 == 1:
    #     psi_0 = get_v(0) * muscle.c    # Asymptotic value for psi during maximal activation, used here if we consider that all MUs are already recruited initially
    # else:
    #     psi_0 = 0
    
    # def get_z(t):
    #     if t > 0 and t <= 0.06993373 * (1 - n_0):
    #         return 1
    #     else:
    #         return 0
    
    # from millard_benchmarks import read_data
    
    # N = 10000
    # path = '..\\biologicalBenchmark\\maximalActivation\\'
    # t, displacement, t_exp, displacement_exp = read_data(path + 'displacement.dat', 'displacement_M', N)
    # displacement_offset = -0.002
    
    # # t = t[:N//4]
    # # displacement = displacement[:N//4]
    
    # l_init = Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset
    
    # def get_l(t):
    #     i = max(round(N*t/2) - 1, 0)
    #     Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + 2 * 0.001 * displacement[i]
    #     return Lmt
        
    
    # res = muscle.simulate('output_Hatze_Model_3.csv', t, get_l, get_v, get_z, l_init, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0)    
    
    # exp_data = read_data(path + 'force_trial6.dat', 'force_M', N)[1] / F0_max
    
    # plt.figure("Maximal activation benchmark - Trial with +/- 2 mm displacement")
    # plt.plot(t, exp_data, label='Experimental')
    # plt.plot(t, res['F_TOT(t)'], label="Hatze with Millard's initialization - n(t=0) = " + str(n_0) + ' - psi(t=0) = ' + str(psi_0))
    # plt.ylabel("Normalized force")
    # plt.xlabel("Time (s)")
    # plt.xlim((0, 2))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    
# =============================================================================
#     Sensitivity plot
# =============================================================================
    
    # N = 10000
    # t = np.linspace(0, 1, N)
    
    # a3  = [4.8, 4.0, 3.2, 2.8, 2.5]
    # fCf = [1.15, 1.25, 1.33, 1.45, 1.60]
    # enco = [11, 14.3, 17, 20]
    # a2_p = [0.08, 0.10, 0.14, 0.17, 0.2]
    # a3_p = [0.03, 0.06, 0.11, 0.15, 0.19]
    # lbda_S0_ratio = [0.02, 0.045, 0.07, 0.09, 0.11]
    # lbda_S0_factor = [0.55, 0.6, 0.65, 0.7, 0.75]
    # alpha_ = [0.02, 0.05, 0.08, 0.11, 0.14]
    
    
    # plt.figure("Muscle Force output")
    
    # for i in range(len(lbda_S0_factor)):
        
    #     muscle.lbda_S0 = 0.6 * muscle.l_opt / (1 + lbda_S0_factor[i] * muscle.alpha_)
    #     res = muscle.simulate(t, get_l, get_v, get_z)
    #     plt.plot(t, res["F_TOT(t)"], label='lbda_S0 = 0.6*l_opt / (1+' + str(lbda_S0_factor[i]) + '*alpha_bar)')

    # plt.ylabel("Normalized force")
    # plt.xlabel("Time (s)")
    # plt.ylim((0, 1.2))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    
    
# =============================================================================
#     Plots
# =============================================================================


    # plt.figure("Metabolic cost")
    # plt.plot(t, res[6], label='Metabolic cost')
    # plt.plot(t, res[7], label='Energy rate')
    # plt.ylabel("Energy")
    # plt.xlabel("Time (s)")
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure("Muscle Force output")
    # plt.plot(t, res[5], label='Force output')
    # # plt.plot(t, res[4], label='Normalized fibre length')
    # plt.ylabel("Force")
    # plt.xlabel("Time (s)")
    # plt.ylim((0, 1.1))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure()
    # plt.plot(t, res[0], label='n(t) : Active population')
    # plt.plot(t, res[1], label='r(t) : Semi-active population')
    # plt.ylabel("Output")
    # plt.xlabel("Time (s)")
    # plt.ylim((0, 1.05))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure()
    # plt.plot(t, res[2], label='psi(t) : active pseudo [Ca]')
    # plt.ylabel("Output")
    # plt.xlabel("Time (s)")
    # plt.ylim((0, 0.00015))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure()
    # plt.plot(t, res[3], label='phi(t) : semi-active pseudo [Ca]')
    # plt.ylabel("Output")
    # plt.xlabel("Time (s)")
    # plt.ylim((0, 0.00015))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.figure()
    # plt.plot(t, res[4], label='xi(t) : normalized CE length')
    # plt.ylabel("Output")
    # plt.xlabel("Time (s)")
    # plt.ylim((-1.05, 1.05))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    
    # z = np.array([get_z(t_i) for t_i in t])
    # plt.figure()
    # plt.plot(t, z, label='z(t) : normalized recruitment rate')
    # plt.ylabel("Output")
    # plt.xlabel("Time (s)")
    # plt.ylim((-1.05, 1.05))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # v = np.array([get_v(t_i) for t_i in t])
    # plt.figure()
    # plt.plot(t, v, label='v(t) : normalized average stimulation rate')
    # plt.ylabel("Output")
    # plt.xlabel("Time (s)")
    # plt.ylim((0, 1.05))
    # plt.xlim((0, 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    