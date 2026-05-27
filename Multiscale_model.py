"""
Author: Andrea Sgarzi
Email: a.sgarzi@ad.unsw.edu.au
Affiliation: University of New South Wales (UNSW), Graduate School of Biomedical Engineering (GSBE)

Description:
This module implements a single-actuator-Hill-type muscle model for simulating force production 
in skeletal muscles and MUs differentiating between slow and fast fibres.

Key features of the model include:
- MU-level excitation-activation dynamics driven by discharge times
- Second-order calcium transient dynamics for slow and fast fibres
- Hill-type muscle mechanics with configurable components:
  - Tendon compliance and pennation
  - Passive elastic element (PE)
  - Force-length (FL) and force-velocity (FV) relationships
  - Yielding (slow fibres) and sag (fast fibres)
- Flexible configuration via a ModelConfig object

The model supports simulations at different scales:
- Muscle scale ("M")
- Motor-unit scale ("MU")
- Calcium-transient scale ("Ca")
"""

from __future__ import annotations  

from dataclasses import dataclass  
from typing import Dict, List, Any  
import numpy as np  
from scipy.integrate import solve_ivp  

# =============================================================================
# Data containers
# =============================================================================

@dataclass  
class Params:  

    # Required
    time: np.ndarray            # time vector
    dt: float                   # time resolution (s)
    muscle: str                 # e.g. 'rat_SOL', 'rat_EDL', 'cat_SOL', 'cat_GM', 'cat_CF'
    scale: str                  # 'Muscle', 'MU', 'Ca'
    MVC: float                  # maximum isometric force [N]
    vmax: float                 # maximum contraction velocity [l0/s] (later scaled by l_M_opt)
    alpha_0: float              # initial pennation angle [rad]
    l_MT: np.ndarray            # muscle-tendon total length (len(time) or len(time)+1 accepted)
    l_M_opt: float              # optimal fibre length (same units as l_MT)
    l_T_slack: float            # tendon slack length (same units as l_MT)

    # Optional defaults
    Ca_max_s_M: float = 324382  # activation (slow, muscle scale)
    Ca_max_s_MU: float = 276339 # activation (slow, MU scale)
    k1_s_M: float = 10.6        # activation kinetics
    k2_s_M: float = 15.6        # activation kinetics
    k1_s_MU: float = 16.7       # activation kinetics
    k2_s_MU: float = 18.3       # activation kinetics 
    Ca_max_f_M: float = 192770  # activation (fast, muscle scale)
    Ca_max_f_MU_catGM: float = 592607 # activation (fast, MU scale, cat GM)
    Ca_max_f_MU_ratGM: float = 627772 # activation (fast, MU scale, rat GM)
    k1_f_M: float = 10          # activation kinetics
    k2_f_M: float = 10.28       # activation kinetics 
    k1_f_MU_catGM: float = 11.02       # activation kinetics
    k2_f_MU_catGM: float = 12.92      # activation kinetics
    k1_f_MU_ratGM: float = 10       # activation kinetics
    k2_f_MU_ratGM: float = 63.82      # activation kinetics
    c1_s: float = 30605         # calcium kinetics (slow)
    c2_s: float = 896181        # calcium kinetics (slow)
    c3_s: float = 2.0           # calcium kinetics (slow)
    c1_f: float = 2056          # calcium kinetics (fast)
    c2_f: float = 467405        # calcium kinetics (fast)
    c3_f: float = 0.435         # calcium kinetics (fast)
    af_s: float = 0.419         # FV curvature (slow)
    af_f: float = 0.361         # FV curvature (fast)
    As_peak: float = 1.6        # sag peak
    As_decay: float = 0.87       # sag decay
    Ts: float = 0.097            # sag time constant

    def __post_init__(self):  

        self.vmax = self.vmax * self.l_M_opt  # convert vmax from l0/s to length units per second

        if self.muscle in {"cat_SOL", "rat_SOL"}:  # tendon stiffness is muscle-specific
            self.eps_0 = 0.06  # Soleus literature (see paper)
        elif self.muscle in {"rat_EDL", "cat_MG", "cat_CF", "rat_MG"}:  # other muscle group
            self.eps_0 = 0.04  # Thelen-type stiffness (see paper notes)
        else:  
            self.eps_0 = 0.06  # animal soleus literature


@dataclass  
class States:  
    MUAP_0: float  # MUAP amplitude (beta)
    Ca_0: float    # Ca concentration
    act_0: float   # activation 
    l_M_0: float   # muscle fibre length (only used if tendon system)
    y_0: float     # yielding (only used if enabled)
    s_0: float     # sag (only used if enabled)


# =============================================================================
# Utility functions (to be extended when recruitment will be included)
# =============================================================================

def compute_fs(distimes: np.ndarray) -> float:  # compute mean discharge frequency from discharge times
    distimes = np.atleast_1d(distimes).ravel()  # ensure 1D array
    if distimes.size >= 2:  # if at least 2 spikes exist
        return float(np.mean(1.0 / np.diff(distimes)))  # mean instantaneous frequency
    return 1.0  # default for twitch / single-spike case


# =============================================================================
# Mechanics (SE, PE, FL, FV, pennation)
# =============================================================================

class Mechanics:  
    def __init__(self, P: Params):  
        self.P = P  # keep params for tendon/FV constants

    def tendon_force(self, eps: float) -> float:  

        """
        Computes normalized tendon force as a function of tendon strain (from John et al. 2013).
        Inputs:
        - eps: float, tendon strain (dimensionless), eps = (l_T - l_T_slack)/l_T_slack
        Outputs:
        - f_SE: float, normalized tendon force (dimensionless)
        """
        
        eps_0 = self.P.eps_0  # species/muscle dependent reference strain
        klin = 1.712 / eps_0  # linear stiffness slope 
        eps_toe = 0.609 * eps_0  # toe region end strain
        F_toe = 0.33  # normalized force at toe end
        k_toe = 3.0  # toe curvature

        if eps > eps_toe:  # linear region
            return 0.001 * (1 + eps) + (klin * (eps - eps_toe) + F_toe)  
        elif eps > 0:  # toe region
            return 0.001 * (1 + eps) + (F_toe * ((np.exp(k_toe * eps / eps_toe) - 1) / (np.exp(k_toe) - 1)))  
        else:  # slack
            return 0.001 * (1 + eps) 
        

    @staticmethod
    def passive_pe(l_M_norm: float) -> float:  

        """
        Computes normalized passive force-length contribution (PE) from normalized fibre length (from Thelen 2003).
        Inputs:
        - l_M_norm: float, normalized fibre length (l_M / l_M_opt)
        Outputs:
           - f_PE: float, normalized passive force (dimensionless) 
        """
        
        kPE, eps0 = 5.0, 0.6  # shape parameters
        if l_M_norm < 1.0:  # below optimal length
            return 0.0  # no passive force
        return (np.exp((kPE * (l_M_norm - 1)) / eps0) - 1) / (np.exp(kPE) - 1)  
    

    @staticmethod
    def pennation(l_M: float, l_M_0: float, alpha0: float) -> float:

        """
        Computes the pennation angle alpha given MT length, tendon length and initial geometry.
        Inputs:
        - l_M: fibre length
        - l_M_0: float, initial fibre length used to compute constant width term.
        - alpha0: float, initial pennation angle.
        Outputs:
        - alpha: float, updated pennation angle (radians), clipped to avoid singularities.
        """
        
        w = l_M_0 * np.sin(alpha0) # constant muscle width term
        sin_alpha = w / max(l_M, 1e-9) # compute sin(alpha) from geometry
        sin_alpha = float(np.clip(sin_alpha, 0.0, np.sin(1.4706289))) # clipped at ~84.3 deg to avoid singularities
        return float(np.arcsin(sin_alpha)) # compute alpha


    @staticmethod
    def force_length(act: float, l_M_norm: float) -> float:  

        """
        Computes the active force-length scaling factor FL(act, l_M_norm) (from Lloyd Besier 2003).
        Inputs:
        - act: float, activation state (0..1).
        - l_M_norm: float, normalized fibre length (l_M / l_M_opt).
        Outputs:
        - FL: float, force-length scaling factor (dimensionless).
        """
        
        a = 0.45  # width
        b = (0.15 * (1 - act)) + 1  # activation-dependent shift
        return float(np.exp(-((l_M_norm - b) / a) ** 2))  


    def fv_velocity(self, act, l_M_norm, f_CE_over_FL, FL, fibre_type, vmax) -> float:  

        """
        Inverts the FV relationship to obtain fibre velocity given normalized CE force ratio (modified from Caillet 2023 PhD thesis).
        Inputs:
        - act: float, activation state (0..1).
        - l_M_norm: float, normalized fibre length.
        - f_CE_over_FL: float, normalized CE force divided by FL (dimensionless).
        - FL: float, force-length scaling.
        - fibre_type: str, "slow" or "fast".
        - vmax: float, maximum contraction velocity in absolute units (already scaled by l_M_opt).
        Outputs:
        - v_M: float, fibre velocity in absolute units per second.
        """
        
        fmax = 1.4  # max eccentric factor
        fv = 0.25 + 0.75 * act  # activation-dependent scaling

        if fibre_type == "slow":  # slow fibres curvature
            kMU = 0.2  # curvature scaling
            af = self.P.af_s  # slow af
        else:  # fast fibres curvature
            kMU = 1.0  # curvature scaling
            af = self.P.af_f  # fast af

        g = FL if l_M_norm < 1 else 1.0  # length dependence for shortening
        af = max(float(af), 1e-8) # clamp
        b = (fmax - 1) / (2 + 2 / af)  # Hill b parameter
        K = kMU * g * fv  # effective scaling
        K = max(float(K), 1e-12) # clamp

        eps = 1e-6  
        f = float(np.clip(f_CE_over_FL, eps, fmax - eps))  # clamp force ratio into valid range
        if f >= 1:  # eccentric / >= isometric
            vel = b * ((f - 1) / (fmax - f))  
            vel *= K  
        else:  # concentric
            vel = (f - 1) / (f / (af * K))  # concentric branch

        return float(vel * vmax)  


    def fv_force(self, act: float, v_norm: float, FL: float, l_M_norm: float, fibre_type: str) -> float:  

        """
        Computes the FV scaling factor given normalized velocity and other context (modified from Caillet 2023 PhD thesis).
        Inputs:
        - act: float, activation state (0..1).
        - v_norm: float, normalized velocity (unitless; consistent with your conventions).
        - FL: float, FL scaling factor (may be 1.0 if disabled).
        - l_M_norm: float, normalized fibre length.
        - fibre_type: str, "slow" or "fast".
        Outputs:
        - FV: float, force-velocity scaling factor (dimensionless).
        """
        
        fmax = 1.4  # max eccentric factor
        fv = 0.25 + 0.75 * act  # activation-dependent scaling

        if fibre_type == "slow":  # slow curvature
            kMU = 0.2  # curvature scaling
            af = self.P.af_s  # slow af
        else:  # fast curvature
            kMU = 1.0  # curvature scaling
            af = self.P.af_f  # fast af

        g = FL if l_M_norm < 1 else 1.0  # length dependence for shortening
        af = max(float(af), 1e-8) # clamp
        b = (fmax - 1) / (2 + 2 / af)  # Hill b parameter
        K = kMU * g * fv  # effective scaling
        K = max(float(K), 1e-12) # clamp

        if v_norm < 0:  # shortening 
            return float(1 / (1 - (v_norm / (af * K))))  
        else:  # lengthening / eccentric
            return float((1 + fmax * (v_norm / (K * b))) / (1 + v_norm / (K * b)))  


# =============================================================================
# Electrophysiology (Ca2+, activation, sag & yield)
# =============================================================================

class Ephys:  
    def __init__(self, P: Params):  
        self.P = P  # keep params for calcium/activation constants

    @staticmethod
    def _is_firing(t_round: float, AP_times: np.ndarray, prec: float) -> int:  

        """
        Checks whether a rounded time instant corresponds to a discharge time (Caillet et al. 2023).
        Inputs:
        - t_round: float, rounded time in seconds.
        - AP_times: np.ndarray, spike times in seconds.
        - prec: float, discretization precision in seconds.
        Outputs:
        - is_fire: int, 1 if firing occurs at that time, 0 otherwise.
        """
        
        t_int = int(t_round / prec + 1e-4)  # convert rounded time to integer grid
        return int(t_int in (AP_times / prec).astype(int))  
    

    def MN_AP(self, t: float, AP_times: np.ndarray, V_N: float = 90.0) -> float:  

        """
        Produces a simplified motoneuron action potential waveform (half-sine) at firing times (Caillet et al. 2023).
        Inputs:
        - t: float, current time in seconds.
        - AP_times: np.ndarray, spike times in seconds.
        - V_N: float, peak membrane potential (default 90).
        Outputs:
        - V: float, membrane potential at time t (0 if not within AP window).
        """
        
        prec = 1e-3  # temporal precision (larger than AP duration)
        t_round = int(t / prec + 1e-4) * prec  # round time to grid
        if not self._is_firing(t_round, AP_times, prec):  # if no discharge at this time
            return 0.0  # membrane potential remains zero

        sin_period = 1.4e-3  # AP sine waveform period
        t_end = t_round + sin_period / 2  # AP end time
        if t <= t_end:  # during AP
            return float(V_N * np.sin(2 * np.pi / sin_period * (t - t_round)))  # sine value
        return 0.0  # after AP


    def MU_AP_2nd(self, t: float, AP_times: np.ndarray, beta: float, dbeta: float) -> float:  

        """
        Computes the second derivative of MUAP state beta using a linear 2nd order ODE (Caillet et al. 2023).
        Inputs:
        - t: float, current time.
        - AP_times: np.ndarray, spike times.
        - beta: float, MUAP state.
        - dbeta: float, MUAP first derivative.
        Outputs:
        - DDbeta: float, MUAP second derivative.
        """
        
        b1, b2, b3 = 2e4, 5e7, 9e7  # coefficients
        return b3 * self.MN_AP(t, AP_times) - b2 * beta - b1 * dbeta + 1e-50  # ODE


    def Ca_2nd(self, l_norm: float, fibre_type: str, beta: float, Ca: float, dCa: float) -> float:  

        """
        Computes second derivative of Ca state using length-dependent amplitude/width 
        and fibre-type coefficients (modified from Hatze 1977 and Caillet et al. 2023).
        Inputs:
        - l_norm: float, normalized fibre length.
        - fibre_type: str, "slow" or "fast".
        - beta: float, MUAP state.
        - Ca: float, Ca state.
        - dCa: float, Ca first derivative.
        Outputs:
        - DDCa: float, Ca second derivative.
        """
        
        if fibre_type == "slow":  # select slow coefficients
            c1, c2, c3 = self.P.c1_s, self.P.c2_s, self.P.c3_s  
        else:  # select fast coefficients
            c1, c2, c3 = self.P.c1_f, self.P.c2_f, self.P.c3_f 

        if l_norm < 1:  # amplitude fit region 1
            amp = 0.8  
        elif l_norm <= 1.1379:  # amplitude fit region 2
            amp = 1.3947 * l_norm - 0.5871  
        elif l_norm < 1.239:  # amplitude fit region 3
            amp = 1.0  # plateau
        else:  # amplitude fit region 4
            amp = 1 - 0.4623 * (l_norm - 1.239)  

        p2 = [0.3783, -0.8320, 1.1885]  # polynomial coefficients for width
        width = (l_norm ** 2) * p2[0] + l_norm * p2[1] + p2[2]  # compute width

        if l_norm < 1.23:  
            width = 0.738  
        elif l_norm > 2.04: 
            width = 1.072 

        return amp * c3 * beta - width * c1 * dCa - (c2 * width**2) * Ca  # ODE


    def activation_dot_from(self, Ca: float, act: float, fibre_type: str) -> float: 

        """
        Computes d(act)/dt given Ca_norm, current act, and fibre_type, with 
        scale-dependent parameters (modified from Hussein et al. 2022).
        Inputs:
        - Ca_norm: float, normalized Ca state.
        - act: float, activation state.
        - fibre_type: str, "slow" or "fast".
        Outputs:
        - dact: float, activation derivative.
        """
        
        P = self.P  # keep parameters

        if fibre_type == "slow" and P.scale in {"Muscle", "Ca_transients"}:  # slow muscle scale
            Ca_norm = Ca * P.Ca_max_s_M  # scale normalized Ca
            k1, k2 = P.k1_s_M, P.k2_s_M  # kinetics
        elif fibre_type == "fast" and P.scale in {"Muscle", "Ca_transients"}:  # fast muscle scale
            Ca_norm = Ca * P.Ca_max_f_M  
            k1, k2 = P.k1_f_M, P.k2_f_M  
        elif fibre_type == "slow" and P.scale == "MU":  # slow MU scale
            Ca_norm = Ca * P.Ca_max_s_MU  
            k1, k2 = P.k1_s_MU, P.k2_s_MU  
        elif fibre_type == "fast" and P.scale == "MU" and P.muscle == "cat_MG":  # fast MU scale
            Ca_norm = Ca * P.Ca_max_f_MU_catGM  
            k1, k2 = P.k1_f_MU_catGM, P.k2_f_MU_catGM  
        elif fibre_type == "fast" and P.scale == "MU" and P.muscle == "rat_MG":  # fast MU scale
            Ca_norm = Ca * P.Ca_max_f_MU_ratGM  
            k1, k2 = P.k1_f_MU_ratGM, P.k2_f_MU_ratGM      
        else:  
            raise ValueError("No scale-type combination")  

        if Ca_norm > act:  # ascending / normalized branch
            return (k1 * Ca_norm - k2 * act) * (1 - act)  # Hussein-like normalization
        else:  # descending branch
            return (k1 * Ca_norm - k2 * act)  # not limited to 1


    @staticmethod
    def yield_dot(y_val: float, V_norm: float) -> float:  
        
        """
        Computes yielding state derivative as a function of normalized velocity (from Brown et al. 1999).
        Inputs:
        - y_val: float, yielding state (dimensionless).
        - V_norm: float, normalized velocity.
        Outputs:
        - dyield: float, yielding derivative.
        """

        cy, Vy, Ty = 0.35, 0.1, 0.2  # yielding parameters
        return (1 - cy * (1 - np.exp((-abs(V_norm)) / Vy)) - y_val) / Ty  # ODE
        


    def sag_dot(self, s: float, t: float) -> float:  

        """
        Computes sag state derivative, with a time-dependent target value As(t) (modified from Brown et al. 2000).
        Inputs:
        - s: float, sag state.
        - t: float, current time.
        Outputs:
        - dsag: float, sag derivative.
        """
        P = self.P
        if P.muscle == "cat_MG":
            tp = 0.262
        else:
            tp = 0.1

        if 0 < t < tp:  # early phase
            As = self.P.As_peak  # peak value
        else:  # later phase
            As = self.P.As_decay  # decay value
        return (As - s) / self.P.Ts  # ODE


# =============================================================================
# Model configuration
# =============================================================================

@dataclass(frozen=True)  # fixed configuration
class ModelConfig:  # configuration of model components
    use_SE: bool = True     # SE + pennation + l_M ODE, otherwise l_M == l_MT
    use_PE: bool = True         # include passive elastic element
    use_FL: bool = True         # include force-length relationship
    use_FV: bool = True         # include force-velocity relationship
    use_yielding: bool = True   # include yielding
    use_sag: bool = True        # include sag 


# =============================================================================
# State names + index mapping 
# =============================================================================

def build_state_names(model_config: ModelConfig) -> List[str]:  

    """
    Builds an ordered list of state names included in the ODE based on ModelConfig.
    Inputs:
    - model_config: ModelConfig to include tendon/PE/FL/FV/yielding/sag.
    Outputs:
    - state_names: List[str], ordered names corresponding to positions in y and sol.y
    """
    
    names = ["beta", "dbeta", "Ca", "dCa", "act"]  # always-present states
    if model_config.use_SE:  # tendon system requires fibre length state
        names.append("l_M")  
    if model_config.use_yielding:  # yielding only if enabled
        names.append("yielding")  
    if model_config.use_sag:  # sag only if enabled
        names.append("sag") 
    return names  # return ordered state list


def build_state_index(state_names: List[str]) -> Dict[str, int]:  

    """
    Creates a mapping from state name to position index in the ODE state vector y.
    Inputs:
    - state_names: List[str], ordered list of state names.
    Outputs:
    - state_index: Dict[str, int], mapping name -> integer index.
    """
    
    return {name: i for i, name in enumerate(state_names)}  


# =============================================================================
# ODE System
# =============================================================================

class ODESystem:  # ODE assembly and consistent force computations
    def __init__(self, P: Params, S: States, mech: Mechanics, eph: Ephys, model_config: ModelConfig):  
        self.P = P  # store parameters reference
        self.S = S  # store initial states reference (for pennation update uses l_M_0)
        self.mech = mech  # store mechanics block
        self.eph = eph  # store electrophysiology block
        self.model_config = model_config  # store model configuration
        self.state_names = build_state_names(model_config)  # compute state names from configuration
        self.state_index = build_state_index(self.state_names)  # compute name to index mapping
        self._v_lMT = np.gradient(np.asarray(self.P.l_MT, dtype=float), self.P.dt)  # dl_MT/dt for no-tendon FV case


    def compute_forces(self, time_index: int, y: np.ndarray, fibre_type: str) -> Dict[str, float]:

        """
        Computes all mechanical quantities (l_T, eps_T, f_SE, f_PE, f_CE, FL, FV, v_M, alpha)
        consistently for the current time_index and ODE state y.
        Inputs:
        - time_index: int, index into P.time / P.l_MT.
        - y: np.ndarray, ODE state vector at this time (size = n_states).
        - fibre_type: str, "slow" or "fast".
        Outputs:
        - forces: Dict[str, float], dictionary of mechanical variables.
        """
        
        P = self.P  # local parameters
        model_config = self.model_config  # local configuration
        state_index_local = self.state_index  # local alias for state index mapping
        time_index = max(0, min(time_index, len(P.time) - 1))  # clamp time index to valid range
        act = float(y[state_index_local["act"]])  # read activation state

        if model_config.use_SE:  # tendon system: l_M is dynamic
            l_M = float(y[state_index_local["l_M"]])  # read muscle fibre length from state
            alpha = self.mech.pennation(l_M, self.S.l_M_0, P.alpha_0) # pick current pennation
            l_T = float(P.l_MT[time_index] - l_M * np.cos(alpha))  # compute tendon length from geometry
            eps_T = float((l_T - P.l_T_slack) / P.l_T_slack)  # tendon strain
            f_SE = float(self.mech.tendon_force(eps_T))  # tendon force
            lM_norm = float(l_M / P.l_M_opt)  # normalized fibre length
        else:  # muscle-only system: l_M == l_MT
            l_M = float(P.l_MT[time_index])  # fibre length equals MT length
            l_T, eps_T, f_SE = 0.0, 0.0, 0.0  # no tendon quantities
            lM_norm = float(l_M / P.l_M_opt)  # normalized fibre length
            alpha = float(P.alpha_0)  # pick current pennation

        f_PE = float(self.mech.passive_pe(lM_norm)) if model_config.use_PE else 0.0  # passive force if enabled
        FL = float(self.mech.force_length(act, lM_norm)) if model_config.use_FL else 1.0  # FL force if enabled

        if model_config.use_SE:  # if tendon: CE force from equilibrium
            f_CE = float(f_SE / max(1e-9, np.cos(alpha)) - f_PE)  # CE force (normalized)
        else:  # no tendon: CE not defined in equilibrium form
            f_CE = 0.0  # set to 0 for completeness

        # Yielding can be applied also without tendon, because v_M is still available from l_MT kinematics
        # Therefore  v_M is always computed when possible (even if FV is disabled).
        if model_config.use_FV:  # FV enabled
            if model_config.use_SE:  # tendon system: invert FV to get v_M
                v_M = float(self.mech.fv_velocity(act, lM_norm, f_CE / max(1e-12, FL), FL, fibre_type, P.vmax))  # fibre velocity
                FV = float(self.mech.fv_force(act, v_M / P.vmax, FL, lM_norm, fibre_type))  # FV factor
            else:  # no tendon: v_M from imposed MT kinematics
                v_M = float(self._v_lMT[time_index])  # imposed fibre velocity
                v_norm = float(v_M / P.vmax)  # normalized velocity
                FV = float(self.mech.fv_force(act, v_norm, FL, lM_norm, fibre_type))  # FV factor
        else:  # FV disabled
            # still compute v_M in the no-tendon case so yielding has access to velocity
            if model_config.use_SE:  # tendon case with FV disabled
                v_M = 0.0  # neutral choice (no FV inversion available)
            else:  # no tendon case: velocity is imposed by l_MT regardless of FV usage
                v_M = float(self._v_lMT[time_index])  # imposed fibre velocity
            FV = 1.0  # neutral FV

        return dict(l_M=l_M, l_T=l_T, eps_T=eps_T,  
            f_SE=f_SE, f_PE=f_PE, f_CE=f_CE,  
            FL=FL, FV=FV, v_M=v_M, alpha=alpha)


    def ode_system(self, t: float, y: np.ndarray, distimes: np.ndarray, fibre_type: str) -> np.ndarray:

        """
        - Defines the full ODE right-hand-side for solve_ivp.
        - Computes electrophysiology derivatives always.
        - Adds sag and tendon-related derivatives (l_M, yielding) depending on model_config.
        Inputs:
        - t: float, current time.
        - y: np.ndarray, current ODE state vector.
        - distimes: np.ndarray, discharge times in seconds.
        - fibre_type: str, "slow" or "fast".
        Outputs:
        - dydt: np.ndarray, derivatives of the ODE state vector.
        """
        
        P = self.P  # local parameters
        model_config = self.model_config  # local config
        state_index_local = self.state_index  # local state-index mapping
        time_index = max(0, min(int(t / P.dt), len(P.time) - 1))  # convert time to nearest index and clamp
        dydt = np.zeros_like(y)  # allocate derivative vector

        beta = float(y[state_index_local["beta"]])  # MUAP state
        dbeta = float(y[state_index_local["dbeta"]])  # MUAP derivative
        Ca = float(y[state_index_local["Ca"]])  # Ca state
        dCa = float(y[state_index_local["dCa"]])  # Ca derivative
        act = float(y[state_index_local["act"]])  # activation state

        DDbeta = self.eph.MU_AP_2nd(t, distimes, beta, dbeta)  # MUAP second derivative

        if model_config.use_SE:  # if tendon system use fibre length state for Ca kinetics
            l_norm_for_Ca = float(y[state_index_local["l_M"]]) / P.l_M_opt  # normalized fibre length
        else:  # otherwise use imposed MT length for Ca kinetics
            l_norm_for_Ca = float(P.l_MT[time_index] / P.l_M_opt)  # normalized imposed length

        DDCa = self.eph.Ca_2nd(l_norm_for_Ca, fibre_type, beta, Ca, dCa)  # Ca second derivative
        dact = self.eph.activation_dot_from(Ca, act, fibre_type)  # activation derivative

        dydt[state_index_local["beta"]] = dbeta  # d(beta)/dt = dbeta
        dydt[state_index_local["dbeta"]] = DDbeta  # d(dbeta)/dt = DDbeta
        dydt[state_index_local["Ca"]] = dCa  # d(Ca)/dt = dCa
        dydt[state_index_local["dCa"]] = DDCa  # d(dCa)/dt = DDCa
        dydt[state_index_local["act"]] = dact  # d(act)/dt = dact

        if model_config.use_sag:  # if sag is enabled
            sag_val = float(y[state_index_local["sag"]])  # current sag state
            dydt[state_index_local["sag"]] = self.eph.sag_dot(sag_val, t)  # sag derivative

        # compute forces/velocity consistently (needed for tendon dynamics AND yielding even without tendon)
        forces = self.compute_forces(time_index, y, fibre_type)  # compute consistent forces

        # yielding can be applied even without tendon (depends on velocity)
        if model_config.use_yielding:  # if yielding enabled
            yld = float(y[state_index_local["yielding"]])  # current yielding state
            dydt[state_index_local["yielding"]] = self.eph.yield_dot(yld, forces["v_M"] / P.vmax)  # yielding derivative

        if model_config.use_SE:  # tendon system adds l_M dynamics and pennation update
            dydt[state_index_local["l_M"]] = forces["v_M"]  # dl_M/dt = fibre velocity

        return dydt  # return derivatives to the integrator


# =============================================================================
# Main manager
# =============================================================================

class MuscleModel:  # main model object
    def __init__(self, P: Params, S: States, distimes, model_config: ModelConfig): 
        self.P = P  # store params
        self.S = S  # store states
        self.model_config = model_config  # store configuration
        self.distimes = np.atleast_1d(np.asarray(distimes, dtype=float)).ravel()  # ensure 1D float distimes
        self.fs = compute_fs(self.distimes)

        self.P.l_MT = np.asarray(self.P.l_MT, dtype=float).ravel()  # standardize l_MT to 1D float
        if len(self.P.l_MT) == len(self.P.time) + 1:  # allow len(time)+1 input (common from earlier scripts)
            self.P.l_MT = self.P.l_MT[:-1]  # drop last sample to match time length
        if len(self.P.l_MT) != len(self.P.time):  # consistent lengths
            raise ValueError(f"Expected l_MT length {len(self.P.time)} (or +1), got {len(self.P.l_MT)}")  # error

        self.mech = Mechanics(P)  # create mechanics block
        self.eph = Ephys(P)  # create electrophysiology block
        self.sys = ODESystem(P, S, self.mech, self.eph, self.model_config)  # create ODE system block

    @property
    def fibre_type(self) -> str:  

        """
        Infers fibre type ("slow" or "fast") from the Params.muscle string.
        Inputs:
        - None (uses self.P.muscle).
        Outputs:
        - fibre_type: str, "slow" or "fast".
        """
        
        m = self.P.muscle  # local muscle name
        if m in {"rat_SOL", "cat_SOL", "cat_LG"}:  # slow muscles
            return "slow"  
        if m in {"rat_EDL", "cat_MG", "cat_CF", "rat_MG"}:  # fast muscles
            return "fast"  
        raise ValueError(f"Unknown muscle '{m}'")  # error if unknown
    

    def _build_y0(self) -> np.ndarray:  

        """
        Builds the initial ODE state vector y0 with the correct size/order based on model_config.
        Inputs:
        - None (uses self.S and self.sys.state_index / self.model_config).
        Outputs:
        - y0: np.ndarray, initial conditions vector (shape: (n_states,)).
        """

        state_index_local = self.sys.state_index  # local for mapping
        y0 = np.zeros(len(self.sys.state_names), dtype=float)  # allocate y0 with correct size

        y0[state_index_local["beta"]] = self.S.MUAP_0  # initial beta
        y0[state_index_local["dbeta"]] = self.S.MUAP_0  # initial dbeta (as in your original)
        y0[state_index_local["Ca"]] = self.S.Ca_0  # initial Ca
        y0[state_index_local["dCa"]] = self.S.Ca_0  # initial dCa
        y0[state_index_local["act"]] = self.S.act_0  # initial activation

        if self.model_config.use_SE:  # tendon requires l_M state
            y0[state_index_local["l_M"]] = self.S.l_M_0  # initial fibre length

        # yielding can be applied even without tendon (depends on velocity)
        if self.model_config.use_yielding:  # yielding optional
            y0[state_index_local["yielding"]] = self.S.y_0  # initial yielding state

        if self.model_config.use_sag:  # sag optional
            y0[state_index_local["sag"]] = self.S.s_0  # initial sag state

        return y0  # return initial condition vector


    def run(self, output_force: bool = True) -> Dict[str, Any]:  

        """
        - Solves the ODE system using solve_ivp (LSODA) on the provided time grid.
        - Returns key state trajectories (MUAP, Ca, act, l_M, sag, yielding) depending on model_config.
        - Recomputes mechanical traces (FL, FV, f_PE, f_SE, v_M) at each time step.
        - Optionally computes a default "active-only" force output.
        Inputs:
        - output_force: bool, if True computes and returns out["force"].
        Outputs:
        - out: Dict[str, Any] with at least:
               * "t": time array
               * "MUAP", "Ca", "act"
               * "l_M"
             and optionally:
               * "yielding" (if tendon + yielding)
               * "sag" (if sag)
               * "FL", "FV", "f_PE", "f_SE", "v_M"
               * "force" (if output_force=True)
        """
        
        fibre_type = self.fibre_type  # fibre type
        state_index_local = self.sys.state_index  # local state index
        print("Computing muscle force...")

        y0 = self._build_y0()  # build initial state vector

        args = (self.distimes.astype(float), fibre_type)  # extra args passed to ode_system
        sol = solve_ivp(  # integrate ODEs
            self.sys.ode_system,  
            [self.P.time[0], self.P.time[-1]],  
            y0,  
            args=args,  
            method="LSODA",  
            t_eval=self.P.time,  
            max_step=self.P.dt / 4,  
        )

        Y = sol.y  # state with size (n_states, n_time)

        out: Dict[str, Any] = {  # initialize output dict
            "t": self.P.time,  # time vector
            "MUAP": Y[state_index_local["beta"], :],  # MUAP state
            "Ca": Y[state_index_local["Ca"], :],  # free Ca state 
            "act": Y[state_index_local["act"], :],  # activation state
        }

        if self.model_config.use_SE:  # tendon case: l_M comes from state vector
            out["l_M"] = Y[state_index_local["l_M"], :]  # fibre length trajectory
        else:  # no tendon case: l_M equals imposed length
            out["l_M"] = self.P.l_MT.copy()  # imposed fibre length

        if self.model_config.use_yielding:  # if yielding is enabled
            out["yielding"] = Y[state_index_local["yielding"], :]  # yielding trajectory

        if self.model_config.use_sag:  # if sag is enabled
            out["sag"] = Y[state_index_local["sag"], :]  # sag trajectory

        T = len(self.P.time)  # number of time samples
        FL = np.zeros(T)  # allocate FL trace
        FV = np.zeros(T)  # allocate FV trace
        f_PE = np.zeros(T)  # allocate passive force trace
        f_SE = np.zeros(T)  # allocate tendon force trace
        v_M = np.zeros(T)  # allocate fibre velocity trace

        for i in range(T):  # loop through time samples
            forces = self.sys.compute_forces(i, Y[:, i], fibre_type)  # compute mechanics consistently
            FL[i] = forces["FL"]  # store FL
            FV[i] = forces["FV"]  # store FV
            f_PE[i] = forces["f_PE"]  # store passive force
            f_SE[i] = forces["f_SE"]  # store tendon force
            v_M[i] = forces["v_M"]  # store fibre velocity

        out.update({"FL": FL, "FV": FV, "f_PE": f_PE, "f_SE": f_SE, "v_M": v_M})  # add mechanics traces

        if output_force:  # if force output requested
            time_factor = np.ones(T, dtype=float)  # default no time modulation

            if self.model_config.use_yielding and fibre_type == "slow" and self.fs < 37:  # apply yielding only to slow fibres if enabled and at submax.freqs.
                time_factor = out.get("yielding", time_factor)  # use yielding array if present
            elif self.model_config.use_sag and fibre_type == "fast" and  5 < self.fs < 100:  # apply sag only to fast fibres if enabled and at submax.freqs.
                time_factor = out.get("sag", time_factor)  # use sag array if present

            out["force"] = self.P.MVC * (time_factor * out["act"] * out["FL"] * out["FV"] + out["f_PE"])

        return out  # return all outputs


# =============================================================================

def Multiscale_model(parameters: dict, states: dict, distimes, model_config: ModelConfig) -> MuscleModel:  

    """
    Constructor that builds Params/States from dictionaries and returns a MuscleModel.
    Inputs:
    - parameters: dict, keys must match Params fields.
    - states: dict, keys must match States fields.
    - distimes: array-like, discharge times in seconds.
    - model_config: ModelConfig, decides which model components are active.
    Outputs:
    - model: MuscleModel instance ready to run.
    """
    
    P = Params(**parameters)  # build Params from dictionary
    S = States(**states)  # build States from dictionary
    return MuscleModel(P, S, distimes, model_config=model_config)  # return configured model instance
