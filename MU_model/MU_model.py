"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

MU model based on Caillet et al. 2023 for simulating animal in-vitro/in-situ isometric and dynamic muscle contractions
given the experimental discharge times as inputs. The muscle is modelled as a single 3-el. (or 1 el.) Hill-type model,
representative of all active MUs of the same type in that muscle. An extensive excitation and activation dynamics are
added on top of a contraction dynamics, with different electrophysiological & contractile properties for slow &
fast MUs.

"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional

# =============================================================================
# Data containers
# =============================================================================

@dataclass
class Params:
    # Required
    time: np.ndarray
    dt: float
    muscle: str                 # e.g. 'rat_SOL', 'rat_EDL', 'rat_GM', 'human_TA', 'human_GM'
    yielding: int               # 0/1
    sag: int                    # 0/1
    MVC: float              
    vmax: float
    alpha_0: float
    l_MT: np.ndarray
    l_M_opt: float
    l_T_slack: float

    # Active state & Ca pars (to run optimizations)
    Ca_max_slow: Optional[float] = None
    Ca_max_fast: Optional[float] = None
    k1_s: Optional[float] = None
    k2_s: Optional[float] = None
    k1_f: Optional[float] = None
    k2_f: Optional[float] = None
    k1_MU: Optional[float] = None
    k2_MU: Optional[float] = None
    c1_slow: Optional[float] = None
    c2_slow: Optional[float] = None
    c3_slow: Optional[float] = None
    c1_fast: Optional[float] = None
    c2_fast: Optional[float] = None
    c3_fast: Optional[float] = None
    af: Optional[float] = None

    def __post_init__(self):

        # Basic checks
        assert self.yielding in (0, 1)
        assert self.sag in (0, 1)
        
        # Activation defaults pars (allows optimization)
        if self.Ca_max_slow is None: # slow muscle
            self.Ca_max_slow = 285916   
        if self.k1_s is None:
            self.k1_s = 10.8 
        if self.k2_s is None:
            self.k2_s = 14.7
        if self.Ca_max_fast is None: # fast muscle
            self.Ca_max_fast = 285916 
        if self.k1_f is None:
            self.k1_f = 10.8 
        if self.k2_f is None:
            self.k2_f = 14.7
        if self.k1_MU is None: # fast MU
            self.k1_MU = 32.5 
        if self.k2_MU is None:
            self.k2_MU = 72.2 

        # Calcium transient default pars (allows optimization)
        if self.c1_slow is None:
            self.c1_slow = 30605
        if self.c2_slow is None:
            self.c2_slow = 896181
        if self.c3_slow is None:
            self.c3_slow = 2
        if self.c1_fast is None:
            self.c1_fast = 2056
        if self.c2_fast == None:
            self.c2_fast = 467405
        if self.c3_fast is None:
            self.c3_fast = 0.435

        # FV default pars (allows optimization)
        if self.af is None:
            self.af = 0.45


@dataclass
class States:
    MUAP_0: float # initial MUAP amplitude
    Ca_0: float   # initial Ca concentration
    act_0: float  # initial activation state
    l_M_0: float  # initial muscle fiber length
    y_0: float    # initial yield state
    s_0: float    # initial sag state


# =============================================================================
# Utils
# =============================================================================

def compute_fs(distimes: np.ndarray) -> float:
    distimes = np.atleast_1d(distimes).ravel()
    if distimes.size >= 2: # spike train
        return float(np.mean(1.0 / np.diff(distimes))) # inst. freq.
    return 1.0 # twitch


# =============================================================================
# Mechanics (SE, PE, FL, FV, pennation)
# =============================================================================

class Mechanics:
    def __init__(self, P: Params):
        self.P = P

    @staticmethod
    def tendon_force(eps: float) -> float:
        eps_0 = 0.06
        klin = 1.212 / eps_0
        eps_toe = 0.609 * eps_0
        F_toe = 0.33
        k_toe = 3.0
        if eps > eps_toe:
            return 0.001 * (1 + eps) + (klin * (eps - eps_toe) + F_toe)
        elif eps > 0:
            return 0.001 * (1 + eps) + (F_toe * ((np.exp(k_toe * eps / eps_toe) - 1) / (np.exp(k_toe) - 1)))
        else:
            return 0.001 * (1 + eps)

    @staticmethod
    def passive_pe(l_M_norm: float) -> float:
        k1, k2 = 5.0, 0.6
        if l_M_norm < 1.0:
            return 0.0
        return (np.exp((k1 * (l_M_norm - 1)) / k2) - 1) / (np.exp(k1) - 1)

    @staticmethod
    def pennation(l_MT: float, l_T: float, l_M_0: float, alpha0: float) -> float:
        w = l_M_0 * np.sin(alpha0)
        cosalpha = 1.0 / (np.sqrt(1.0 + (w / (l_MT - l_T)) ** 2))
        alpha = np.arccos(cosalpha)
        return float(min(alpha, 1.4706289))

    @staticmethod
    def force_length(act: float, l_M_norm: float) -> float:
        a = 0.45
        b = (0.15 * (1 - act)) + 1
        return float(np.exp(-((l_M_norm - b) / a) ** 2))

    def fv_velocity(self, act: float, l_M_norm: float, f_CE_over_FL: float, FL: float, MU_type: str, vmax: float) -> float:
        fmax = 1.4
        af = self.P.af
        fv = 0.25 + 0.75*act
        kMU = 0.2 if MU_type == "slow" else 1.0
        g = FL if l_M_norm < 1 else 1.0
        b = (fmax - 1) / (2 + 2 / af)
        K = kMU * g * fv
        if f_CE_over_FL >= 1:
            vel = b * ((f_CE_over_FL - 1) / (fmax - f_CE_over_FL))
            vel *= K
        else:
            vel = (f_CE_over_FL - 1) / (f_CE_over_FL / (af * K))
        return float(vel * vmax)

    def fv_force(self, act: float, v_norm: float, FL: float, l_M_norm: float, MU_type: str) -> float:
        fmax = 1.4
        af = self.P.af
        fv = 0.25 + 0.75*act
        kMU = 0.2 if MU_type == "slow" else 1.0
        g = FL if l_M_norm < 1 else 1.0
        b = (fmax - 1) / (2 + 2 / af)
        K = kMU * g * fv
        if v_norm < 0:
            return float(1 / (1 - (v_norm / (af * K))))
        else:
            return float((1 + fmax * (v_norm / (K * b))) / (1 + v_norm / (K * b)))


# =============================================================================
# Electrophysiology (Ca2+, activation, sag & yield)
# =============================================================================

class Ephys:
    def __init__(self, P: Params):
        self.P = P

    @staticmethod
    def _is_firing(t_round: float, AP_times: np.ndarray, prec: float) -> int: # Caillet 2023
        t_int = int(t_round / prec + 1e-4)
        return int(t_int in (AP_times / prec).astype(int))

    def MN_AP(self, t: float, AP_times: np.ndarray, V_N: float = 90.0) -> float: # Caillet 2023
        prec = 1e-3
        t_round = int(t / prec + 1e-4) * prec
        if not self._is_firing(t_round, AP_times, prec):
            return 0.0
        sin_period = 1.4e-3
        t_end = t_round + sin_period / 2
        if t <= t_end:
            return float(V_N * np.sin(2 * np.pi / sin_period * (t - t_round)))
        return 0.0

    def MU_AP_2nd(self, t: float, AP_times: np.ndarray, beta: float, dbeta: float) -> float: # Caillet 2023
        c4, c5, c6 = 2e4, 5e7, 9e7
        return c6 * self.MN_AP(t, AP_times) - c5 * beta - c4 * dbeta + 1e-50

    def Ca_2nd(self, l_norm: float, MU_type: str, beta: float, Ca: float, dCa: float) -> float:

        # Use overridable parameters from Params
        if MU_type == "slow":
            c1, c2, c3 = self.P.c1_slow, self.P.c2_slow, self.P.c3_slow
        else:
            c1, c2, c3 = self.P.c1_fast, self.P.c2_fast, self.P.c3_fast

        # Linear fitting of Konishi, Blinks data
        if l_norm < 1: 
            amp = 0.8
        elif l_norm <= 1.1379:
            amp = 1.3947 * l_norm - 0.5871
        elif l_norm < 1.239:
            amp = 1.0
        else:
            amp = 1 - 0.4623 * (l_norm - 1.239)

        p2 = [0.3783, -0.8320, 1.1885] 
        width = (l_norm ** 2) * p2[0] + l_norm * p2[1] + p2[2]

        if l_norm < 1.23:
            width = 0.738
        elif l_norm > 2.04:
            width = 1.072

        return amp * c3 * beta - width * c1 * dCa - (c2 * width**2) * Ca 

    def activation_dot(self, y, MU_type: str, s: float, muscle: str) -> float:

        # Defaults; can specialize on muscle if desired
        if MU_type == "slow":
            Ca, a = y[2] * self.P.Ca_max_slow, y[4]
            k1_eff, k2_eff = self.P.k1_s, self.P.k2_s
            Ca_eff = Ca
        else:  # fast
            if muscle == "rat_EDL":
                Ca, a = y[2] * self.P.Ca_max_fast, y[4]
                k1_eff, k2_eff = self.P.k1_f, self.P.k2_f
            elif muscle == "rat_GM":
                Ca, a = y[2] * self.P.Ca_max_fast, y[4]
                k1_eff, k2_eff = self.P.k1_MU, self.P.k2_MU
            Ca_eff = Ca * (s if self.P.sag != 0 else 1.0)

        if Ca_eff > a: # Adapted from Hussein 2022
            return (k1_eff * Ca_eff - k2_eff * a) * (1 - a) # ascending phase normlized
        else:
            return (k1_eff * Ca_eff - k2_eff * a) # descending phase not normalized

    @staticmethod
    def yield_dot(y_val: float, V_norm: float) -> float: # yielding from Brown 1999
        cy, Vy, Ty = 0.35, 0.1, 0.2
        return (1 - cy * (1 - np.exp((-abs(V_norm)) / Vy)) - y_val) / Ty

    def sag_dot(self, s: float, t: float, fs: float) -> float: # Sag from Brown 1999
        Ts = 0.043
        if 0 < t < (fs - 23.227) / 82.426 and fs < 100:
            As = (fs - 1.024) / 22.891
        else:
            As = 0.8
        return (As - s) / Ts


# =============================================================================
# ODE systems (muscle-tendon and muscle-only)
# =============================================================================

class Systems:
    def __init__(self, P: Params, S: States, mech: Mechanics, eph: Ephys):
        self.P = P
        self.S = S
        self.mech = mech
        self.eph = eph

    def MT(self, t: float, y: np.ndarray, alpha_track: np.ndarray, distimes: np.ndarray, MU_type: str):

        idx = int(t / self.P.dt)
        if idx == 0:
            alpha_track[idx] = self.P.alpha_0

        # Geometry and forces
        l_T = self.P.l_MT[idx] - y[5] * np.cos(alpha_track[idx])
        eps_T = (l_T - self.P.l_T_slack) / self.P.l_T_slack
        f_SE = self.mech.tendon_force(eps_T)
        f_PE = self.mech.passive_pe(y[5] / self.P.l_M_opt)
        f_CE = f_SE / np.cos(alpha_track[idx]) - f_PE

        # Next-step pennation
        if idx + 1 < len(self.P.time):
            alpha_track[idx + 1] = self.mech.pennation(self.P.l_MT[idx + 1], l_T, self.S.l_M_0, self.P.alpha_0)

        # ODE blocks
        dbeta = y[1]
        DDbeta = self.eph.MU_AP_2nd(t, distimes, y[0], dbeta)
        dCa = y[3]
        DDCa = self.eph.Ca_2nd(y[5] / self.P.l_M_opt, MU_type, y[0], y[2], dCa)
        dact = self.eph.activation_dot(y, MU_type, y[7], self.P.muscle)
        FL = self.mech.force_length(y[4], y[5] / self.P.l_M_opt)
        dldot = self.mech.fv_velocity(y[4], y[5] / self.P.l_M_opt, f_CE / max(1e-12, FL), FL, MU_type, self.P.vmax)
        dyield = self.eph.yield_dot(y[6], dldot / self.P.vmax)
        dsag = self.eph.sag_dot(y[7], t, fs=compute_fs(distimes))

        return [dbeta, DDbeta, dCa, DDCa, dact, dldot, dyield, dsag]

    def M_only(self, t: float, y: np.ndarray, distimes: np.ndarray, MU_type: str):

        dbeta = y[1]
        DDbeta = self.eph.MU_AP_2nd(t, distimes, y[0], dbeta)
        dCa = y[3]
        DDCa = self.eph.Ca_2nd(self.P.l_MT[int(t / self.P.dt)], MU_type, y[0], y[2], dCa)
        dact = self.eph.activation_dot(y, MU_type, y[5], self.P.muscle)
        dsag = self.eph.sag_dot(y[5], t, fs=compute_fs(distimes))

        return [dbeta, DDbeta, dCa, DDCa, dact, dsag]


# =============================================================================
# Main model manager
# =============================================================================

class MuscleModel:
    def __init__(self, P: Params, S: States, distimes):
        self.P = P
        self.S = S
        self.distimes = np.atleast_1d(np.asarray(distimes, dtype=float)).ravel()
        self.fs = compute_fs(self.distimes) 

        # Blocks
        self.mech = Mechanics(P)
        self.eph = Ephys(P)
        self.sys = Systems(P, S, self.mech, self.eph)

        # Preallocation
        T = len(P.time)
        self.alpha = np.empty(T + 1)
        for name in [
            "l_T", "eps_T", "f_SE", "f_CE", "f_PE", "f_FL", "vel", "f_M",
            "l_M", "MUAP", "active_state", "free_Ca", "yielding", "sag"
        ]:
            setattr(self, name, np.empty(T))

    @property
    def MU_type(self) -> str:
        m = self.P.muscle
        if m in {"rat_SOL", "cat_SOL"}:
            return "slow"
        if m in {"rat_EDL", "rat_GM"}:
            return "fast"

    def run_MT_simulation(self):

        distimes = self.distimes.astype(float)

        print("Computing muscle force...")
        MU_type = self.MU_type

        # Set initial states and args & solve ODE system
        y0 = [
            self.S.MUAP_0, self.S.MUAP_0,  # beta, dbeta
            self.S.Ca_0, self.S.Ca_0,      # Ca, dCa
            self.S.act_0,                  # activation
            self.S.l_M_0,                  # l_M 
            self.S.y_0,                    # yielding 
            self.S.s_0,                    # sag 
        ]
        args = (self.alpha, distimes, MU_type)
        sol = solve_ivp(
            self.sys.MT,
            [self.P.time[0], self.P.time[-1]],
            y0,
            args=args,
            method="LSODA",
            t_eval=self.P.time,
            max_step=self.P.dt / 4,
        )

        # Extract solutions
        self.MUAP = sol.y[0]
        self.free_Ca = sol.y[2]
        self.active_state = sol.y[4]
        self.l_M = sol.y[5]
        self.yielding = sol.y[6]
        self.sag = sol.y[7]

        # Recompute mechanics frame-by-frame
        self.alpha.fill(self.P.alpha_0)
        for l in range(len(self.P.time)):
            self.l_T[l] = self.P.l_MT[l] - self.l_M[l] * np.cos(self.alpha[l])
            self.eps_T[l] = (self.l_T[l] - self.P.l_T_slack) / self.P.l_T_slack
            self.f_SE[l] = self.mech.tendon_force(self.eps_T[l])
            self.f_PE[l] = self.mech.passive_pe(self.l_M[l] / self.P.l_M_opt)
            self.f_CE[l] = self.f_SE[l] / np.cos(self.alpha[l]) - self.f_PE[l]
            self.f_FL[l] = self.mech.force_length(self.active_state[l], self.l_M[l] / self.P.l_M_opt)
            self.vel[l] = self.mech.fv_velocity(self.active_state[l], self.l_M[l] / self.P.l_M_opt, self.f_CE[l] / max(1e-12, self.f_FL[l]), self.f_FL[l], MU_type, self.P.vmax)
            self.f_M[l] = self.mech.fv_force(self.active_state[l], self.vel[l] / self.P.vmax, self.f_FL[l], self.l_M[l] / self.P.l_M_opt, MU_type)
            
            if l + 1 < len(self.P.time):
                self.alpha[l + 1] = self.mech.pennation(self.P.l_MT[l + 1], self.l_T[l], self.S.l_M_0, self.P.alpha_0)

        # Aggregate forces (apply yield only to slow MUs if enabled)
        if self.P.yielding == 1:
            yield_factor = self.yielding 
        else:
            yield_factor = 1.0

        MU_force_norm = yield_factor * self.active_state * self.f_M * self.f_FL + self.f_PE
        F_MU = self.P.MVC * MU_force_norm

        return F_MU, self.MUAP, self.free_Ca, self.active_state, self.l_M, self.yielding, self.sag

    def run_M_simulation(self):

        distimes = self.distimes.astype(float)

        print("Computing muscle force...")
        MU_type = self.MU_type

        # Set initial states and args & solve ODE system
        y0 = [
            self.S.MUAP_0, self.S.MUAP_0, 
            self.S.Ca_0, self.S.Ca_0, 
            self.S.act_0, 
            self.S.s_0
        ]
        args = (distimes, MU_type)
        sol = solve_ivp(
            self.sys.M_only,
            [self.P.time[0], self.P.time[-1]],
            y0,
            args=args,
            method="LSODA",
            t_eval=self.P.time,
            max_step=self.P.dt / 4,
        )

        # Extract solutions
        self.MUAP = sol.y[0]
        self.free_Ca = sol.y[2]
        self.active_state = sol.y[4]
        self.sag = sol.y[5]

        # Recompute mechanics frame-by-frame
        for l in range(len(self.P.time)):
            self.f_PE[l] = Mechanics.passive_pe(self.P.l_MT[l])
            self.f_CE[l] = Mechanics.force_length(self.active_state[l], self.P.l_MT[l])

        # Aggregate forces
        MU_force = self.active_state * self.f_CE + self.f_PE
        F_MU = self.P.MVC * MU_force

        return F_MU, self.MUAP, self.free_Ca, self.active_state, self.sag


# =============================================================================
# Factory
# =============================================================================

def MU_model(parameters: dict, states: dict, distimes) -> MuscleModel:
    P = Params(**parameters)
    S = States(**states)
    return MuscleModel(P, S, distimes)



