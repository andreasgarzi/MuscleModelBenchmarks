
"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Sat Jun  1 14:57:36 2024
___________________________________

Plot all the element relationships of the MN-driven model
"""
#%%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#%% Class for muscle-tendon relationships
class Relationships:
    def __init__(self, MU_type='slow'):
        self.fmax = 1.4
        self.af = 0.17
        self.set_MU_type(MU_type)

    def set_MU_type(self, MU_type):
        self.MU_type = MU_type
        self.kMU = 0.5 if MU_type == 'slow' else 1

    def T_force(self, eps, eps_0, k_toe):
        eps_toe = 0.609 * eps_0
        klin = 1.712 / eps_0
        F_toe = 0.333
        if eps > eps_toe:
            return 0.001*(1+eps) + (klin*(eps-eps_toe)+F_toe)
        elif eps > 0:
            return 0.001*(1+eps) + (F_toe*((np.exp(k_toe*eps/eps_toe)-1)/(np.exp(k_toe)-1)))
        else:
            return 0.001*(1+eps)

    def velo_fFV(self, CE_force, FL_force, act, l_M):
        fv = 0.9 + 0.1 * act
        g = FL_force if l_M < 1 else 1
        b = (self.fmax - 1) / (2 + 2 / self.af)
        K = self.kMU * fv * g
        if CE_force >= 1:
            return b * ((CE_force - 1) / (self.fmax - CE_force)) * K
        else:
            return (CE_force - 1) / (CE_force / (self.af * K))

    def f_fFV(self, v_M, FL_force, act, l_M):
        fv = 0.9 + 0.1 * act
        g = FL_force if l_M < 1 else 1
        b = (self.fmax - 1) / (2 + 2 / self.af)
        K = self.kMU * fv * g
        if v_M < 0:
            return 1 / (1 - (v_M / (self.af * K)))
        else:
            return (1 + self.fmax * (v_M / (K * b))) / (1 + v_M / (K * b))

    def Force_Length_func(self, X, active_state):
        a = 0.45
        b = (0.15 * (1 - active_state)) + 1
        return np.exp(-((X - b) / a) ** 2)

    def PEE_force(self, l_M, k1, k2):
        if l_M < 1:
            return 0
        else:
            return (np.exp((k1 * (l_M - 1)) / k2) - 1) / (np.exp(k1) - 1)

# Utility: Color gradients 
def hex_to_RGB(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(1, 6, 2)]

def get_color_gradient(c1, c2, n):
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = np.linspace(0, 1, n)
    rgb_colors = [(1-m)*c1_rgb + m*c2_rgb for m in mix_pcts]
    return ["#" + "".join(f"{int(round(val*255)):02x}" for val in color) for color in rgb_colors]

# Parameters 
points = 2000
n_a = 5
fibre_type = 2
l_T_slack = 24

f_T_eps_0 = [0.013, 0.023, 0.033, 0.06, 0.055]
f_T_k_toe = [2, 2, 2, 3, 3]
f_PE_k1 = [5, 5, 5, 4, 3]
f_PE_k2 = [0.8, 1.0, 1.2, 0.6, 0.6]

a_vals = np.linspace(0.2, 1, n_a)
MU_types = ['fast', 'slow']
l_M = np.linspace(0.7, 1.8, points)
l_M_PE = np.linspace(0.8, 2.2, points)
v_M = np.linspace(-1.2, 1.2, points)
f_M_v = np.linspace(1e-9, 1.4, points)
eps_T = np.linspace(1e-9, 0.07, points)

# Output arrays 
f_T = np.zeros((points, n_a))
f_PE = np.zeros((points, n_a))
fl = np.zeros((n_a, points, fibre_type))
fv = np.zeros((n_a, points, fibre_type))
vf = np.zeros((n_a, points, fibre_type))

# Simulation 
for l in range(n_a):
    rel = Relationships()
    for i in range(points):
        fl[l, i, :] = [rel.Force_Length_func(l_M[i], a_vals[l])* a_vals[l]] 
        f_T[i, l] = rel.T_force(eps_T[i], f_T_eps_0[l], f_T_k_toe[l])
        f_PE[i, l] = rel.PEE_force(l_M_PE[i], f_PE_k1[l], f_PE_k2[l])

        for t, mu in enumerate(MU_types):
            rel.set_MU_type(mu)
            fv[l, i, t] = rel.f_fFV(v_M[i], fl[l, i, t], 1, a_vals[l])
            vf[l, i, t] = rel.velo_fFV(f_M_v[i], fl[l, i, t], 1, a_vals[l])

# Color maps 
cg_s = get_color_gradient("#0b165e", "#00eeff", n_a)
cg_T = get_color_gradient("#0c520b", "#18e314", n_a)
cg_PE1 = get_color_gradient('#eb7adc', '#440a66', 3)
cg_PE2 = get_color_gradient('#0c520b', '#18e314', 2)
cg_f = get_color_gradient('#e31010', '#fffe00', 3)

# Plotting 
fig, axs = plt.subplots(2, 2, figsize=(11, 5))
fig.suptitle('MN-driven model sensitivity (contractile part)', weight='bold', y=0.94)

# Tendon force
ax = axs[0, 0]
for i in range(n_a):
    ax.plot(eps_T*100, f_T[:, i], color=cg_T[i], label=f'$\epsilon^T_0$ = {f_T_eps_0[i]*100:.1f}%')
ax.annotate(r'$\epsilon^Ttoe$ = 0.609($\epsilon^T_0$)', xy=(0.1,1))
ax.set(xlabel=r'$\epsilon^T$ [%]', ylabel=r'$\overline {F^T}$')
ax.set_title('SE (Tendon)', weight='bold')
ax.grid(True)
ax.legend(fontsize=10)

# Passive force
ax = axs[0, 1]
for i in range(n_a):
    style = '-' if i < 3 else '--'
    color = cg_PE1[1] if i < 3 else cg_PE2[0]
    ax.plot(l_M_PE, f_PE[:, i], color=color, linestyle=style, label=f'k1={f_PE_k1[i]}, k2={f_PE_k2[i]}')
ax.set(xlabel=r'$\overline {L^M}$', ylabel=r'$\overline {F^M}$', xlim=(0.8, 2.2), ylim=(-0.4, 0.9))
ax.set_title('PE', weight='bold')
ax.grid(True)
ax.legend()

# Force-length
ax = axs[1, 0]
for i in range(n_a):
    ax.plot(l_M, fl[i, :, 0], color=cg_s[i], label=f'a = {a_vals[i]:.1f}')
ax.set(xlabel=r'$\overline {L^M}$', ylabel=r'$\overline {F^M_l}$')
ax.set_title('F-L relationship', weight='bold')
ax.grid(True)
ax.legend(fontsize=10)

# Force-velocity
ax = axs[1, 1]
for i in [0, 2, 4]:
    ax.plot(v_M, fv[i, :, 0], color=cg_s[i], label=f'a = {a_vals[i]:.1f}, slow')
    ax.plot(v_M, fv[i, :, 1], color=cg_f[i//2], label=f'a = {a_vals[i]:.1f}, fast')
ax.set(xlabel=r'$\overline {V^M}$', ylabel=r'$\overline {F^M_v}$', xlim=(-1.2, 1.2))
ax.set_title('F-V relationship ($\overline {L^M}$ = 1)', weight='bold')
ax.grid(True)
ax.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.show()

#%%
""" Extract experimental digitized data from Blinks, Konishi """

from scipy.optimize import curve_fit
cwd = Path.cwd()
data_path = Path.home() / 'Dropbox' / 'UNSW_Andrea_Luca_PhD' / 'Data' / 'Digitized_Blinks_Konishi'

# Load experimental data 
os.chdir(data_path)

konishi = pd.read_csv("Konishi_l_Capeak.csv", delimiter=' ').to_numpy()
konishi2 = pd.read_csv("Konishi_l_Catimetopeak.csv", delimiter=' ').to_numpy()
blinks = pd.read_csv("Blinks_l_Capeak.csv", delimiter=' ').to_numpy()
blinks[:, 1] /= 100  # Scale

blinks[:,0] /= 2.1  # Normalise frog sarcomere length
konishi[:,0] /= 2.1
konishi2[:,0] /= 2.1
blinks[:, 1] = np.minimum(blinks[:, 1], 1) # correct values greater than 1 from digitalization
konishi[:, 1] = np.minimum(konishi[:, 1], 1)

os.chdir(cwd)

# Concatenate and sort data 
l_Ca = np.concatenate((blinks, konishi), axis=0)
l_Ca = blinks
l_Ca = l_Ca[np.argsort(l_Ca[:, 0])] # length sorted

# Define Gaussian-like bell curve (with max = 1 for max. force potential)
def fixed_gaussian(x, sigma):
    return np.exp(-((x - 1)**2) / (2 * sigma**2))

# Fit bell curve
popt, _ = curve_fit(fixed_gaussian, l_Ca[:, 0], l_Ca[:, 1],
                    p0=[0.2])

l_sm = np.linspace(0.8, 2.3, 500)
Ca_sm = fixed_gaussian(l_sm, *popt)

# Fit time-to-peak Ca2+ data 
l_Cattp = konishi2[np.argsort(konishi2[:, 0])] # length sorted

p2 = np.polyfit(l_Cattp[:, 0], l_Cattp[:, 1], deg=2)
fit2 = np.poly1d(p2)
l_smttp = np.linspace(l_Cattp[:, 0].min(), l_Cattp[:, 0].max(), 500)
Ca_smttp = fit2(l_smttp)

# Plot Ca_peak 
fig2 = plt.subplot(2, 1, 1)
plt.plot(blinks[:, 0], blinks[:, 1], 'rx', label='Blinks 1978')
plt.plot(konishi[:, 0], konishi[:, 1], 'gx', label='Konishi 1991')
plt.plot(l_sm, Ca_sm, 'k', label='Fit')
plt.ylim([0,1.2])
plt.ylabel('Norm. p Δ[$Ca^{2+}$]', weight='bold')
plt.legend(loc='upper right')
plt.grid()

# Plot Ca time-to-peak
fig2 = plt.subplot(2, 1, 2)
plt.plot(l_Cattp[:, 0], l_Cattp[:, 1], 'gx', label='Konishi 1991')
plt.plot(l_smttp, Ca_smttp, 'k', label='Fit')
plt.xlabel('Normalised sarcomere length', weight='bold')
plt.ylabel('Norm. ttp Δ[$Ca^{2+}$]', weight='bold')
plt.legend(loc='upper right')
plt.grid()
plt.show()
#%%
