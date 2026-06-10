"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Sat Jun  1 14:57:36 2024
___________________________________

Plot all the element relationships of the model
"""

from pathlib import Path
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Relationships present in the model
class Relationships:
    def __init__(self, fibre_type='slow'):
        self.fmax = 1.4
        self.af_s = 0.419
        self.af_f = 0.361
        self.set_fibre_type(fibre_type)

    def set_fibre_type(self, fibre_type):
        self.fibre_type = fibre_type
        if fibre_type == 'slow':
            self.kMU = 0.2
            self.af = self.af_s
        elif fibre_type == 'fast':
            self.kMU = 1.0
            self.af = self.af_f
        else:
            raise ValueError("fibre_type must be 'slow' or 'fast'")

    @staticmethod
    def tendon_force(eps, eps_0):
        klin = 1.712 / eps_0
        eps_toe = 0.609 * eps_0
        F_toe = 0.33
        k_toe = 3.0

        if eps > eps_toe:
            return 0.001 * (1 + eps) + (klin * (eps - eps_toe) + F_toe)
        elif eps > 0:
            return 0.001 * (1 + eps) + (
                F_toe * ((np.exp(k_toe * eps / eps_toe) - 1) / (np.exp(k_toe) - 1))
            )
        else:
            return 0.001 * (1 + eps)

    @staticmethod
    def passive_pe(l_M_norm):
        kPE = 5.0
        eps0 = 0.6

        if l_M_norm < 1.0:
            return 0.0
        return (np.exp((kPE * (l_M_norm - 1.0)) / eps0) - 1.0) / (np.exp(kPE) - 1.0)

    @staticmethod
    def force_length(act, l_M_norm):
        a = 0.45
        b = (0.15 * (1.0 - act)) + 1.0
        return np.exp(-((l_M_norm - b) / a) ** 2)

    def fv_force(self, act, v_norm, FL, l_M_norm):
        fv = 0.25 + 0.75 * act
        g = FL if l_M_norm < 1.0 else 1.0
        af = max(float(self.af), 1e-8)
        b = (self.fmax - 1.0) / (2.0 + 2.0 / af)
        K = max(float(self.kMU * g * fv), 1e-12)

        if v_norm < 0:  # shortening
            return 1.0 / (1.0 - (v_norm / (af * K)))
        else:  # lengthening
            return (1.0 + self.fmax * (v_norm / (K * b))) / (1.0 + v_norm / (K * b))


# Color gradients
def hex_to_rgb(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(1, 6, 2)]

def get_color_gradient(c1, c2, n):
    c1_rgb = np.array(hex_to_rgb(c1)) / 255
    c2_rgb = np.array(hex_to_rgb(c2)) / 255
    mix = np.linspace(0, 1, n)
    rgb = [(1 - m) * c1_rgb + m * c2_rgb for m in mix]
    return [
        "#" + "".join(f"{int(round(val * 255)):02x}" for val in color)
        for color in rgb
    ]


# Parameters for plotting
points = 1000
n_a = 5

act_vals = np.linspace(0.2, 1.0, n_a)
eps0_vals = [0.04, 0.05, 0.06, 0.07, 0.08]

l_M = np.linspace(0.7, 1.8, points)
l_M_PE = np.linspace(0.8, 2.2, points)
v_norm = np.linspace(-1.2, 1.2, points)
eps_T = np.linspace(-0.005, 0.10, points)

cg_blue = get_color_gradient("#0b165e", "#00eeff", n_a)
cg_green = get_color_gradient("#0c520b", "#18e314", len(eps0_vals))
cg_red = get_color_gradient("#8c0000", "#ffb000", n_a)

# Preallocate arrays
f_T = np.zeros((points, len(eps0_vals)))
f_PE = np.zeros(points)
FL_scaled = np.zeros((n_a, points))
FV_slow = np.zeros((n_a, points))
FV_fast = np.zeros((n_a, points))

rel_slow = Relationships('slow')
rel_fast = Relationships('fast')

# Compute curves
for j, eps0 in enumerate(eps0_vals):
    for i in range(points):
        f_T[i, j] = Relationships.tendon_force(eps_T[i], eps0)

for i in range(points):
    f_PE[i] = Relationships.passive_pe(l_M_PE[i])

for a_idx, act in enumerate(act_vals):
    FL_ref = Relationships.force_length(act, 1.0)

    for i in range(points):
        # Force-length scaled by active state
        FL_scaled[a_idx, i] = act * Relationships.force_length(act, l_M[i])

        # Force-velocity at normalized length = 1
        FV_slow[a_idx, i] = rel_slow.fv_force(act, v_norm[i], FL_ref, 1.0)
        FV_fast[a_idx, i] = rel_fast.fv_force(act, v_norm[i], FL_ref, 1.0)


# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.subplots_adjust(hspace=0.35)

labels = ['A', 'B', 'C', 'D']

for ax, lab in zip(axs.flat, labels):
    ax.text(
        -0.14, 1.1, lab,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top'
    )

# Tendon force-strain
ax = axs[0, 1]
for j, eps0 in enumerate(eps0_vals):
    ax.plot(
        eps_T * 100,
        f_T[:, j],
        color=cg_green[j],
        label=rf'$\epsilon_0^T$ = {eps0*100:.1f}%'
    )
ax.set_xlabel(r'$\epsilon^T$ [%]', fontsize=13)
ax.set_ylabel(r'$\overline{F}^{T}$', fontsize=13)
ax.set_title('SE', fontweight='bold')
ax.grid(True)
ax.legend(fontsize=10)

# Passive PE
ax = axs[0, 0]
ax.plot(l_M_PE, f_PE, color='#7a1fa2', linewidth=2)
ax.set_xlabel(r'$\overline{L}^{M}$', fontsize=13)
ax.set_ylabel(r'$\overline{F}_{PE}$', fontsize=13)
ax.set_title('PE', fontweight='bold')
ax.set_xlim(0.8, 1.8)
ax.set_ylim(-0.1, 2.2)
ax.grid(True)

# Force-velocity: slow + fast overlapped
ax = axs[1, 1]
for a_idx, act in enumerate(act_vals):
    ax.plot(
        v_norm, FV_slow[a_idx, :],
        color=cg_blue[a_idx],
    )
    ax.plot(
        v_norm, FV_fast[a_idx, :],
        color=cg_red[a_idx],
    )
ax.set_xlabel(r'$\overline{v}$', fontsize=13)
ax.set_ylabel(r'$\overline{f}_{FV}$', fontsize=13)
ax.set_title(r'CE - FV ($\overline{L}^{CE}=1$)', fontweight='bold')
ax.set_xlim(-1.2, 1.2)
ax.grid(True)
ax.legend(fontsize=10, loc='lower right')

handles = []

# slow (prima)
for a_idx, act in enumerate(act_vals):
    handles.append(
        Line2D(
            [0], [0],
            color=cg_blue[a_idx],
            lw=1,
            label=f'a = {act:.1f}, slow'
        )
    )

# fast (dopo)
for a_idx, act in enumerate(act_vals):
    handles.append(
        Line2D(
            [0], [0],
            color=cg_red[a_idx],
            lw=1,
            label=f'a = {act:.1f}, fast'
        )
    )

ax.legend(handles=handles, fontsize=8, loc='lower right')

# Force-length scaled by active state
ax = axs[1, 0]
for a_idx, act in enumerate(act_vals):
    ax.plot(
        l_M,
        FL_scaled[a_idx, :],
        color=cg_blue[a_idx],
        label=f'a = {act:.1f}'
    )
ax.set_xlabel(r'$\overline{L}^{M}$', fontsize=13)
ax.set_ylabel(r'$\overline{f}_{FL}$', fontsize=13)
ax.set_title('CE - FL', fontweight='bold')
ax.grid(True)
ax.legend(fontsize=10)


fig.savefig("relationships_plot.tif", dpi=500, bbox_inches="tight")
plt.show()

#####################################################################################################
# Ca2+ experimental digitized data from Blinks and Konishi
#####################################################################################################

base_path = Path() / 'benchmark_Data' / 'Ca_transients'

# Load digitized datasets
konishi_peak = pd.read_csv(base_path / "Konishi_l_Capeak.csv", delimiter=' ').to_numpy()
konishi_ttp = pd.read_csv(base_path / "Konishi_l_Catimetopeak.csv", delimiter=' ').to_numpy()
blinks_peak = pd.read_csv(base_path / "Blinks_l_Capeak.csv", delimiter=' ').to_numpy()

# Normalize data
# Peak amplitude from Blinks was digitized in percentage
blinks_peak[:, 1] /= 100.0

# Normalize frog sarcomere length by optimal sarcomere length = 2.1 um
for arr in (blinks_peak, konishi_peak, konishi_ttp):
    arr[:, 0] /= 2.1

# Correct digitized peak values slightly above 1
blinks_peak[:, 1] = np.minimum(blinks_peak[:, 1], 1.0)
konishi_peak[:, 1] = np.minimum(konishi_peak[:, 1], 1.0)


# Fit f1: Ca2+ peak amplitude vs normalized sarcomere length
peak_data = np.concatenate((blinks_peak, konishi_peak), axis=0)
peak_data = peak_data[np.argsort(peak_data[:, 0])]  # sort by length

plateau_start = 1.1379
plateau_end = 1.239
plateau_value = 1.0

left_mask = peak_data[:, 0] < plateau_start
right_mask = peak_data[:, 0] > plateau_end

x_left = peak_data[left_mask, 0]
y_left = peak_data[left_mask, 1]
x_right = peak_data[right_mask, 0]
y_right = peak_data[right_mask, 1]

# Linear segment on the left, constrained to reach the plateau
x_first_left = x_left[0]
y_first_left = y_left[0]

slope_left = (plateau_value - y_first_left) / (plateau_start - x_first_left)
intercept_left = y_first_left - slope_left * x_first_left

# Linear segment on the right, constrained to start from plateau_value at plateau_end
def linear_model_right(x, slope):
    return plateau_value + slope * (x - plateau_end)

popt_right, _ = curve_fit(linear_model_right, x_right, y_right)
slope_right = popt_right[0]

# Smooth piecewise representation of f1
l_sm = np.linspace(0.4, 2.3, 500)
Ca_sm = np.zeros_like(l_sm)

for i, x in enumerate(l_sm):
    if x < x_first_left:
        Ca_sm[i] = y_first_left
    elif x < plateau_start:
        Ca_sm[i] = slope_left * x + intercept_left
    elif x <= plateau_end:
        Ca_sm[i] = plateau_value
    else:
        Ca_sm[i] = plateau_value + slope_right * (x - plateau_end)

print("1) y_left =", y_first_left)
print("1) x_left =", x_first_left)
print("2) Left intercept =", intercept_left)
print("2) Left slope =", slope_left)
print("3) Slope right =", slope_right)

# Fit f2: time-to-peak Ca2+ vs normalized sarcomere length
ttp_data = konishi_ttp[np.argsort(konishi_ttp[:, 0])]  # sort by length

p2 = np.polyfit(ttp_data[:, 0], ttp_data[:, 1], deg=2)
fit2 = np.poly1d(p2)

l_smttp = np.linspace(ttp_data[:, 0].min(), ttp_data[:, 0].max(), 500)
Ca_smttp = fit2(l_smttp)

# Constant extrapolation outside the experimental range
l_left = np.linspace(0.6, ttp_data[:, 0].min(), 100)
left_smttp = np.ones(len(l_left)) * 0.73823954

l_right = np.linspace(ttp_data[:, 0].max(), 2.4, 100)
right_smttp = np.ones(len(l_right)) * 1.072

# Plot
fig, axes = plt.subplots(2, 1, figsize=(6, 6), dpi=200)

# f1
axes[0].plot(blinks_peak[:, 0], blinks_peak[:, 1], 'rx', label='Blinks 1978')
axes[0].plot(konishi_peak[:, 0], konishi_peak[:, 1], 'gx', label='Konishi 1991')
axes[0].plot(l_sm, Ca_sm, 'k', label='Fit')
axes[0].set_xlim([0.8, 2.1])
axes[0].set_ylim([0, 1.2])
axes[0].set_ylabel(r'$\overline{\mathbf{f_1}}$')
axes[0].grid()
axes[0].set_xticklabels([])

axes[0].text(
    0.02, 0.95, 'A',
    transform=axes[0].transAxes,
    fontsize=12,
    fontweight='bold',
    va='top'
)

# f2
axes[1].plot(ttp_data[:, 0], ttp_data[:, 1], 'gx', label='Konishi 1991')
axes[1].plot(l_smttp, Ca_smttp, 'k', label='Fit')
axes[1].plot(l_left, left_smttp, 'k')
axes[1].plot(l_right, right_smttp, 'k')
axes[1].set_xlim([0.8, 2.1])
axes[1].set_xlabel(r'$\overline{\mathbf{l}^{CE}}$')
axes[1].set_ylabel(r'$\overline{\mathbf{f_2}}$')
axes[1].grid()

axes[1].text(
    0.02, 0.95, 'B',
    transform=axes[1].transAxes,
    fontsize=12,
    fontweight='bold',
    va='top'
)

fig.savefig("calcium_fit.tif", dpi=500, bbox_inches="tight")
plt.show()
