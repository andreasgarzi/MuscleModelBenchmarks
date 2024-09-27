"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Sat Jun  1 14:57:36 2024
___________________________________

Plot all the element relationships of the model
"""

import sys
sys.path.insert(0,'Modules')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from PE_force import PEE_force
from Tendon_force import T_force
from force_fFV import f_fFV
from velocity_fFV import velo_fFV
from Force_Length_MOD import Force_Length_func

points = 300 # x-points for plots
n_a = 5 # n. different active states [0-1]
fibre_type = 2 # n. different fibre types (fast & slow)
l_T_slack = 24

# preallocating...
f_T0 = np.empty((points),dtype=object)
f_T1 = np.empty((points),dtype=object)
f_T2 = np.empty((points),dtype=object)
f_T3 = np.empty((points),dtype=object)
f_T4 = np.empty((points),dtype=object)
f_PE0 = np.empty((points),dtype=object)
f_PE1 = np.empty((points),dtype=object)
f_PE2 = np.empty((points),dtype=object)
f_PE3 = np.empty((points),dtype=object)
f_PE4 = np.empty((points),dtype=object)
f_PE5 = np.empty((points),dtype=object)
f_PE6 = np.empty((points),dtype=object)
fl = np.empty((n_a,points),dtype=object)
fv = np.empty((n_a,points,fibre_type),dtype=object)
vf = np.empty((n_a,points,fibre_type),dtype=object)

# defining x-axis quantities...
a = np.linspace(0.2,1,n_a)
MU_type = ['fast','slow']
l_M = np.linspace(0.7, 1.8,points)
l_M_PE = np.linspace(0.9, 2.2,points)
v_M = np.linspace(-1.2,1.2,points)
eps_T = np.linspace(0,0.07,points)
l_T = (eps_T*l_T_slack + l_T_slack)/l_T_slack

#%% MU recruitment threshold analysis

# MN_pop = 27
# MN = np.arange(1, MN_pop+1, 1)
# F_MU = 0.000786*(3.0*MN/MN_pop+8.20**((MN/MN_pop)**5.29))

# plt.figure()
# plt.plot(MN, F_MU)
# plt.grid()
# plt.show()

#%%
"""Calculate relationship values"""

# calculating the force values...
for i in range(points):
    f_T0[i] = T_force(eps_T[i], 0)
    f_T1[i] = T_force(eps_T[i], 1)
    f_T2[i] = T_force(eps_T[i], 2)
    f_T3[i] = T_force(eps_T[i], 3)
    f_T4[i] = T_force(eps_T[i], 4)
    
    f_PE0[i] = PEE_force(l_M_PE[i], 0)
    f_PE1[i] = PEE_force(l_M_PE[i], 1)
    f_PE2[i] = PEE_force(l_M_PE[i], 2)
    f_PE3[i] = PEE_force(l_M_PE[i], 3)
    f_PE4[i] = PEE_force(l_M_PE[i], 4)
    f_PE5[i] = PEE_force(l_M_PE[i], 5)
    f_PE6[i] = PEE_force(l_M_PE[i], 6)
    
    for l in range(n_a):
        fl[l,i] = Force_Length_func(l_M[i],a[l])*a[l]
        for t in range(fibre_type):
            fv[l,i,t] = f_fFV(v_M[i], fl[l,i], a[l], 1, MU_type[t]) # for fixed l_M = 1
            vf[l,i,t] = velo_fFV(fv[l,i,t], fl[l,i], a[l], 1, MU_type[t])

 # f_T_2 = np.empty((points),dtype=object)
 # for i in range(points):
 #     f_T_2[i] = T_force(eps_T[i])

#______________________________________________________________________________
"""Create color map and plot relationships"""
   
c1 = "#0b165e"  #dark blue
c2 = "#00eeff"  #light blue

c1t = "#0c520b"  #dark green
c2t = "#18e314"  #light green

c1PE1 = '#eb7adc' #light pink
c2PE1 = '#440a66'

c1PE2 = '#7ac1eb' #light blue 2
c2PE2 = '#5205ed'

c3 = '#e31010'  #red
c4 = '#ff9900'  #orange

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

cg1 = get_color_gradient(c1, c2, 5)
cgt = get_color_gradient(c1t, c2t, 5)
cgPE1 = get_color_gradient(c1PE1, c2PE1, 3)
cgPE2 = get_color_gradient(c1PE2, c2PE2, 3)
cg2 = get_color_gradient(c1, c2, 3)
cg3 = get_color_gradient(c3, c4, 3)


# plotting...
plt.rcParams['figure.dpi'] = 360
figure(figsize=(12, 10))
fig = plt.subplot(2,2,1)
ax = fig
plt.plot(eps_T*100, f_T0, color=cgt[0], label = r'$\epsilon^T_0$ = 1.3%')
plt.plot(eps_T*100, f_T1, color=cgt[1], label = r'$\epsilon^T_0$ = 2.3%')
plt.plot(eps_T*100, f_T2, color=cgt[2], label = r'$\epsilon^T_0$ = 3.3%')
plt.plot(eps_T*100, f_T3, color=cgt[3], label = r'$\epsilon^T_0$ = 4.3%')
plt.plot(eps_T*100, f_T4, color=cgt[4], label = r'$\epsilon^T_0$ = 5.3%')
ax.annotate(r'$\epsilon^Ttoe$ = 0.609($\epsilon^T_0$)', xy=(0.1,1))
#plt.plot(l_T, f_T, 'b', label = 'John et al. 2013')
plt.xlabel(r'$\epsilon^T$ [%]')
plt.ylabel('$\overline {F^T}$')
plt.grid()
plt.title('SEE (Tendon)')
plt.legend()
plt.ylim((-0.25, 4))

plt.subplot(2,2,2)
#plt.plot(l_M_PE, f_PE0, 'k--', label='k1 = 5, k2 = 0.6')
plt.plot(l_M_PE, f_PE1, color=cg2[1], label='k1 = 5, k2 = 0.8') 
plt.plot(l_M_PE, f_PE2, color=cg2[1], linestyle='dashed', label='k1 = 5, k2 = 1.0') 
plt.plot(l_M_PE, f_PE3, color=cg2[1], linestyle='dashdot', label='k1 = 5, k2 = 1.2') 
plt.plot(l_M_PE, f_PE4, color=cg3[0], label='k1 = 4, k2 = 0.6') 
plt.plot(l_M_PE, f_PE5, color=cg3[0], linestyle='dashed', label='k1 = 3, k2 = 0.6') 
plt.plot(l_M_PE, f_PE6, color=cg3[0], linestyle='dashdot', label='k1 = 2, k2 = 0.6') 
plt.xlabel('$\overline {L^M}$')
plt.ylabel('$\overline {F^M}$')
plt.grid()
plt.title('PEE')
plt.legend()
plt.ylim((-0.25, 4))

plt.subplot(2,2,3)
plt.plot(l_M, fl[0,:], color=cg1[0], label = 'a = 0.2')
plt.plot(l_M, fl[1,:], color=cg1[1], label = 'a = 0.4')
plt.plot(l_M, fl[2,:], color=cg1[2], label = 'a = 0.6')
plt.plot(l_M, fl[3,:], color=cg1[3], label = 'a = 0.8')
plt.plot(l_M, fl[4,:], color=cg1[4], label = 'a = 1.0')
plt.xlabel('$\overline {L^M}$')
plt.ylabel('$\overline {F^M_l}$')
plt.grid()
plt.legend()
plt.title('Active F-L')

plt.subplot(2,2,4)
plt.plot(v_M, fv[0,:,0], color=cg2[0], label = 'a = 0.2, fast')
plt.plot(v_M, fv[2,:,0], color=cg2[0], linestyle='dashed', label = 'a = 0.6, fast')
plt.plot(v_M, fv[4,:,0], color=cg2[0], linestyle='dashdot',label = 'a = 1.0, fast')
plt.plot(v_M, fv[0,:,1], color=cg3[1], label = 'a = 0.2, slow')
plt.plot(v_M, fv[2,:,1], color=cg3[1], linestyle='dashed', label = 'a = 0.6, slow')
plt.plot(v_M, fv[4,:,1], color=cg3[1], linestyle='dashdot',label = 'a = 1.0, slow')

plt.plot(vf[0,:,0], fv[0,:,0], color=cg2[0], label = 'a = 0.2, fast')
plt.plot(vf[2,:,0], fv[2,:,0], color=cg2[0], linestyle='dashed', label = 'a = 0.6, fast')
plt.plot(vf[4,:,0], fv[4,:,0], color=cg2[0], linestyle='dashdot',label = 'a = 1.0, fast')
plt.plot(vf[0,:,1], fv[0,:,1], color=cg3[1], label = 'a = 0.2, slow')
plt.plot(vf[2,:,1], fv[2,:,1], color=cg3[1], linestyle='dashed', label = 'a = 0.6, slow')
plt.plot(vf[4,:,1], fv[4,:,1], color=cg3[1], linestyle='dashdot',label = 'a = 1.0, slow')

plt.xlabel('$\overline {V^M}$')
plt.ylabel('$\overline {F^M_v}$')
plt.grid()
plt.legend()
plt.title('Active F-V, $\overline {L^M}$ = 0.8')

plt.suptitle('MN-driven model sensitivity (contractile part)', weight='bold', y=0.94)
plt.show()


