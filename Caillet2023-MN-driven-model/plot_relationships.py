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
from Force_Length_MOD import Force_Length_func

points = 300 # x-points for plots
n_a = 5 # n. different active states [0-1]
fibre_type = 2 # n. different fibre types (fast & slow)
l_T_slack = 24

# preallocating...
f_T = np.empty((points),dtype=object)
f_PE = np.empty((points),dtype=object)
fl = np.empty((n_a,points),dtype=object)
fv = np.empty((n_a,points,fibre_type),dtype=object)

# defining x-axis quantities...
a = np.linspace(0.2,1,n_a)
MU_type = ['fast','slow']
l_M = np.linspace(0.7, 1.8,points)
v_M = np.linspace(-1.2,1.2,points)
eps_T = np.linspace(0,0.04,points)
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


# calculating the force values...
for i in range(points):
    f_T[i] = T_force(eps_T[i])
    f_PE[i] = PEE_force(l_M[i])
    for l in range(n_a):
        fl[l,i] = Force_Length_func(l_M[i],a[l])*a[l]
        for t in range(fibre_type):
            fv[l,i,t] = f_fFV(v_M[i], fl[l,i], a[l], 1, MU_type[t]) # for fixed l_M = 1
    

 # f_T_2 = np.empty((points),dtype=object)
 # for i in range(points):
 #     f_T_2[i] = T_force(eps_T[i])

   
# plotting...
plt.rcParams['figure.dpi'] = 360
figure(figsize=(12, 10))
fig = plt.subplot(2,2,1)
plt.plot(eps_T*100, f_T, 'b', label = 'John et al. 2013')
#plt.plot(l_T, f_T, 'b', label = 'John et al. 2013')
#plt.plot(eps_T*100, f_T_2, 'b', linestyle = 'dashed', label = 'Caillet et al. 2023')
plt.xlabel(r'$\epsilon^T$ [%]')
plt.ylabel('$\overline {F^T}$')
plt.grid()
plt.title('SEE (Tendon)')
plt.legend()

plt.subplot(2,2,2)
plt.plot(l_M, f_PE)
plt.xlabel('Norm. Lm')
plt.ylabel('Norm. PE force')
plt.grid()
plt.title('PEE')

plt.subplot(2,2,3)
plt.plot(l_M, fl[0,:],'r', label = 'a = 0.2')
plt.plot(l_M, fl[1,:], 'b', label = 'a = 0.4')
plt.plot(l_M, fl[2,:], 'y', label = 'a = 0.6')
plt.plot(l_M, fl[3,:], 'g', label = 'a = 0.8')
plt.plot(l_M, fl[4,:], 'k', label = 'a = 1.0')
plt.xlabel('Norm. Lm')
plt.ylabel('Norm. F-L force * a')
plt.grid()
plt.legend()
plt.title('Active F-L')

plt.subplot(2,2,4)
plt.plot(v_M, fv[0,:,0], label = 'a = 0.2, fast, lm = 1')
plt.plot(v_M, fv[2,:,0], label = 'a = 0.6, fast, lm = 1')
plt.plot(v_M, fv[4,:,0], label = 'a = 1.0, fast, lm = 1')
plt.plot(v_M, fv[0,:,1], label = 'a = 0.2, slow, lm = 1')
plt.plot(v_M, fv[2,:,1], label = 'a = 0.6, slow, lm = 1')
plt.plot(v_M, fv[4,:,1], label = 'a = 1.0, slow, lm = 1')
plt.xlabel('Norm. Vm')
plt.ylabel('Norm. F-V force')
plt.grid()
plt.legend()
plt.title('Active F-V')

plt.show()