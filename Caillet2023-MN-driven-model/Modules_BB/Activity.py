"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBME
Created on Wed Jun 26 16:36:01 2024
____________________________________

Description:
"""

def activity(Ca, w_l):
    
    if Ca < 0:
        Ca = 0
    # Belnave 1996, tetanic Cmax value is 0.6 micromol for mouse
    a = (0.01 + (((Ca/(13*10**-6))*w_l)**2.66))/(1 + ((Ca/(13*10**-6))*w_l)**2.66)
    #a = (0.01 + (((Ca/(0.6*10**-6))*w_l)**2.66))/(1 + ((Ca/(0.6*10**-6))*w_l)**2.66)
    
    return a