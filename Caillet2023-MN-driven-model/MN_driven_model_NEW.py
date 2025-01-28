# -*- coding: utf-8 -*-
"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE

Created on Tue Jan 28 15:33:16 2025
___________________________________

"""

import numpy as np
import scipy as sp
import scipy.interpolate
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class MN_driven_model():

    
    def __init__(self, parameters, states, benchmark, discharges):
        
        if parameters is None:
            raise ValueError('No parameters set!') 
        elif isinstance(parameters, dict) == False:
            raise ValueError('Parameters are not in dict format!') 
        elif isinstance(parameters, dict) == True:
            self.P = parameters # set parameters
            
        if states is None:
            raise ValueError('No states set!') 
        elif isinstance(states, dict) == False:
            raise ValueError('States are not in dict format!') 
        elif isinstance(states, dict) == True:
            self.S = states # set parameters
          
        if benchmark == 'Maximal':
            self.freq = 70
        elif benchmark == 'Submaximal':
            self.disch = discharges    
        else:
            raise ValueError('Benchmark type was not specified! (either ''Maximal'' or ''Submaximal''') 
            

