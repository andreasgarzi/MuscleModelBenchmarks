"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBME
Created on Wed Jun 26 16:32:46 2024
____________________________________

Description:
"""

def shift_fun(l_M): 
    
    w_l =  8.28*((1-(2.86/8.28)**(1/3.68))*l_M + (2.86/8.28)**(1/3.68))**3.68
    
    return w_l