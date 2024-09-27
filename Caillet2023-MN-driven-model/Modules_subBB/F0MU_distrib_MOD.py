""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2023
Imperial College London
Department of Civil Engineering
Function necessary to compute the results presented in the manuscript Caillet et al. 'Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy' (2023)
---------

Computes, using results from the literature, a distribution of MU max iso forces F0MU(j) for the simulated population of N or Nr MUs.
If only the experimental population of Nr MUs identified from HDEMG is considered, two approches are considered to deirved the F0MU distribution across that experimental sample
evenly: the Nr MUs are supposed to be homogeneously spread across the MU pool
identified: the Nr Mus are identified into the MU according to their torque recruitment thresholds and their representative F0MU values are adapted accordingly

"""
import numpy as np
from MN_properties_relationships_MOD import F0MU_norm_distrib_func

def F0MU_distrib_func(Nr, F0M):
    
    MU_pool_list = np.arange(1, Nr+1, 1) #list of all TA MUs = 1:1:N(400)
    
    #### Let's scale the normalized distribution of MU F0MU (from literature) to Newtons
    # Let's compute the sum of the normalized F0MU(i)
    cumulative_all_MUs=sum(F0MU_norm_distrib_func(MU_pool_list,  Nr))
    # Obtaining, using the subject-specific F0M, the N-% relationship
    scale_factor = F0M/cumulative_all_MUs
    # and scaling the normalized distribution of F0MU 
    F0MU_distribution_complete_MU_pool = F0MU_norm_distrib_func(MU_pool_list,  Nr) * scale_factor
    
    F0MU_distribution = F0MU_distribution_complete_MU_pool[0:Nr]
    F0MU_distribution = np.reshape(F0MU_distribution, (Nr,1))

    return F0MU_distribution


# plt.scatter(MU_list_identified, F0MU_distrib_func(MVC, Nr, MN_pop, F0M, Input, Real_MN_pop, spread='evenly'), s=1)
