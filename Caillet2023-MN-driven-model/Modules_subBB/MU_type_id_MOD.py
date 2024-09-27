""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2023
Imperial College London
Department of Civil Engineering
Function necessary to compute the results presented in the manuscript Caillet et al. 'Motoneuron-driven computational muscle modelling with motor unit resolution and subject-specific musculoskeletal anatomy' (2023)
---------

This function identifies, according to the location of the investigated MU into the MU pool (in ascending order of Fth), whether the MU is slow- or fast-type.
Please see manuscript (Methods) for details. 

"""

def MU_type_id_func(i):
    if i < 3: MU_type='slow' # 85 % slow MUs (reference is for TA human muscle, to be corrected)
    else: MU_type='fast'

    return MU_type