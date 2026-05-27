
"""
Author: Andrea Sgarzi
Email: a.sgarzi@ad.unsw.edu.au
Affiliation: University of New South Wales (UNSW), Graduate School of Biomedical Engineering (GSBE)

Benchmark trial definitions used by run_benchmark.py.
"""

import numpy as np

BENCHMARK_TRIALS = {
    # _____________________________________________________________________
    # Muscle scale - slow muscle
    # _____________________________________________________________________

    # Rat SOL, dyn2 benchmarks (Krylow, Sandercock et al. 1997)
    "slow_M_dyn2_0.05mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "dyn2", "amplitude_mm": "0.05", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
        "optimization": {
            "label": "M_S_dyn2: optimise MVC on 0.05 mm dynamic trial",
            "parameters": ["MVC"],
            "x0": [1.2],
            "bounds": [[1.1, 1.5]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "slow_M_dyn2_0.10mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "dyn2", "amplitude_mm": "0.10", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },
    "slow_M_dyn2_0.25mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "dyn2", "amplitude_mm": "0.25", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },
    "slow_M_dyn2_0.50mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "dyn2", "amplitude_mm": "0.50", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },    
    "slow_M_dyn2_1.00mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "dyn2", "amplitude_mm": "1.00", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },    
    "slow_M_dyn2_2.00mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "dyn2", "amplitude_mm": "2.00", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,         
        "optimization": {
            "label": "M_S_dyn2: optimise af_s on 2.00 mm dynamic trial",
            "parameters": ["af_s"],
            "x0": [0.4],
            "bounds": [[0.1, 1.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },

    # Cat SOL, iso-f benchmarks (Perreault et al. 2003)
    "slow_M_sub_iso_c10": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isof", "stim": "c", "freq": 10, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },
    "slow_M_sub_iso_c20": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isof", "stim": "c", "freq": 20, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },
    "slow_M_sub_iso_c30": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isof", "stim": "c", "freq": 30, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
        "optimization": {
            "label": "M_S_iso_f: optimise MVC, Ca_max_s_M, k1_s_M, k2_s_M on 30 Hz (constant) isometric trial",
            "parameters": ["MVC", "Ca_max_s_M", "k1_s_M", "k2_s_M"],
            "x0": [27.0, 5e5, 10.0, 14.0],
            "bounds": [[25.0, 30.0], [1e5, 1e6], [10.0, 20.0], [10.0, 20.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "slow_M_sub_iso_v10": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isof", "stim": "v", "freq": 10, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },
    "slow_M_sub_iso_v20": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isof", "stim": "v", "freq": 20, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },    
    "slow_M_sub_iso_v30": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isof", "stim": "v", "freq": 10, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },

    # Cat SOL, iso-l benchmarks (Perreault et al. 2003, Kim et al. 2015)
    "slow_M_len_0mm_1Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "0", "freq": 1, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_0mm_10Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "0", "freq": 10, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_0mm_20Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "0", "freq": 20, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_0mm_40Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "0", "freq": 40, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_8mm_1Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "8", "freq": 1, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },    
    "slow_M_len_8mm_10Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "8", "freq": 10, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_8mm_20Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "8", "freq": 20, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_8mm_40Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "8", "freq": 40, "t_end": 1.4, 
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,          
        "optimization": {
            "label": "M_S_iso_l: optimise MVC on 40 Hz isometric (0 mm offset) length trial",
            "parameters": ["MVC"],
            "x0": [28.0],
            "bounds": [[25.0, 31.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "slow_M_len_16mm_1Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "16", "freq": 1, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_16mm_10Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "16", "freq": 10, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_16mm_20Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "16", "freq": 20, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "slow_M_len_16mm_40Hz": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "isol", "length": "16", "freq": 40, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },

    # Cat SOL, dyn1 benchmarks (Perreault et al. 2003)
    "slow_M_sub_dyn_c10_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "c", "freq": 10, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_c10_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "c", "freq": 10, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_c20_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "c", "freq": 20, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_c20_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "c", "freq": 20, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_c30_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "c", "freq": 30, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_c30_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "c", "freq": 30, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_v10_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "v", "freq": 10, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_v10_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "v", "freq": 10, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_v20_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "v", "freq": 20, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_v20_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "v", "freq": 20, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_v30_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "v", "freq": 30, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "slow_M_sub_dyn_v30_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "dyn1", "stim": "v", "freq": 30, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },

    # ____________________________________________________________________________
    # Muscle scale - fast muscle
    # ____________________________________________________________________________

    # Rat EDL muscle, iso-f (experiments ad hoc for the study)
    "fast_M_isof_1Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 1,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "fast_M_isof_30Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 30,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "fast_M_isof_50Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 50,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "fast_M_isof_60Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 60,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "fast_M_isof_70Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 70,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "fast_M_isof_80Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },    
    "fast_M_isof_90Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 90,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "fast_M_isof_100Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 100,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "fast_M_isof_120Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isof", "freq": 120,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,       
        "optimization": {
            "label": "M_F_isof: optimise l_M_0 on 120 Hz isometric trial",
            "parameters": ["l_M_0"],
            "x0": [3.0],
            "bounds": [[1.0, 6.6]],
            "method": "Nelder-Mead",
            "target": "force",
            "rebuild_par": True,
        },
    },

    # Rat EDL muscle, iso-l (experiments ad hoc for the study)
    "fast_M_isol_0.5mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "0.5",
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,               
        "optimization": {
            "label": "M_F_isol: optimise MVC, Ca_max_f_M, k1_f_M, k2_f_M on 80 Hz isometric length trial",
            "parameters": ["MVC", "Ca_max_f_M", "k1_f_M", "k2_f_M"],
            "x0": [2.0, 5e5, 10.0, 10.0],
            "bounds": [[1.9, 2.9], [1e5, 1e6], [10.0, 15.0], [10.0, 15.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "fast_M_isol_1mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "1.0", 
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "fast_M_isol_1.5mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "1.5", 
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "fast_M_isol_2.mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "2.0", 
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "fast_M_isol_2.5mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "2.5", 
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "fast_M_isol_3mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "3.0", 
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "fast_M_isol_3.5mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "3.5", 
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "fast_M_isol_4mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "isol", "length_mm": "4.0", 
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },

    # Cat CF muscle, dyn (Brown et al. 1999)
    "fast_M_dyn_120_short_0.95": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "short", "freq": 120, "t_end": 0.16,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
        "optimization": {
            "label": "M_F_dyn: optimise af_f on -3 l0/s shortening trial at 120 Hz",
            "parameters": ["af_f"],
            "x0": [0.4],
            "bounds": [[0.1, 3.0]],
            "method": "Nelder-Mead",
            "target": "force_dynamic_ratio",
        },
    },
    "fast_M_dyn_120_length_0.95": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "length", "l_M_0_scale": "0.95", "freq": 120, "t_end": 0.16,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_20_short_0.95": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "short", "l_M_0_scale": "0.95", "freq": 20, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_20_length_0.95": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "length", "l_M_0_scale": "0.95", "freq": 20, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_40_short_0.8": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "short", "l_M_0_scale": "0.8", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.80, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_40_length_0.8": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "length", "l_M_0_scale": "0.8", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.80, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_40_short_1.1": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "short", "l_M_0_scale": "1.1", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*1.10, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_40_length_1.1": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "length", "l_M_0_scale": "1.1", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*1.10, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_60_short_0.95": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "short", "l_M_0_scale": "0.95", "freq": 60, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "fast_M_dyn_60_length_0.95": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "dyn", "trial": "length", "l_M_0_scale": "0.95", "freq": 60, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },

    # __________________________________________________________________
    # Motor-unit scale
    # __________________________________________________________________

    # S MU, cat LG (Celichowski et al. 1999)
    "cat_LG_1Hz": {
        "scale": "MU", "muscle": "cat_LG", "freq": 1, "benchmark": "MU_S",
        "MVC": 0.04, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
        "t_end": 1.8, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
        },
    "cat_LG_12.5Hz": {
        "scale": "MU", "muscle": "cat_LG", "freq": 12.5, "benchmark": "MU_S",
        "MVC": 0.04, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
        "t_end": 1.8, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        },
    "cat_LG_40Hz": {
        "scale": "MU", "muscle": "cat_LG", "freq": 40, "benchmark": "MU_S",
        "MVC": 0.04, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
        "t_end": 1.8, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
        "optimization": {
            "label": "cat_LG_40Hz: optimise active state parameters on 40 Hz catMG isometric trial",
            "parameters": ["MVC","Ca_max_s_MU_catLG", "k1_s_MU_catLG", "k2_s_MU_catLG"],
            "x0": [0.078, 5e5, 10, 10],
            "bounds": [[0.06, 0.08], [1e5, 1e6], [10, 20], [10, 20]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },

    # FF MU, cat MG (Burke et al. 1974)
    #"cat_MG_1Hz": {
    #    "scale": "MU", "muscle": "cat_MG", "freq": 1,
    #    "MVC": 0.81, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
    #    "t_end": 1.0, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,        
    #              },
    #"cat_MG_25Hz": {
    #    "scale": "MU", "muscle": "cat_MG", "freq": 25,
    #    "MVC": 0.81, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
    #    "t_end": 1.0, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
    #    },
    #"cat_MG_40Hz": {
    #    "scale": "MU", "muscle": "cat_MG", "freq": 40,
    #    "MVC": 0.81, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
    #    "t_end": 1.0, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,        
    #    },

    # FR MU, cat MG (Celichowski et al. 1974)
    "cat_MG_1Hz": {
        "scale": "MU", "muscle": "cat_MG", "freq": 1, "benchmark": "MU_FR",
        "MVC": 0.4, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
        "t_end": 1.2, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        },
    "cat_MG_20Hz": {
        "scale": "MU", "muscle": "cat_MG", "freq": 20, "benchmark": "MU_FR",
        "MVC": 0.4, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
        "t_end": 1.2, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
        },
    "cat_MG_40Hz": {
        "scale": "MU", "muscle": "cat_MG", "freq": 40, "benchmark": "MU_FR",
        "MVC": 0.4, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
        "t_end": 1.2, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        "optimization": {
            "label": "MU_F_isof: optimise sag parameters on 40 Hz ratMG isometric trial",
            "parameters": ["MVC","Ca_max_f_MU_catMG", "k1_f_MU_catMG", "k2_f_MU_catMG"],
            "x0": [0.9, 5e5, 10, 10],
            "bounds": [[0.2, 1.4], [1e5, 1e6], [10, 100], [10, 100]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },

    # FR MU, rat MG (Celichowski et al. 1999)
    "rat_MG_25Hz": {
        "scale": "MU", "muscle": "rat_MG", "freq": 25, "benchmark": "MU_FR",
        "MVC": 0.073, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 20*np.pi/180,
        "t_end": 0.7, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
        },
    "rat_MG_30Hz": {
        "scale": "MU", "muscle": "rat_MG", "freq": 30, "benchmark": "MU_FR",
        "MVC": 0.073, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 20*np.pi/180,
        "t_end": 0.7, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
        "optimization": {
            "label": "MU_F_isof: optimise sag parameters on 30 Hz ratMG isometric trial",
            "parameters": ["As_peak", "As_decay", "Ts"],
            "x0": [1.2, 0.9, 0.1],
            "bounds": [[1.0, 3.0], [0.1, 1.0], [0.01, 2.0]],
            "method": "Nelder-Mead",
            "target": "force",
            },
        },
    "rat_MG_35Hz": {
        "scale": "MU", "muscle": "rat_MG", "freq": 35, "benchmark": "MU_FR",
        "MVC": 0.073, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 20*np.pi/180,
        "t_end": 0.7, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
        },
    "rat_MG_40Hz": {
        "scale": "MU", "muscle": "rat_MG", "freq": 40, "benchmark": "MU_FR",
        "MVC": 0.073, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 20*np.pi/180,
        "t_end": 0.7, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": True,
        },
    "rat_MG_150Hz": {
        "scale": "MU", "muscle": "rat_MG", "freq": 150, "benchmark": "MU_FR",
        "MVC": 0.073, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 20*np.pi/180,
        "t_end": 0.7, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        "optimization": {
            "label": "MU_F_isof: optimise MVC, Ca_max_f_MU_ratGM, k1_f_MU_ratGM, k2_f_MU_ratGM on 150 Hz tetanic trial",
            "parameters": ["MVC", "Ca_max_f_MU_ratGM", "k1_f_MU_ratGM", "k2_f_MU_ratGM"],
            "x0": [0.078, 5e5, 10.0, 10.0],
            "bounds": [[0.06, 0.08], [1e5, 1e6], [10.0, 100.0], [10.0, 100.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },

    # __________________________________________________________________
    # Calcium transients
    # __________________________________________________________________

    # Slow fibres (Calderon, Caputo et al. 2014, 2021)
    "Ca_slow_23_100Hz": {
        "scale": "Ca_transients", "muscle": "rat_SOL", "freq": 102, "benchmark": "Ca_slow_23_100Hz",
        "MVC": 0, "l_T_slack": 0, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 0,
        "t_end": 1.4, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        "optimization": {
            "label": "Ca transient: optimise c1_s, c2_s, c3_s",
            "parameters": ["c1_s", "c2_s", "c3_s"],
            "x0": [6.029e3, 1.8e5, 0.54],
            "bounds": [[1e3, 1e5], [1e5, 1e6], [0.1, 2.0]],
            "method": "Nelder-Mead",
            "target": "calcium",
            "maxiter": 300,
        },
    },

    # Fast fibres (Hollingworth et al. 1996)
    "Ca_fast_35_125Hz": {
        "scale": "Ca_transients", "muscle": "rat_EDL", "freq": 125, "benchmark": "Ca_fast_35_125Hz",
        "MVC": 0, "l_T_slack": 0, "l_M_opt": 30, "l_M_0": 1.6*30, "alpha_0": 0,
        "t_end": 1.4, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        "optimization": {
            "label": "Ca transient: optimise c1_f, c2_f, c3_f",
            "parameters": ["c1_f", "c2_f", "c3_f"],
            "x0": [2.4e3, 4.3e5, 0.6],
            "bounds": [[1e3, 1e5], [1e5, 1e6], [0.1, 2.0]],
            "method": "Nelder-Mead",
            "target": "calcium",
            "maxiter": 300,
        },
    },
}