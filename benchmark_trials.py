
"""
Author: Andrea Sgarzi
Email: a.sgarzi@ad.unsw.edu.au
Affiliation: University of New South Wales (UNSW), Graduate School of Biomedical Engineering (GSBE)

Benchmark trial definitions used by run_benchmark.py.
"""

import numpy as np

benchmark_Trials = {
    # _____________________________________________________________________
    # Muscle scale - slow muscle
    # _____________________________________________________________________

    # Rat SOL, slow_dyn2 benchmarks (Krylow, Sandercock et al. 1997)
    "rat_SOL_0.05mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "slow_dyn2", "amplitude_mm": "0.05", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
        "optimization": {
            "label": "Muscle slow dyn2: optimise MVC on 0.05 mm dynamic trial",
            "parameters": ["MVC"],
            "x0": [1.2],
            "bounds": [[1.1, 1.5]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "rat_SOL_0.10mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "slow_dyn2", "amplitude_mm": "0.10", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },
    "rat_SOL_0.25mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "slow_dyn2", "amplitude_mm": "0.25", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },
    "rat_SOL_0.50mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "slow_dyn2", "amplitude_mm": "0.50", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },    
    "rat_SOL_1.00mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "slow_dyn2", "amplitude_mm": "1.00", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,  
    },    
    "rat_SOL_2.00mm": {
        "scale": "Muscle", "muscle": "rat_SOL", "benchmark": "slow_dyn2", "amplitude_mm": "2.00", "freq": 70, "t_end": 2.0,
        "MVC": 1.32, "l_T_slack": 17.1, "l_M_opt": 17.1, "l_M_0": 17.1, "alpha_0": 6*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,         
        "optimization": {
            "label": "Muscle slow dyn2: optimise af_s on 2.00 mm dynamic trial",
            "parameters": ["af_s"],
            "x0": [0.4],
            "bounds": [[0.1, 1.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },

    # Cat SOL, isof benchmarks (Perreault et al. 2003)
    "cat_SOL_10Hz_c": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isof", "stim": "c", "freq": 10, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },
    "cat_SOL_20Hz_c": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isof", "stim": "c", "freq": 20, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },
    "cat_SOL_30Hz_c": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isof", "stim": "c", "freq": 30, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
        "optimization": {
            "label": "Muscle slow isof: optimise MVC, Ca_max_s_M, k1_s_M, k2_s_M on 30 Hz (constant) isometric trial",
            "parameters": ["MVC", "Ca_max_s_M", "k1_s_M", "k2_s_M"],
            "x0": [27.0, 5e5, 10.0, 14.0],
            "bounds": [[25.0, 30.0], [1e5, 1e6], [10.0, 20.0], [10.0, 20.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "cat_SOL_10Hz_v": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isof", "stim": "v", "freq": 10, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },
    "cat_SOL_20Hz_v": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isof", "stim": "v", "freq": 20, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },    
    "cat_SOL_30Hz_v": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isof", "stim": "v", "freq": 30, "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False, 
    },

    # Cat SOL, isol benchmarks (Perreault et al. 2003, Kim et al. 2015)
    "cat_SOL_1Hz_0mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "0", "freq": 1, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_10Hz_0mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "0", "freq": 10, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_20Hz_0mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "0", "freq": 20, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_40Hz_0mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "0", "freq": 40, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_1Hz_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "8", "freq": 1, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },    
    "cat_SOL_10Hz_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "8", "freq": 10, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_20Hz_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "8", "freq": 20, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_40Hz_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "8", "freq": 40, "t_end": 1.4, 
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,          
        "optimization": {
            "label": "Muscle slow isol: optimise MVC on 40 Hz isometric (0 mm offset) length trial",
            "parameters": ["MVC"],
            "x0": [28.0],
            "bounds": [[25.0, 31.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "cat_SOL_1Hz_16mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "16", "freq": 1, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_10Hz_16mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "16", "freq": 10, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_20Hz_16mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "16", "freq": 20, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },
    "cat_SOL_40Hz_16mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_isol", "length": "16", "freq": 40, "t_end": 1.4,
        "MVC": 30.25, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,        
    },

    # Cat SOL, slow_dyn1 benchmarks (Perreault et al. 2003)
    "cat_SOL_10Hz_c_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "c", "freq": 10, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_10Hz_c_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "c", "freq": 10, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_20Hz_c_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "c", "freq": 20, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_20Hz_c_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "c", "freq": 20, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_30Hz_c_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "c", "freq": 30, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_30Hz_c_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "c", "freq": 30, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_10Hz_v_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "v", "freq": 10, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_10Hz_v_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "v", "freq": 10, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_20Hz_v_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "v", "freq": 20, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_20Hz_v_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "v", "freq": 20, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_30Hz_v_1mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "v", "freq": 30, "displacement_mm": "1", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },
    "cat_SOL_30Hz_v_8mm": {
        "scale": "Muscle", "muscle": "cat_SOL", "benchmark": "slow_dyn1", "stim": "v", "freq": 30, "displacement_mm": "8", "t_end": 2.0,
        "MVC": 26.13, "l_T_slack": 65, "l_M_opt": 30, "l_M_0": 30, "alpha_0": 7.5*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": True, "use_sag": False,     
    },

    # ____________________________________________________________________________
    # Muscle scale - fast muscle
    # ____________________________________________________________________________

    # Rat EDL muscle, isof (experiments ad hoc for the study)
    "rat_EDL_isof_1Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 1,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "rat_EDL_isof_30Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 30,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "rat_EDL_isof_50Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 50,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "rat_EDL_isof_60Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 60,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "rat_EDL_isof_70Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 70,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "rat_EDL_isof_80Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },    
    "rat_EDL_isof_90Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 90,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "rat_EDL_isof_100Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 100,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
        },
    "rat_EDL_isof_120Hz": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isof", "freq": 120,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 4.73, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,       
        "optimization": {
            "label": "Muscle fast isof: optimise l_M_0 on 120 Hz isometric trial",
            "parameters": ["l_M_0"],
            "x0": [3.0],
            "bounds": [[1.0, 6.6]],
            "method": "Nelder-Mead",
            "target": "force",
            "rebuild_par": True,
        },
    },

    # Rat EDL muscle, isol (experiments ad hoc for the study)
    "rat_EDL_isol_0.50mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "0.5", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,               
        "optimization": {
            "label": "Muscle fast isol: optimise MVC, Ca_max_f_M, k1_f_M, k2_f_M on 80 Hz isometric length trial",
            "parameters": ["MVC", "Ca_max_f_M", "k1_f_M", "k2_f_M"],
            "x0": [2.0, 5e5, 10.0, 10.0],
            "bounds": [[1.9, 2.9], [1e5, 1e6], [10.0, 15.0], [10.0, 15.0]],
            "method": "Nelder-Mead",
            "target": "force",
        },
    },
    "rat_EDL_isol_1.00mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "1.0", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "rat_EDL_isol_1.50mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "1.5", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "rat_EDL_isol_2.00mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "2.0", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "rat_EDL_isol_2.50mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "2.5", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "rat_EDL_isol_3.00mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "3.0", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "rat_EDL_isol_3.50mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "3.5", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },
    "rat_EDL_isol_4.00mm": {
        "scale": "Muscle", "muscle": "rat_EDL", "benchmark": "fast_isol", "length_mm": "4.0", "freq": 80,
        "MVC": 2.49, "l_T_slack": 5, "l_M_opt": 6.6, "l_M_0": 6.6, "alpha_0": 10*np.pi/180,
        "use_SE": True, "use_PE": False, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,                  
    },

    # Cat CF muscle, dyn (Brown et al. 1999)
    "cat_CF_120Hz_0.95L0_short": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "short", "freq": 120, "t_end": 0.16,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
        "optimization": {
            "label": "Muscle fast dyn: optimise af_f on -3 l0/s shortening trial at 120 Hz",
            "parameters": ["af_f"],
            "x0": [0.4],
            "bounds": [[0.1, 3.0]],
            "method": "Nelder-Mead",
            "target": "force_dynamic_ratio",
        },
    },
    "cat_CF_120Hz_0.95L0_length": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "length", "l_M_0_scale": "0.95", "freq": 120, "t_end": 0.16,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "cat_CF_20Hz_0.95L0_short": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "short", "l_M_0_scale": "0.95", "freq": 20, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "cat_CF_20Hz_0.95L0_length": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "length", "l_M_0_scale": "0.95", "freq": 20, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "cat_CF_40Hz_0.8L0_short": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "short", "l_M_0_scale": "0.8", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.80, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "cat_CF_40Hz_0.8L0_length": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "length", "l_M_0_scale": "0.8", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.80, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "cat_CF_40Hz_1.1L0_short": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "short", "l_M_0_scale": "1.1", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "short", "l_M_0_scale": "1.1", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*1.10, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,
    },
    "cat_CF_40Hz_1.1L0_length": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "length", "l_M_0_scale": "1.1", "freq": 40, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*1.10, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "cat_CF_60Hz_0.95L0_short": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "short", "l_M_0_scale": "0.95", "freq": 60, "t_end": 0.17,
        "MVC": 15.4, "l_T_slack": 24.3, "l_M_opt": 21.84, "l_M_0": 21.84*0.95, "alpha_0": 0,
        "use_SE": True, "use_PE": True, "use_FL": True, "use_FV": True, "use_yielding": False, "use_sag": False,        
    },
    "cat_CF_60Hz_0.95L0_length": {
        "scale": "Muscle", "muscle": "cat_CF", "benchmark": "fast_dyn", "trial": "length", "l_M_0_scale": "0.95", "freq": 60, "t_end": 0.17,
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
            "label": "MU S isof: optimise active state parameters on 40 Hz catMG isometric trial",
            "parameters": ["MVC","Ca_max_s_MU", "k1_s_MU", "k2_s_MU"],
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
        "t_end": 1.2, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        },
    "cat_MG_40Hz": {
        "scale": "MU", "muscle": "cat_MG", "freq": 40, "benchmark": "MU_FR",
        "MVC": 0.4, "l_T_slack": 0, "l_M_opt": 20, "l_M_0": 20, "alpha_0": 9.2*np.pi/180,
        "t_end": 1.2, "use_SE": False, "use_PE": False, "use_FL": True, "use_FV": False, "use_yielding": False, "use_sag": False,
        "optimization": {
            "label": "MU FR isof (F): optimise sag parameters on 40 Hz ratMG isometric trial",
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
            "label": "MU FR (F) isof: optimise sag parameters on 30 Hz ratMG isometric trial",
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
            "label": "MU FR (F) isof: optimise MVC, Ca_max_f_MU_ratMG, k1_f_MU_ratMG, k2_f_MU_ratMG on 150 Hz tetanic trial",
            "parameters": ["MVC", "Ca_max_f_MU_ratMG", "k1_f_MU_ratMG", "k2_f_MU_ratMG"],
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
    "slow_23_100Hz_Ca": {
        "scale": "Ca_transients", "muscle": "rat_SOL", "freq": 102, "benchmark": "slow_23_100Hz_Ca",
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
    "fast_35_125Hz_Ca": {
        "scale": "Ca_transients", "muscle": "rat_EDL", "freq": 125, "benchmark": "fast_35_125Hz_Ca",
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