
"""

"Flexing Computational Muscle: Modeling and Simulation of Musculotendon Dynamics", 2013

--> Millard's Biological Benchmarks

"""


import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp

import Virtual_Muscle
import Hatze_Model_3

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['lines.markersize'] = 4
matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rc('axes', grid=False, labelsize=12, titlesize=13, ymargin=0.01)
matplotlib.rc('legend', numpoints=1, fontsize=10)
font = {'fontname':'Times New Roman'}


# =============================================================================
#   AUXILIARY FUNCTIONS
# =============================================================================


def read_data(file, data_type, N):

    
    """
    Reads the experimental data files from Millard's benchmarks.
    
    Parameters:
    -----------
    
    file : str
        path of the .dat experimental data file
    
    data_type : str
        type of experimental data file from the benchmark, to specify the length of the header
            - 'displacement_M' for the displacement file of the Maximal-Activation benchmark
            - 'displacement_SM' for the displacement file of the Submaximal-Activation benchmark
            - 'force_M' for a force profile data file of the Maximal-Activation benchmark
            - 'force_SM' for a force profile data file of the Submaximal-Activation benchmark
            - 'force_force_SM' for a force profile data file of an isometric trial used to calibrate the model
    
    N : int
        number of points of the upsampled data file, corresponding to the number points used to evaluate the solution of the integration
    
    
    Output:
    -----------
    
    t_new : Numpy array of shape (N,)
        upsampled time vector used to evaluate the solution of the integration
    
    data_new : Numpy array of shape (N,)
        upsampled eperimental data file
    
    t : 1D Numpy array
        experimental time vector
    
    data : 1D Numpy array
        experimental data file
    
    """
    if data_type == 'force_force_npy':  # Kim 2015 digitized force profiles + time
        data_new = np.load(file)
        t_new, t, data = [], [], []
    elif data_type == 'discharge_npy': # Kim 2015 digitized discharge times
        data_new = np.round(np.array(np.load(file)), 3)
        t_new, t, data = [], [], []
    else:
        if data_type == 'displacement_M':  # Millard 2013 data
          skip = 16
        elif data_type == 'displacement_SM':
         skip = 18
        elif data_type == 'force_M':
         skip = 15
        elif data_type == 'force_SM':
         skip = 18
        elif data_type == 'force_force_SM':
         skip = 19

        t, data = np.transpose(np.loadtxt(file, delimiter='\t', skiprows=skip))[:2]
    
        f = sp.interpolate.interp1d(t, data)
    
        t_new = np.linspace(0, 2, N)
        data_new = f(t_new)
     
    return t_new, data_new, t, data



def create_muscle(muscle_model, name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=0.8, f_05=(35,35)):
    
    """
    Reads the experimental data files from Millard's benchmarks.
    
    Parameters:
    -----------
    
    muscle_model : str
        strings that defines the model used for the muscle, i.e. 'Hatze' or 'Virtual Muscle'
    
    name : str
        name of the muscle
    
    L0_M : float
        experimental optimal fibre length
        
    LT_opt : float
        optimal tendon length
    
    LT_slack : float
        experimental tendon slack length
    
    alpha0 : float
        experimental pennation angle (relatively to the axis of the tendon)
    
    F0_max : float
        experimental maximum isometric force
    
    fibre_type_profile : float in [0,1]
        proportion of fast fibres in the muscle
    
    Ur : float in [0,1] (optional, defaut is 0.8)
        threshold activation level for fast fibres in the 'Virtual Muscle' model
    
    f_05 : tuple of 2 floats (optional, default is (35,35))
        stimulation frequency required to produce 0.5 * F0_max, for slow and fast fibres, as defined for the 'Virtual Muscle' model
    
    
    Output:
    -----------
    
    muscle : muscle object
        muscle object using the designated muscle model, created according to all the required parameters as inputs
    
    
    """
    
    if muscle_model == 'Hatze':
        
        l_opt = L0_M * np.cos(alpha0 * np.pi / 180) + LT_opt
        theta = np.pi/2 - alpha0 * np.pi / 180
        l0 = L0_M * np.cos(alpha0 * np.pi / 180) + LT_slack
        if fibre_type_profile >= 0.5:
            fibre_type = 'fast'
        else:
            fibre_type = 'slow'
        
        muscle_Hatze = Hatze_Model_3.Hatze_Muscle(name, l_opt, fibre_type, theta=theta, lbda_T0=LT_slack, l0=l0)
        
        return muscle_Hatze

    elif muscle_model == 'Virtual Muscle':

        F_pcsa_fast = fibre_type_profile
        Lce0 = L0_M
        Lse0 = LT_opt
        Lmt_max = 1.1 * (Lce0 * np.cos(alpha0 * np.pi / 180) + Lse0)
        mass = F0_max * 1060 * Lce0 / 31.8e4
        
        muscle_VM = Virtual_Muscle.Muscle(name, F_pcsa_fast, Lmt_max, Lce0, Lse0, mass, f_05=f_05, Ur=Ur)
        
        return muscle_VM
    
    else:
        
        print('Muscle model not implemented')
        
        return None



# =============================================================================
#   MAXIMAL-ACTIVATION BENCHMARK
# =============================================================================


def maximal_activation_biological_benchmark(muscle_model):
    
    """
    Runs the Maximal-Activation benchmark for the designated muscle model.
    Previously optimized values are used for some of the muscle parameters.
    First the experimental parameters are defined, and the muscle object is created accordingly.
    Then the 6 trials of the benchmark are simulated and the mean absolute and max absolute errors are computed.
    The 6 normalized force profiles are plotted as results of the benchmark, for each displacement amplitude.
    
    Parameters:
    -----------
    
    muscle_model : str
        strings that defines the model used for the muscle, i.e. 'Hatze' or 'Virtual Muscle'
    
    
    Output:
    -----------
    
    benchmark_output : Numpy array of shape (N_trials, N)
        force profiles (in Newtons) produced by the model for each trial of the benchmark
    
    experimental_data : Numpy array of shape (N_trials, N)
        upsampled experimental force profiles (in Newtons) for each trial of the benchmark
    
    errors : Numpy array of shape (N,)
        vector of mean absolute errors for each trial of the benchmark
    
    muscle : muscle object
        muscle object computed to run the benchmark
    
    
    """
    
    
    start = time.time()
    print("\nMaximal activation - Biological Benchmark: " + muscle_model + '\n')
    
    # Number of integration steps
    N = 20000

    # Read data
    path = '..\\biologicalBenchmark\\slowMuscle_maximalActivation\\'
    t, displacement = read_data(path + 'displacement.dat', 'displacement_M', N)[:2]
    
    # Muscle parameters
    name = 'Rat soleus'
    alpha0 = 6
    L0_M = 0.0171
    LT_slack = L0_M
    LT_opt = LT_slack * 1.08
    F0_max_exp = 1.17
    fibre_type_profile = 0.1       # Novak (Physiol. Res., 2010) or Fukutani (Scientific reports, 2023)
    f_05 = (35,35)
    
    # Benchmark parameters for amplitudes of displacement
    max_displacement_amplitudes = [0.05, 0.1, 0.25, 0.5, 1, 2]
    displacement_offset = -0.002
    
    N_trials = len(max_displacement_amplitudes)
    benchmark_output = np.zeros((N_trials, N))
    experimental_data = np.zeros((N_trials, N))
    errors_ds = np.zeros(N_trials)
    p = np.zeros(N_trials)
    
    # Run trial for each displacement amplitude value
    for k in range(N_trials):
        
        print('Trial ' + str(k+1))
        
        # Deduce inputs depending on the muscle model and compute the model output 
        if muscle_model == 'Hatze':
            
            # Create muscle object
            muscle = create_muscle(muscle_model, name, L0_M, LT_opt, LT_slack, alpha0, F0_max_exp, fibre_type_profile)

            # Parameters values optimized by fitting (least-squares) the 1 mm displacement trial
            F0_max = 1.2338784018443751
            muscle.set_contraction_params(4.102275785196238)
            
            # Define the length profile of the musculotendon system throughout the simulation
            def get_l(t):
                i = max(round(N*t/2) - 1, 0)
                Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + max_displacement_amplitudes[k] * 0.001 * displacement[i]
                return Lmt
            
            l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset
            
            # Define the normalized average stimulation rate profile
            def get_v(t):
                return 70 / muscle.max_stim_rate      # 70 Hz stimulation
            
            # Define the normalized recruitment rate profile
            def get_z(t):
                return 0
            
            # Define the initial state
            n_0 = 1
            if n_0 == 1:
                psi_0 = muscle.c * get_v(0)
            else:
                psi_0 = 0
            
            # Simulate the trial. Rockenfeller's approach and stretch potentiation are used by default
            F_ = muscle.simulate('maximalActivationBenchmark_max' + str(k+1) + '_Hatze.csv', t, get_l, get_v, get_z, l_init, write_output=1, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)['F_TOT(t)']
            F = F_ * F0_max
        
        
        elif muscle_model == 'Virtual Muscle':
            
            # Parameters values optimized by fitting (least-squares) the 1 mm trial of the benchmark
            Ur = 0.8863448995764386
            F0_max = 1.089233970512266
            Vmax_slow = -9.714092107715397
            Vmax_fast = -10.075826599196894
            
            muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur, f_05=f_05)
            muscle.fibres_list['slow'].Vmax = Vmax_slow
            muscle.fibres_list['fast'].Vmax = Vmax_fast
            
            # Define activation
            def get_activation(t):
                return 1
            
            # Define the length profile of the musculotendon system throughout the simulation
            def get_muscle_length(t):
                i = max(round(N*t/2) - 1, 0)
                Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + max_displacement_amplitudes[k] * 0.001 * displacement[i]
                return Lmt
            
            Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset
            
            # The muscle is initially fully activated
            U_eff_0 = 1
            f_eff_0 = 2
            f_int_0 = 2
            
            muscle.eta = 0.016
            
            # Simulate the trial
            F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max
        
        else:
            print("Muscle model not implemented")
            return None, None, None, None
        
        benchmark_output[k] = F

        # Read experimental data
        t_exp, exp_data = read_data(path + 'force_trial' + str(k+1) + '.dat', 'force_M', N)[:2]
        experimental_data[k] = exp_data
        
        # Compute errors only on experimental values
        for j in range(len(t_exp)):
            for i in range(len(t)):
                if abs(t_exp[j]-t[i]) < 0.0001:
                    p[k] = p[k] + 1
                    errors_ds[k] = errors_ds[k] + abs(benchmark_output[k][i] - exp_data[j]) / F0_max_exp
                    break
        
    # Compute error between model output and experimental data according to Millard (2013) and Krylow & Sandercock (1997)
    errors = np.mean(np.absolute(benchmark_output - experimental_data), 1) / F0_max_exp
    errors_max = np.amax(np.absolute(benchmark_output - experimental_data), 1) / F0_max_exp
    errors_ds /= p
    
    # Plot benchmark results
    fig, axs = plt.subplots(N_trials, 1, figsize=(8, 12), sharex=True)
    plt.title(muscle_model + ' - Maximal activation Biological Benchmark\n')
    
    for k in range(N_trials):
        axs[k].set_title('\u00B1' + str(max_displacement_amplitudes[k]) + ' mm - Error = ' + str(round(100 * errors_ds[k], 3)) + '% - Max error = ' + str(round(100 * errors_max[k], 3)) + '%')
        axs[k].plot(t, benchmark_output[k] / F0_max, label=muscle_model)
        axs[k].plot(t, experimental_data[k] / F0_max_exp, linewidth=0.75, label='Krylow & Sandercock (1997)')
        axs[k].set_ylim((0, 1.5))
    axs[5].set_xlabel('Time (s)')
    axs[2].set_ylabel('Normalized force')
    plt.suptitle("Maximal activation - Biological Benchmark - " + muscle_model)
    axs[0].legend(loc='lower right')
    plt.tight_layout()
    
    duration = time.time() - start
    print("\nBenchmark computation time: " + str(round(duration) // 60) + ' min ' + str(round(duration) % 60) + ' s')
    
    return benchmark_output, experimental_data, errors, muscle



# =============================================================================
#   SUBMAXIMAL-ACTIVATION BENCHMARK
# =============================================================================


def submaximal_activation_biological_benchmark(muscle_model):
    
    """
    Runs the first half of the Subaximal-Activation benchmark for the designated muscle model.
    Previously optimized values are used for the recruitment/activation parameters.
    First the experimental parameters are defined, and the muscle object is created accordingly.
    Then the 6 trials with constant stimulation frequency (3 with 1 mm displacement and 3 with 8 mm) of the benchmark are simulated and the mean absolute and max absolute errors are computed.
    The 6 force profiles are plotted as results of the benchmark, for each displacement amplitude and firing rate.
    
    Parameters:
    -----------
    
    muscle_model : str
        strings that defines the model used for the muscle, i.e. 'Hatze' or 'Virtual Muscle'
    
    
    Output:
    -----------
    
    benchmark_output : Numpy array of shape (N_trials, N)
        force profiles (in Newtons) produced by the model for each trial of the benchmark
    
    experimental_data : Numpy array of shape (N_trials, N)
        upsampled experimental force profiles (in Newtons) for each trial of the benchmark
    
    errors : Numpy array of shape (N,)
        vector of mean absolute errors for each trial of the benchmark
    
    muscle : muscle object
        muscle object computed to run the benchmark
    
    
    """
    
    start = time.time()
    print("\nSubmaximal activation - Biological Benchmark: " + muscle_model)
    
    # Number of integration steps
    N = 10001

    # Read displacement data
    path = '..\\biologicalBenchmark\\slowMuscle_submaximalActivation\\'
    exp_path = path + 'c_freq\\'
    t, displacement_1mm = read_data(path + 'displacement_1.dat', 'displacement_SM', N)[:2]
    t, displacement_8mm = read_data(path + 'displacement_8.dat', 'displacement_SM', N)[:2]
    disp = [displacement_1mm, displacement_8mm]
    displacement_offset = 0.004
    
    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05
    f_05 = (50, 50)

    # Create muscle object
    muscle = create_muscle(muscle_model, name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, f_05=f_05)
    
    # Stimulation rates for constant stimulation trials
    freqs = [10, 20, 30]
    
    N_trials = 6
    benchmark_output = np.zeros((N_trials, N))
    experimental_data = np.zeros((N_trials, N))
    
    k = 0
    muscle.max_stim_rate = 28
    
    for j in range(len(disp)):
        
        if j == 0:
            disp_amp = '\u00B1 1 mm'
        else:
            disp_amp = '\u00B1 8 mm'
        
        for v in range(len(freqs)):
            
            print('\nTrial ' + str(k+1) + ' : Displacement = ' + disp_amp + ' - Firing rate = ' + str(freqs[v]) + ' Hz\n')               
            
            if muscle_model == 'Hatze':
                
                # Recruitment parameters optimized with isometric trials
                dt = [0.05616549235153895, 0.08792997639338, 0.08723816853973808]
                z = [0.9493746039373664, 0.7969650284435564, 0.80320403808114]
                
                # Define the length profile throughout the experiment
                l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
                
                def get_l(t):
                    i = max(round(N*t/2) - 1, 0)
                    Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + 0.001 * disp[j][i]
                    return Lmt
                
                # Define the stimulation rate profile
                def get_v(t):
                    if t >= 0 and t <= 1.5:
                        return freqs[v] / muscle.max_stim_rate
                    else:
                        return 0
                
                # Define initial state
                n_0 = 0
                if n_0 == 1:
                    psi_0 = muscle.c * get_v(0)
                else:
                    psi_0 = 0
                
                # Define the recruitment rate profile
                def get_z(t):
                    if t >= 0 and t < min(dt[v], 1.5):
                        return z[v]
                    elif t >= 1.5 and t <= min(1.5 + dt[v], 2):
                        return -1
                    else:
                        return 0
                
                # Simulate the trial. Rockenfeller's approach is used by default
                res = muscle.simulate('submaximalActivationBenchmark_constant' + str(k+1) + '_Hatze.csv', t, get_l, get_v, get_z, l_init, write_output=1, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
                F = F0_max * res['F_TOT(t)']
                
            
            elif muscle_model == 'Virtual Muscle':
                
                # Activation parameters optimized with isometric trials
                Ur_opti = [0.0178089138153906, 0.06179286085032508, 0.054234682899651165]
                Act_opti = [0.016925840599448114, 0.043546874706306554, 0.05156517880435046]
                
                # Create muscle object
                muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur_opti[v], f_05=f_05)
                
                # Define the length profile throughout the experiment
                Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
                
                def get_muscle_length(t):
                    i = max(round(N*t/2) - 1, 0)
                    Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + 0.001 * disp[j][i]
                    return Lmt
                
                # Define the activation profile
                def get_activation(t):
                    if t <= 1.5:
                        return Act_opti[v]
                    else:
                        return 0
                
                # The muscle is initially fully activated
                U_eff_0 = 0
                f_eff_0 = 0
                f_int_0 = 0
                
                # Run the trial
                F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max 
            
            benchmark_output[k] = F

            # Read experimental data
            freq = freqs[v]
            disp_label = '1' if j == 0 else '8'
            exp_data_file = f'force_c{freq}_{disp_label}.dat'
            exp_data = read_data(exp_path + exp_data_file, 'force_SM', N)[1]
            experimental_data[k] = exp_data
            
            k += 1
    
    # Compute error between model output and experimental data according to Perreault (2003)
    benchmark_output_ds = benchmark_output[:, ::5]
    experimental_data_ds = experimental_data[:, ::5]
    errors = np.mean(np.absolute(benchmark_output_ds - experimental_data_ds), 1) / F0_max
    errors_max = np.amax(np.absolute(benchmark_output_ds - experimental_data_ds), 1) / F0_max
    
    # Plot benchmark results
    fig, axs = plt.subplots(len(freqs), len(disp), figsize=(8, 12))
    plt.title(muscle_model + ' - Submaximal activation Biological Benchmark\n')
    
    k = 0
    for j in range(len(disp)):
        if j == 0:
            disp_amp = '\u00B1 1 mm'
        else:
            disp_amp = '\u00B1 8 mm'
        for v in range(len(freqs)):
            axs[v,j].set_title(disp_amp + ' - ' + str(freqs[v]) + ' Hz - Error = ' + str(round(100 * errors[k], 3)) + '% - Max error = ' + str(round(100 * errors_max[k], 3)) + '%')
            axs[v,j].plot(t, benchmark_output[k], label=muscle_model)
            axs[v,j].plot(t, experimental_data[k], linewidth=0.75, label='Perreault (2003)')
            axs[v,j].set_ylim((0, 40))
            k += 1
    axs[2,0].set_xlabel('Time (s)')
    axs[2,1].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Force (N)')
    axs[1,0].set_ylabel('Force (N)')
    axs[2,0].set_ylabel('Force (N)')
    plt.suptitle("Submaximal activation - Biological Benchmark - " + muscle_model)
    axs[0, 1].legend(loc='upper right')
    plt.tight_layout()
    
    duration = time.time() - start
    print("\nBenchmark computation time: " + str(round(duration) // 60) + ' min ' + str(round(duration) % 60) + ' s')
    
    return benchmark_output, experimental_data, errors, muscle



def lengths_biological_benchmark(muscle_model):
    
    """
    Runs the Subaximal-Activation benchmark at different lengths for the designated muscle model.
    The muscle is the same of the Submax. benchmark. Data is taken from Kim 2015 et al.
    Previously optimized values are used for the recruitment/activation parameters.
    First the experimental parameters are defined, and the muscle object is created accordingly.
    
    Parameters:
    -----------
    
    muscle_model : str
        strings that defines the model used for the muscle, i.e. 'Hatze' or 'Virtual Muscle'
    
    
    Output:
    -----------
    
    benchmark_output : Numpy array of shape (N_trials, N)
        force profiles (in Newtons) produced by the model for each trial of the benchmark
    
    experimental_data : Numpy array of shape (N_trials, N)
        upsampled experimental force profiles (in Newtons) for each trial of the benchmark
    
    errors : Numpy array of shape (N,)
        vector of mean absolute errors for each trial of the benchmark
    
    muscle : muscle object
        muscle object computed to run the benchmark
    
    
    """
    start = time.time()

    N = 14000
    t = np.linspace(0, 1.4, N)
    
    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05
    displacement_offset = 0.004
    muscle.max_stim_rate = 28

    # Create muscle object
    muscle = create_muscle(muscle_model, name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile)
    
    # Stimulation rates for constant stimulation trials
    exp_path = '..\\biologicalBenchmark\\slowMuscle_lengthEffect\\'
    freqs = [1, 10, 20, 40] # stim. freqs
    l_offset = [0, 0.008, 0.016] # applied l variations according to Kim 2015 et al.
    t_end = [0.2, 0.85, 0.85, 0.85] # end of stimulation

    N_trials = 12
    benchmark_output = np.zeros((N_trials, N))
    experimental_data = np.zeros((N_trials, N))

    k = 0

    for j in range(len(l_offset)):
        
        for v in range(len(freqs)):
            
            print('\nTrial ' + str(k+1) + ' : Length offset = ' + str(l_offset[j]) + ' mm - Firing rate = ' + str(freqs[v]) + ' Hz\n')               
            
            if muscle_model == 'Hatze':
                
                # Recruitment parameters optimized with isometric trials
                dt = [] # to be updated
                z = [] # to be updated
                
                dis_file = f'8_{freqs[k-1]}_times.npy'
                t_start = read_data(exp_path + dis_file, 'discharge_npy', N)[1]
                t_start = t_start[0] # first discharge time to offset simulation time

                # Define the length profile throughout the experiment
                l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - l_offset[j]
                
                def get_l_isometric(t):
                    Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - l_offset[j]
                    return Lmt
                
                # Define the stimulation rate profile
                def get_v(t):
                    if t >= t_start and t <= t_end[v]:
                        return freqs[v] / muscle.max_stim_rate
                    else:
                        return 0
                
                # Define initial state
                n_0 = 0
                if n_0 == 1:
                    psi_0 = muscle.c * get_v(0)
                else:
                    psi_0 = 0
                
                # Define the recruitment rate profile
                def get_z(t):
                    if t >= t_start and t < dt[v]:
                        return z[v]
                    elif t >= t_end[v] and t <= t_end[v] + dt[v]:
                        return -1
                    else:
                        return 0
                
                # Simulate the trial. Rockenfeller's approach is used by default
                res = muscle.simulate('lengths_Benchmark_' + str(l_offset[j]) + '_' + str(freqs[v]) + '.csv', t, get_l_isometric, get_v, get_z, l_init, write_output=0, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
                F = F0_max * res['F_TOT(t)']

            benchmark_output[k] = F

            # Read experimental data
            exp_data_file = f'{l_offset[j]}_{freqs[v]}_interp.npy'
            exp_data = read_data(exp_path + exp_data_file, 'force_force_npy', N)[1]
            experimental_data[k] = exp_data

            k += 1
    
    # Compute error between model output and experimental data according to Perreault (2003)
    errors = np.mean(np.absolute(benchmark_output - experimental_data), 1) / F0_max
    errors_max = np.amax(np.absolute(benchmark_output - experimental_data), 1) / F0_max
    
    # Plot benchmark results
    fig, axs = plt.subplots(len(freqs), len(l_offset), figsize=(8, 12))
    plt.title(muscle_model + ' - Submaximal activation Biological Benchmark - Length Effect\n')
    
    k = 0
    for j in range(len(l_offset)):
        for v in range(len(freqs)):
            axs[v,j].set_title(str(l_offset[j]) + ' mm - ' + str(freqs[v]) + ' Hz - Error = ' + str(round(100 * errors[k], 3)) + '% - Max error = ' + str(round(100 * errors_max[k], 3)) + '%')
            axs[v,j].plot(t, benchmark_output[k], label=muscle_model)
            axs[v,j].plot(t, experimental_data[k], linewidth=0.75, label='Kim (2015)')
            axs[v,j].set_ylim((0, 30))
            k += 1
    axs[3,0].set_xlabel('Time (s)')
    axs[3,1].set_xlabel('Time (s)')
    axs[3,2].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Force (N)')
    axs[1,0].set_ylabel('Force (N)')
    axs[2,0].set_ylabel('Force (N)')
    axs[3,0].set_ylabel('Force (N)')
    plt.suptitle("Submaximal activation - Biological Benchmark - " + muscle_model)
    axs[0, 2].legend(loc='upper right')
    plt.tight_layout()
    
    duration = time.time() - start
    print("\nBenchmark computation time: " + str(round(duration) // 60) + ' min ' + str(round(duration) % 60) + ' s')
    
    return benchmark_output, experimental_data, errors, muscle



def submaximal_activation_biological_benchmark_random(muscle_model, opt_mode='constant'):
    
    """
    Runs the second half of the Submaximal-Activation benchmark for the designated muscle model.
    Previously optimized values are used for the recruitment/activation parameters. The type of optimized profile can be specified for Hatze's model.
    First the experimental parameters are defined, and the muscle object is created accordingly.
    Then the 6 trials with random stimulation frequency (3 with 1 mm displacement and 3 with 8 mm) of the benchmark are simulated and the mean absolute and max absolute errors are computed.
    The 6 force profiles are plotted as results of the benchmark, for each displacement amplitude and firing rate.
    
    Parameters:
    -----------
    
    muscle_model : str
        strings that defines the model used for the muscle, i.e. 'Hatze' or 'Virtual Muscle'
    
    opt_mode : 'constant' or 'affine' (optional, default is 'constant')
        type of optimized recruitment rate and stimulation rate profiles, either piecewise constant or piecewise affine (for Hatze's model)
    
    
    Output:
    -----------
    
    benchmark_output : Numpy array of shape (N_trials, N)
        force profiles (in Newtons) produced by the model for each trial of the benchmark
    
    experimental_data : Numpy array of shape (N_trials, N)
        upsampled experimental force profiles (in Newtons) for each trial of the benchmark
    
    errors : Numpy array of shape (N,)
        vector of mean absolute errors for each trial of the benchmark
    
    muscle : muscle object
        muscle object computed to run the benchmark
    
    
    """
    
    start = time.time()
    print("\nSubmaximal activation (Random stimulation) - Biological Benchmark: " + muscle_model)
    
    # Number of integration steps
    N = 10001    

    # Read displacement data
    path = '..\\biologicalBenchmark\\slowMuscle_submaximalActivation\\'
    exp_path = path + 'v_freq\\'
    t, displacement_1mm = read_data(path + 'displacement_1.dat', 'displacement_SM', N)[:2]
    t, displacement_8mm = read_data(path + 'displacement_8.dat', 'displacement_SM', N)[:2]
    disp = [displacement_1mm, displacement_8mm]
    displacement_offset = 0.004

    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05
    f_05 = (50, 50)
    
    # Create muscle object
    muscle = create_muscle(muscle_model, name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, f_05=f_05)
    
    # Stimulation rates for constant stimulation trials
    freqs = [10, 20, 30]    
    
    N_trials = 6
    benchmark_output = np.zeros((N_trials, N))
    experimental_data = np.zeros((N_trials, N))
    
    k = 0
    muscle.max_stim_rate = 28

    for j in range(len(disp)):
        
        if j == 0:
            disp_amp = '\u00B1 1 mm'
        else:
            disp_amp = '\u00B1 8 mm'
        
        for v in range(len(freqs)):
            
            print('\nTrial ' + str(k+7) + ' : Displacement = ' + disp_amp + ' - Firing rate = ' + str(freqs[v]) + ' Hz\n')               
            
            if muscle_model == 'Hatze':
                
                if opt_mode == 'constant':
                    
                    # Piecewise constant normalized recruitment rate optimized with isometric trials
                    z_opti = np.array([[ 0.41246592,  0.53552881,  0.53617644,  0.22492637,  0.35434548,
                                         0.84970298,  0.54136677,  0.94816105,  0.83652618,  0.23890421,
                                         0.81195758,  0.73103521,  0.52088711,  0.78602182,   0.9750695,
                                         -0.8809053, -0.84182333,  -0.9992443, -0.75648932, -0.39532292],
                                       [ 0.70478248,  0.39943196,  0.87030146,  0.99090034,  0.63982061,
                                         0.64308067,  0.70356407,  0.23978038,  0.21302069,  0.78915262,
                                         0.97896509,  0.92078193,  0.99993007,  0.68269722,   0.8461046,
                                        -0.41685006, -0.77384687, -0.93126838, -0.75553171, -0.87995325],
                                       [          1,   0.8738393,  0.80689017,  0.66238506,   0.8478222,
                                         0.88012829,  0.96334272,   0.0755728,  0.70872772,  0.99989468,
                                         0.96097031,  0.74941019,  0.41142655,  0.68365556,   0.8420215,
                                        -0.71490995, -0.78188484, -0.38676073, -0.69138371, -0.71640857]])
                    
                    # Piecewise constant stimulation rate optimized with isometric trials (already denormalized)
                    v_opti = np.array([[10.86333328, 20.94131785, 10.76952106,  6.88769715,  2.77360154,
                                         1.53428885,  2.04018496,  2.63225065,  1.84976808,  2.43747312,
                                          2.5192973,  2.69007884,  24.9848858,  7.17091324,  4.79479756],
                                       [13.42413493,  9.33418601,  8.97459547,  8.83957019,  7.75499405,
                                         6.93425765, 11.93150426, 35.18135027, 24.24955042, 13.60730413,
                                         9.85625072,  9.60481938, 11.28605213, 11.93094446, 22.47229531],
                                       [ 1.80930325, 19.99749297,  46.2338573,   26.869582, 17.96913211, 
                                        42.83530393, 27.54374668, 37.46683102,  41.2761062, 37.56078939,
                                        50.40510269, 37.31621452, 40.84126595, 36.41615431, 23.48021985]])
                    
                    # Define the stimulation rate profile
                    def get_v(t):
                        if t >= 0 and t < 1.5:
                            i = int(10 * t)
                            return v_opti[v][i] / muscle.max_stim_rate
                        else:
                            return 0
                    
                    # Define the recruitment rate profile
                    def get_z(t):
                        if t >= 0 and t < 2:
                            i = int(10 * t)
                            return z_opti[v][i]
                        else:
                            return 0  
                    
                elif opt_mode == 'affine':
                    
                    # Coefficients of the piecewise affine control parameters optimized with isometric trials
                    v_opti = [[ 8.61899464, 10.92711615,   7.5257141, 7.69860522,   7.82664887,   7.8492032,
                                   7.39309,  8.42020555,  7.46824168, 8.57392181,   7.44071785,  7.56849651,
                                8.29495676,   7.3581274,  8.36881309],
                              [11.95862217, 13.87192904, 15.46614572, 14.19763154, 14.27869365, 11.37473726,
                               14.55435802, 17.55060357, 21.46292075, 17.81823772, 18.01440078, 15.98785155,
                               15.11912966, 18.11976234, 10.80352789],
                              [20.64369146,  21.2024033, 30.05296469,  27.9789737, 26.26911758,  28.0438216,
                               26.43377508, 27.18331855, 26.95258803, 27.53023374, 26.74112175, 25.79439005,
                               26.14982233,  25.5421824, 24.10037554]]
                    
                    z_opti = [[ 0.80804563,  0.09407439,  0.46411189,  0.81157285,  0.91820565,  0.99973653,
                                0.72624774,  0.74878354,   0.8564491,  0.97441882,  0.73219854,  0.98400928,
                                0.40624579,  0.73301529,   0.6446128, -0.82100789, -0.95518615, -0.87287727,
                               -0.88777842,  -0.6282604],
                              [ 0.86958354,  0.47220355,  0.59738732,  0.92009258,  0.85763474,  0.37197799,
                                0.78407256,  0.91150931,  0.82244275,  0.69241288,  0.72340102,  0.37752192,
                                 0.6870374,  0.86442666,  0.79343097, -0.72803228, -0.87240892, -0.50451121,
                               -0.54960024, -0.9997303 ],
                              [   0.921803,  0.86468309, 0.76981905,  0.68924031,  0.85053995,  0.82267651,
                                0.79565826,  0.83033548, 0.74572085,  0.999,       0.72535527,  0.97232148,
                                0.80687365,  0.69143503, 0.86748496, -0.78283707, -0.66613212, -0.70816975,
                               -0.96099449, -0.8457315 ]]
                    
                    v0_opti = [[ 0.01172887,  0.03128295,  0.11010645,  0.00773227, -0.10481958, -0.14079302,
                                -0.08657388,  -0.0577682, -0.08882458, -0.06337969, -0.03070662, -0.12578352,
                                  0.2344424, -0.05560905,  -0.0098505],
                               [-0.23194994, -0.34842399, -0.25698043, -0.15237392, -0.14301809, -0.07793733,
                                 -0.0561062,  0.29563796,  -0.0253728, -0.01835564, -0.08882697, -0.05214594,
                                -0.03147111, -0.04205199,  0.05477635],
                               [-0.06482842, -0.09974689,  0.13673105,   0.0553227,   0.0079321,  0.10155026,
                                 0.11203208,  0.09799439,  0.15513978,  0.16175365,  0.14096922,  0.14750023,
                                 0.11517711,  0.08982815, -0.01524716]]
                    
                    z0_opti = [[ 0.02845502,  0.17601186, -0.06697911, -0.02468923, -0.26940083,  0.13768156,
                                -0.15842798,  0.43444788,  0.16380727,   0.1028859,  0.29778352, -0.27619208,
                                 0.18352532, -0.07016722, -0.33169857, -0.23022444, -0.23374981,  0.24213722,
                                -0.13984584, -0.25807677],
                               [  0.0327294,  0.15092359, -0.04697508, -0.03825318, -0.09880783, -0.38528225,
                                -0.21692492, -0.12589687, -0.06146304,  0.05979499, -0.00111525,  0.06882861,
                                 0.16469285, -0.14332284,  0.13342498,  0.45672982, -0.11135628,  0.00929016,
                                -0.01997315,  -0.0885796],
                               [ 0.12017547,  0.06218318,  -0.2005668, -0.02873642,  0.02000663, -0.06080781,
                                -0.02954694,  0.02526454,  -0.1474261, -0.06403555,  0.04329089, -0.06961803,
                                -0.03098754, -0.08660776,  0.09470218,  0.01170097, -0.44253063,  0.06462573,
                                -0.22889362,  0.21881069]]
                    
                    # Define the stimulation rate profile
                    def get_v(t):
                        if t >= 0 and t < 1.5:
                            i = int(10 * t)
                            return v0_opti[v][i] * t + v_opti[v][i] / muscle.max_stim_rate
                        else:
                            return 0
                    
                    # Define the recruitment rate profile
                    def get_z(t):
                        if t >= 0 and t < 2:
                            i = int(10 * t)
                            return z_opti[v][i] + z0_opti[v][i] * t
                        else:
                            return 0  
    
                # Define the muscle length profile
                l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
                
                def get_l(t):
                    i = max(round(N*t/2) - 1, 0)
                    Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + 0.001 * disp[j][i]
                    return Lmt
                
                # Define the initial state
                n_0 = 0
                if n_0 == 1:
                    psi_0 = muscle.c * get_v(0)
                else:
                    psi_0 = 0
                
                # Run the trial. Rockenfeller's approach is used by default
                res = muscle.simulate('submaximalActivationBenchmark_random' + str(k+1) + '_Hatze.csv', t, get_l, get_v, get_z, l_init, write_output=1, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
                F = F0_max * res['F_TOT(t)']
            
            
            elif muscle_model == 'Virtual Muscle':
                
                # Piecewise activation optimized with isometric trials
                Act = [np.array([0.16314664, 0.00912902, 0.06218633, 0.01005659, 0.01      ,
                                 0.00741152, 0.02356047, 0.0120471 , 0.00631099, 0.03061831,
                                 0.01185442, 0.00879885, 0.12044628, 0.01522538, 0.01497605]),
                       np.array([0.16670141, 0.01      , 0.00345094, 0.05239024, 0.01887405,
                                 0.01960318, 0.04708376, 0.11760841, 0.05635439, 0.03484559,
                                 0.02218307, 0.02340521, 0.02616652, 0.04169834, 0.05754865]),
                       np.array([0.27718039, 0.01000001, 0.01849928, 0.06802672, 0.03883657,
                                 0.07342034, 0.05328407, 0.04779378, 0.05593187, 0.06575734,
                                 0.05005319, 0.04970774, 0.05517411, 0.04532544, 0.08887427])]
                Ur = [0.7273032686481556, 0.8191285350416326, 0.7076087215361527]
                
                # Create muscle object
                muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur[v], f_05=f_05)
                        
                # Deduct inputs depending on the muscle model and compute the model output 
                def get_muscle_length(t):
                    i = max(round(N*t/2) - 1, 0)
                    Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + 0.001 * disp[j][i]
                    return Lmt
                
                Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
                
                # Define activation
                def get_activation(t):
                    if t <= 1.5:
                        i = int(10*t)
                        return Act[v][i]
                    else:
                        return 0
                
                # The muscle is initially fully activated
                U_eff_0 = 0
                f_eff_0 = 0
                f_int_0 = 0
                
                # Simulate the trial
                F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max

            
            benchmark_output[k] = F

            # Read experimental data
            freq = freqs[v]
            disp_label = '1' if j == 0 else '8'
            exp_data_file = f'force_v{freq}_{disp_label}.dat'
            exp_data = read_data(exp_path + exp_data_file, 'force_SM', N)[1]
            experimental_data[k] = exp_data
            
            k += 1
    
    # Compute error between model output and experimental data according to Perreault (2003), comparing only the actual experimental values
    benchmark_output_ds = benchmark_output[:, ::5]
    experimental_data_ds = experimental_data[:, ::5]
    errors = np.mean(np.absolute(benchmark_output_ds - experimental_data_ds), 1) / F0_max
    errors_max = np.amax(np.absolute(benchmark_output_ds - experimental_data_ds), 1) / F0_max
    
    # Plot benchmark results
    fig, axs = plt.subplots(len(freqs), len(disp), figsize=(8, 12))
    plt.title(muscle_model + ' - Submaximal activation Biological Benchmark - Random stimulation\n')
    
    k = 0
    for j in range(len(disp)):
        if j == 0:
            disp_amp = '\u00B1 1 mm'
        else:
            disp_amp = '\u00B1 8 mm'
        for v in range(len(freqs)):
            axs[v,j].set_title(disp_amp + ' - ' + str(freqs[v]) + ' Hz - Error = ' + str(round(100 * errors[k], 3)) + '% - Max error = ' + str(round(100 * errors_max[k], 3)) + '%')
            axs[v,j].plot(t, benchmark_output[k], label=muscle_model)
            axs[v,j].plot(t, experimental_data[k], linewidth=0.75, label='Perreault (2003)')
            axs[v,j].set_ylim((0, 40))
            k += 1
    axs[2,0].set_xlabel('Time (s)')
    axs[2,1].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Force (N)')
    axs[1,0].set_ylabel('Force (N)')
    axs[2,0].set_ylabel('Force (N)')
    plt.suptitle("Submaximal activation (Random stimulation) - Biological Benchmark - " + muscle_model)
    axs[0, 1].legend(loc='upper right')
    plt.tight_layout()
    
    duration = time.time() - start
    print("\nBenchmark computation time: " + str(round(duration) // 60) + ' min ' + str(round(duration) % 60) + ' s')
    
    return benchmark_output, experimental_data, errors, muscle


def fast_biological_benchmark_iso(muscle_model):
    
    """
    Runs the benchmarks of fast muscle (cat caudfemoralis, from Brown 1999) at different lengths in isometric conditions (Figure 3B).
    Previously optimized values are used for the recruitment/activation parameters.
    First the experimental parameters are defined, and the muscle object is created accordingly.
    
    Parameters:
    -----------
    
    muscle_model : str
        strings that defines the model used for the muscle, i.e. 'Hatze' or 'Virtual Muscle'
    
    
    Output:
    -----------
    
    benchmark_output : Numpy array of shape (N_trials, N)
        force profiles (in Newtons) produced by the model for each trial of the benchmark
    
    experimental_data : Numpy array of shape (N_trials, N)
        upsampled experimental force profiles (in Newtons) for each trial of the benchmark
    
    errors : Numpy array of shape (N,)
        vector of mean absolute errors for each trial of the benchmark
    
    muscle : muscle object
        muscle object computed to run the benchmark
    
    
    """
    start = time.time()

    N = 3600
    t = np.linspace(0, 0.36, N)
    
    # Muscle parameters
    name = 'Cat caudofemoralis'
    alpha0 = 0
    L0_M = 0.056
    LT_slack = 0.017
    LT_opt = LT_slack * 1.08
    F0_max = 15.4
    fibre_type_profile = 0.95
    muscle.max_stim_rate = 100

    # Create muscle object
    muscle = create_muscle(muscle_model, name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile)
    
    # Stimulation rates for constant stimulation trials
    exp_path = '..\\biologicalBenchmark\\fastMuscle\\iso\\'
    freqs = [30, 30, 30, 30, 30] # stim. freqs
    t_end = [0.25, 0.25, 0.25, 0.25, 0.25] # end of stimulation
    scale = [0.8, 0.9, 1.0, 1.1, 1.2]

    N_trials = 5
    benchmark_output = np.zeros((N_trials, N))
    experimental_data = np.zeros((N_trials, N))

    k = 0
        
    for v in range(len(freqs)):
            
        print('\nTrial ' + str(k+1) + ' : Length = ' + str(scale[v]) + ' L0 - Firing rate = ' + str(freqs[v]) + ' Hz\n')               
            
        if muscle_model == 'Hatze':
                
            # Recruitment parameters optimized with isometric trials
            dt = [] # to be updated
            z = [] # to be updated

            # Define the length profile throughout the experiment
            l_init = ((L0_M * scale[v])* np.cos(alpha0 * np.pi / 180)) + LT_opt
                
            def get_l_isometric(t):
                Lmt = ((L0_M * scale[v])* np.cos(alpha0 * np.pi / 180)) + LT_opt
                return Lmt
                
            # Define the stimulation rate profile
            def get_v(t):
                if t >= 0 and t <= t_end[v]:
                    return freqs[v] / muscle.max_stim_rate
                else:
                    return 0
                
            # Define initial state
            n_0 = 0
            if n_0 == 1:
                psi_0 = muscle.c * get_v(0)
            else:
                psi_0 = 0
                
            # Define the recruitment rate profile
            def get_z(t):
                if t >= 0 and t < dt[v]:
                    return z[v]
                elif t >= t_end[v] and t <= t_end[v] + dt[v]:
                    return -1
                else:
                    return 0
                
            # Simulate the trial. Rockenfeller's approach is used by default
            res = muscle.simulate('fast_benchmark_iso_length_' + str(scale[v]) + '_' + str(freqs[v]) + '.csv', t, get_l_isometric, get_v, get_z, l_init, write_output=0, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
            F = res['F_TOT(t)']

        benchmark_output[k] = F

        # Read experimental data
        exp_data_file = f'iso_l_{scale[v]}_interp.npy'
        exp_data = read_data(exp_path + exp_data_file, 'force_force_npy', N)[1]
        experimental_data[k] = exp_data

        k += 1
    
    # Compute error between model output and experimental data according to Perreault (2003)
    errors = np.mean(np.absolute(benchmark_output - experimental_data), 1)
    errors_max = np.amax(np.absolute(benchmark_output - experimental_data), 1)
    
    # Plot benchmark results
    plt.title(muscle_model + ' - Submaximal activation Biological Benchmark - Length Effect\n')
    
    for k in range(5):

       plt.plot(t, benchmark_output[k], label=muscle_model)
       plt.plot(t, experimental_data[k], linewidth=0.75, label='Brown 1999')

    plt.ylim((0, 1))
    plt.xlim((0, 0.36))
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.suptitle("Fast muscle - Biological Benchmark - " + muscle_model)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    duration = time.time() - start
    print("\nBenchmark computation time: " + str(round(duration) // 60) + ' min ' + str(round(duration) % 60) + ' s')
    
    return benchmark_output, experimental_data, errors, muscle


def fast_biological_benchmark_dyn(muscle_model, activation):
    
    """
    Runs the benchmarks of fast muscle (cat caudfemoralis, from Brown 1999) in dynamic conditions (Figure 3B).
    Previously optimized values are used for the recruitment/activation parameters.
    First the experimental parameters are defined, and the muscle object is created accordingly.
    
    Parameters:
    -----------
    
    muscle_model : str
        strings that defines the model used for the muscle, i.e. 'Hatze' or 'Virtual Muscle'
    
    
    Output:
    -----------
    
    benchmark_output : Numpy array of shape (N_trials, N)
        force profiles (in Newtons) produced by the model for each trial of the benchmark
    
    experimental_data : Numpy array of shape (N_trials, N)
        upsampled experimental force profiles (in Newtons) for each trial of the benchmark
    
    errors : Numpy array of shape (N,)
        vector of mean absolute errors for each trial of the benchmark
    
    muscle : muscle object
        muscle object computed to run the benchmark
    
    
    """
    
    start = time.time()
    print(activation + '- maximal activation - Fast Benchmark: ' + muscle_model)

    # Muscle parameters
    name = 'Cat caudofemoralis'
    alpha0 = 1
    L0_M = 0.056
    LT_slack = 0.017
    LT_opt = LT_slack * 1.08
    F0_max = 15.4
    fibre_type_profile = 0.95
    muscle.max_stim_rate = 100

    path = '..\\biologicalBenchmark\\fastMuscle\\dyn\\'

    # Create muscle object
    muscle = create_muscle(muscle_model, name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile)
    
    if activation == 'sub':

        # Number of integration steps
        N = 10001
        t = np.linspace(0, 0.2, N-1)

        # Read displacement data
        disp_short = read_data(path + 'disp_sub_short_interp.npy', 'force_force_npy', N)[1]
        disp_length = read_data(path + 'disp_sub_length_interp.npy', 'force_force_npy', N)[1]
        disp = [disp_short, disp_length]

        # Stimulation rates for constant stimulation trials
        freqs = [20, 40, 60, 40]
        scale = [0.95, 0.8, 1.1, 0.95]
    
        N_trials = 8
        benchmark_output = np.zeros((N_trials, N))
        experimental_data = np.zeros((N_trials, N))
    
        k = 0
        muscle.max_stim_rate = 100
    
        for j in range(2):
        
            if j == 0:
                disp_type = '\u00B1 shortening'
                disp = disp[:,j]
            else:
                disp_type = '\u00B1 lengthening'
                disp = disp[:,j]

            for v in range(len(freqs)):
            
                print('\nTrial ' + str(k+1) + ' : ' + disp_type + ' - Firing rate = ' + str(freqs[v]) + ' Hz\n')               
                
                # Recruitment parameters optimized with isometric trials
                dt = [] # to be updated
                z = [] # to be updated
                
                # Define the length profile throughout the experiment
                l_init = ((L0_M * scale[v])* np.cos(alpha0 * np.pi / 180)) * disp[0] + LT_opt
                
                def get_l(t):
                    Lmt = ((L0_M * scale[v])* np.cos(alpha0 * np.pi / 180)) * disp + LT_opt
                    return Lmt
                
                # Define the stimulation rate profile
                def get_v(t):
                    if t >= 0 and t <= 0.2:
                        return freqs[v] / muscle.max_stim_rate
                    else:
                        return 0
                
                # Define initial state
                n_0 = 0
                if n_0 == 1:
                    psi_0 = muscle.c * get_v(0)
                else:
                    psi_0 = 0
                
                # Define the recruitment rate profile
                def get_z(t):
                    if t >= 0 and t < dt[v]:
                        return z[v]
                    else:
                        return 0
                
                # Simulate the trial. Rockenfeller's approach is used by default
                res = muscle.simulate('Fast_benchmark_dyn_length_' + str(k+1) + '_Hatze.csv', t, get_l, get_v, get_z, l_init, write_output=1, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
                F = res['F_TOT(t)']
                
                benchmark_output[k] = F

                # Read experimental data
                freq = freqs[v]
                disp_label = 'short' if j == 0 else 'length'
                exp_data_file = f'{freq}_{scale[v]}_{disp_label}_interp.npy'
                exp_data = read_data(path + exp_data_file, 'force_force_npy', N)[1]
                experimental_data[k] = exp_data
            
                k += 1
    
        # Compute error between model output and experimental data according to Perreault (2003)
        errors = np.mean(np.absolute(benchmark_output - experimental_data), 1) 
        errors_max = np.amax(np.absolute(benchmark_output - experimental_data), 1) 
    
        # Plot benchmark results
        fig, axs = plt.subplots(2, 2, figsize=(8, 12))
        plt.title(muscle_model + ' - Fast muscle benchmark\n')
    
    elif activation == 'max':

        # Number of integration steps
        N = 1601
        t = np.linspace(0, 0.16, N-1)

        # Read displacement data
        disp_short = read_data(path + 'disp_max_short_interp.npy', 'force_force_npy', N)[1]
        disp_length = read_data(path + 'disp_max_length_interp.npy', 'force_force_npy', N)[1]

        # Stimulation rates for constant stimulation trials
        freqs = 120
    
        N_trials = 2
        benchmark_output = np.zeros((N_trials, N))
        experimental_data = np.zeros((N_trials, N))
    
        k = 0
        muscle.max_stim_rate = 100
    
        for j in range(2):
        
            if j == 0:
                disp_type = '\u00B1 shortening'
                disp = disp_short
            else:
                disp_type = '\u00B1 lengthening'
                disp = disp_length

            print('\nTrial ' + str(k+1) + ' : ' + disp_type + ' - Firing rate = 120 Hz\n')               
                
            # Recruitment parameters optimized with isometric trials
            dt = [] # to be updated
            z = [] # to be updated
                
            # Define the length profile throughout the experiment
            l_init = (L0_M * 0.95) * disp[0] + LT_opt
                
            def get_l(t):
                Lmt = (L0_M * 0.95) * disp + LT_opt
                return Lmt
                
            # Define the stimulation rate profile
            def get_v(t):
                if t >= 0 and t <= 0.16:
                    return freqs[v] / muscle.max_stim_rate
                else:
                    return 0
                
            # Define initial state
            n_0 = 0
            if n_0 == 1:
                psi_0 = muscle.c * get_v(0)
            else:
                psi_0 = 0
                
            # Define the recruitment rate profile
            def get_z(t):
                if t >= 0 and t < dt[v]:
                    return z[v]
                else:
                    return 0
                
            # Simulate the trial. Rockenfeller's approach is used by default
            res = muscle.simulate('Fast_benchmark_dyn_length_' + str(k+1) + '_Hatze.csv', t, get_l, get_v, get_z, l_init, write_output=1, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
            F = res['F_TOT(t)']
                
            benchmark_output[k] = F

            # Read experimental data
            disp_label = 'short' if j == 0 else 'length'
            exp_data_file = f'{freq}_0.95_{disp_label}_interp.npy'
            exp_data = read_data(path + exp_data_file, 'force_force_npy', N)[1]                
            experimental_data[k] = exp_data
            
            k += 1
    
        # Compute error between model output and experimental data according to Perreault (2003)
        errors = np.mean(np.absolute(benchmark_output - experimental_data), 1) 
        errors_max = np.amax(np.absolute(benchmark_output - experimental_data), 1) 


    return benchmark_output, experimental_data, errors, muscle




# =============================================================================
#     HATZE - PARAMETERS CALIBRATION
# =============================================================================


def calibrate_maximal_activation_benchmark_Hatze():
    
    """
    Calibrates Hatze's muscle model for the Maximal-Activation benchmark by optimizing the values of the maximum isometric force and the a3 parameters (Force-Velocity relationship).
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data, for the trial with 1 mm displacement.
    
    Output:
    -----------
    
    F0_max : float
        optimized value of the maximum isometric force
    
    a3 : float
        optimized value of the a3 parameter for Hatze's Force-Velocity relationship
    
    """
    
    start = time.time()
    print("\nParameters optimization - Maximal Activation Benchmark: Hatze\n")
    
    # Number of integration steps
    N = 10000    

    # Read data
    path = '..\\biologicalBenchmark\\slowMuscle_maximalActivation\\'
    t, displacement = read_data(path + 'displacement.dat', 'displacement_M', N)[:2]
    
    # Muscle parameters
    name = 'Rat soleus'
    alpha0 = 6
    L0_M = 0.0171
    LT_slack = L0_M
    LT_opt = LT_slack * 1.08
    F0_max_0 = 1.17
    fibre_type_profile = 0.1       # Novak (Physiol. Res., 2010) or Fukutani (Scientific reports, 2023)

    # Create muscle object
    muscle = create_muscle('Hatze', name, L0_M, LT_opt, LT_slack, alpha0, F0_max_0, fibre_type_profile)
    
    # Benchmark parameters for amplitudes of displacement
    max_displacement_amplitudes = 1
    displacement_offset = -0.002
        
    # Read experimental data
    exp_data = read_data(path + 'force_trial5.dat', 'force_M', N)[1]
    
    # Function to optimize by least_squares fitting
    def optimize_activation(params):

        F0_max, a3 = params
        
        a3 *= 3.2
        
        print('F0_max', F0_max)
        print('a3', a3)
        
        muscle.set_contraction_params(a3)
        
        # Define the muscle length profile        
        def get_l(t):
            i = max(round(N*t/2) - 1, 0)
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + max_displacement_amplitudes * 0.001 * displacement[i]
            return Lmt
        
        l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset
        
        # Define the average stimulation rate profile
        def get_v(t):
            return 70 / muscle.max_stim_rate      # 70 Hz stimulation
        
        # Define the average recruitment rate profile
        def get_z(t):
            if t >= 0 and t <= 0.06993373 * (1 - n_0):  # recruitment time needed to get q(t) = 1 (fully activated muscle) with current value of the 'enco' attribute
                return 1
            else:
                return 0
        
        n_0 = 1
        if n_0 == 1:
            psi_0 = muscle.c * get_v(0)
        else:
            psi_0 = 0
        
        # Simulate the trial
        F_ = muscle.simulate('maximalActivationBenchmark_Trial5.csv', t, get_l, get_v, get_z, l_init, write_output=0, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1, potentiation=1)['F_TOT(t)']
        F = F_ * F0_max
        
        return F - exp_data
    
    print('\nStart optimization')
    res = sp.optimize.least_squares(optimize_activation, [F0_max_0, 1], bounds=([0, 0], [2, 2]), diff_step=[0.0001, 0.0001])
            
    F0_max = res.x[0]
    a3 = res.x[1] * 3.2
            
    print('\nOptimized F0_max', F0_max)
    print('Optimized a3', a3)
    
    # Assess the optimization by comparing the experimental and simulated force profiles
    muscle.set_contraction_params(a3)
    
    # Define the muscle length profile    
    def get_l(t):
        i = max(round(N*t/2) - 1, 0)
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + max_displacement_amplitudes * 0.001 * displacement[i]
        return Lmt
    
    l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset
    
    # Define the average stimulation rate profile
    def get_v(t):
        return 70 / muscle.max_stim_rate      # 70 Hz stimulation
    
    # Define the average recruitment rate profile
    def get_z(t):
        if t >= 0 and t <= 0.06993373 * (1 - n_0):  # recruitment time needed to get q(t) = 1 (fully activated muscle) with current value of the 'enco' attribute
            return 1
        else:
            return 0
    
    # Define the initial state
    n_0 = 1
    if n_0 == 1:
        psi_0 = muscle.c * get_v(0)
    else:
        psi_0 = 0
    
    # Simulate the trial with optimized parameters
    F_ = muscle.simulate('maximalActivationBenchmark_Trial6.csv', t, get_l, get_v, get_z, l_init, write_output=0, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1, potentiation=1)['F_TOT(t)']
    
    benchmark_output = F_ * F0_max
    experimental_data = exp_data
    
    # Compute error between model output and experimental data according to Krylow & Sandercock (1997)
    errors = np.mean(np.absolute(benchmark_output - experimental_data)) / F0_max_0
    
    # Plot the result of the least-squares fitting
    plt.figure('Hatze - Parameters optimization - Maximal Activation Benchmark\n')
    plt.title('Hatze - Parameters activation - Error = ' + str(round(100 * errors, 1)) + '% - Maximal Activation Benchmark\n')
    plt.plot(t, benchmark_output, label='Hatze - Optimized')
    plt.plot(t, experimental_data, linewidth=0.75, label='Krylow & Sandercock (1997)')
    plt.ylim((0, 1.5))
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized force')
    plt.legend()
    plt.tight_layout()
    
    duration = time.time() - start
    print("\nBenchmark computation time: " + str(round(duration) // 60) + ' min ' + str(round(duration) % 60) + ' s')
    
    return F0_max, a3
    


def calibrate_activation_submaximal_Hatze(k):
    
    """
    Calibrates Hatze's muscle model for the trials with constant stimulation rates of the Submaximal-Activation benchmark.
    The values of the normalized recruitment rate 'z' and the duration of the recruitment phase 'dt' are optimized.
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data measured during isometric trials.
    
    Parameters:
    -----------
    
    k : int
        1, 2 or 3, to designate the mean stimulation rate of the isometric trial that is calibrated (10, 20 or 30 Hz respectively)
    
    Output:
    -----------
    
    z : float
        optimized value of the normalized recruitment rate
    
    dt : float
        optimized value of the duration of the motor unit recruitment phase
    
    """

    freq_type = 'c'

    N = 10000
    t = np.linspace(0, 2, N)
    
    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05

    # Create muscle object
    muscle = create_muscle('Hatze', name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile)
    
    # Stimulation rates for constant stimulation trials
    freqs = [10, 20, 30]
    
    # Read the experimental data for the isometric trial
    path = '..\\biologicalBenchmark\\slowMuscle_submaximalActivation\\c_freq\\'
    displacement_offset = 0.004
    freq = [10, 20, 30][k-1]
    isometric_file = f'force_isometric_{freq_type}{freq}.dat'
    isometric_force = read_data(path + isometric_file, 'force_force_SM', N)[1]
    
    muscle.max_stim_rate = 28

    def optimize_activation(params):

        dt, z = params
        
        dt *= 0.07
                
        print('dt', dt)
        print('z', z)
        
        # Define the muscle length profile
        l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        
        def get_l_isometric(t):
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
            return Lmt
        
        # Define the normalized average stimulation rate profile
        def get_v(t):
            if t >= 0 and t <= 1.5:
                return freqs[k-1] / muscle.max_stim_rate
            else:
                return 0
        
        # Define the normalized recruitment rate profile
        def get_z(t):
            if t >= 0 and t < dt:
                return z
            elif t >= 1.5 and t <= 1.5 + dt:
                return -1
            else:
                return 0
                
        # Define initial state
        n_0 = 0
        if n_0 == 1:
            psi_0 = muscle.c * get_v(0)
        else:
            psi_0 = 0
        
        # Simulate the isometric trial
        res = muscle.simulate('submaximalActivationBenchmark_constant' + str(k+1) + '.csv', t, get_l_isometric, get_v, get_z, l_init, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
        F = F0_max * res['F_TOT(t)']

        return F - isometric_force
    
    print('\nStart optimization')
    res = sp.optimize.least_squares(optimize_activation, [0.8, 0.8], bounds=([0, 0], [2, 1]), diff_step=[0.1, 0.01])
    
    dt = res.x[0] * 0.07
    z = res.x[1]

    print('\nOptimized dt', dt)
    print('Optimized z', z)
    
    # Assess the optimization by comparing the experimental and simulated force profiles
    l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        
    def get_l_isometric(t):
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        return Lmt
    
    # Define the optimized normalized average stimulation rate profile
    def get_v(t):
        if t >= 0 and t <= 1.5:
            return freqs[k-1] / muscle.max_stim_rate
        else:
            return 0
    
    # Define the optimized normalized recruitment rate profile
    def get_z(t):
        if t >= 0 and t < dt:
            return z
        elif t >= 1.5 and t <= 1.5 + dt:
            return -1
        else:
            return 0
    
    # Define initial state
    n_0 = 0
    if n_0 == 1:
        psi_0 = muscle.c * get_v(0)
    else:
        psi_0 = 0
    
    # Simulate the isometric trial with optimized parameters
    res = muscle.simulate('submaximalActivationBenchmark_constant' + str(k+1) + '.csv', t, get_l_isometric, get_v, get_z, l_init, write_output=0, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
    F = F0_max * res['F_TOT(t)']
    
    # Plot the result of the least-squares fitting
    plt.figure()
    plt.plot(t, isometric_force, label='Experimental isometric force')
    plt.plot(t, F, label='Model output with calibrated activation')
    plt.ylabel('Force')
    plt.xlabel('Time (s)')
    plt.xlim((0, 2))
    plt.legend()
    plt.show()
    
    return dt, z

def calibrate_lengths_Hatze(k):
    
    """
    Calibrates Hatze's muscle model for the trials with constant stimulation rates of the Submaximal-Activation benchmark at different lengths (see Kim 2015).
    The animal and experiment are the same of the submaximal benchmarks from Perreault et al.
    The values of the normalized recruitment rate 'z' and the duration of the recruitment phase 'dt' are optimized.
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data measured during isometric trials.
    
    Parameters:
    -----------
    
    k : int
        1:4, to designate the mean stimulation rate & imposed length of the isometric trial that is calibrated.
    
    Output:
    -----------
    
    z : float
        optimized value of the normalized recruitment rate
    
    dt : float
        optimized value of the duration of the motor unit recruitment phase
    
    """

    N = 14000
    t = np.linspace(0, 1.4, N)
    
    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05

    # Create muscle object
    muscle = create_muscle('Hatze', name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile)
    
    # Stimulation rates for constant stimulation trials
    freqs = [1, 10, 20, 40]
    t_end = [0.2, 0.85, 0.85, 0.85] 

    # Read the experimental data for the isometric trial
    exp_path = '..\\biologicalBenchmark\\slowMuscle_lengthEffect\\'
    displacement_offset = 0.004
    isometric_file = f'8_{freqs[k-1]}_interp.npy'
    isometric_force = read_data(exp_path + isometric_file, 'force_force_npy', N)[1]
    # dis_file = f'8_{freqs[k-1]}_times.npy'
    # t_start = read_data(exp_path + dis_file, 'discharge_npy', N)[1]
    # if k == 1: t_start = t_start.astype(float) # first discharge time to offset simulation time
    # else: t_start = t_start.astype(float)[0]
    t_start = 0

    muscle.max_stim_rate = 40

    def optimize_activation(params):

        dt, z = params
                
        if k == 1:
            t_end[k-1] = dt

        print('dt', dt)
        print('z', z)
        
        # Define the muscle length profile
        l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) - displacement_offset
        
        def get_l_isometric(t):
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) - displacement_offset 
            return Lmt
        
        # Define the normalized average stimulation rate profile
        def get_v(t):
            if t >= t_start and t <= t_end[k-1]:
                return freqs[k-1] / muscle.max_stim_rate
            else:
                return 0
        
        # Define the normalized recruitment rate profile
        def get_z(t):
            if t >= t_start and t < dt:
                return z
            elif t >= t_end[k-1] and t <= t_end[k-1] + dt:
                return -1
            else:
                return 0
                
        # Define initial state
        n_0 = 0
        if n_0 == 1:
            psi_0 = muscle.c * get_v(0)
        else:
            psi_0 = 0
        
        # Simulate the isometric trial
        res = muscle.simulate('lengths_Benchmark' + str(k) + '.csv', t, get_l_isometric, get_v, get_z, l_init, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
        F = F0_max * res['F_TOT(t)']

        return F - isometric_force
    
    print('\nStart optimization')
    res = sp.optimize.least_squares(optimize_activation, [0.2, 0.8], bounds=([0, 0.5], [t_end[k-1], 1]), diff_step=[0.01, 0.01])
    
    dt = res.x[0] 
    z = res.x[1]

    print('\nOptimized dt', dt)
    print('Optimized z', z)
    
    # Assess the optimization by comparing the experimental and simulated force profiles
    l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        
    def get_l_isometric(t):
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        return Lmt
    
    # Define the optimized normalized average stimulation rate profile
    def get_v(t):
        if t >= t_start and t <= t_end[k-1]:
            return freqs[k-1] / muscle.max_stim_rate
        else:
            return 0
    
    # Define the optimized normalized recruitment rate profile
    def get_z(t):
        if t >= t_start and t < dt:
            return z
        elif t >= t_end[k-1] and t <= t_end[k-1] + dt:
            return -1
        else:
            return 0
    
    # Define initial state
    n_0 = 0
    if n_0 == 1:
        psi_0 = muscle.c * get_v(0)
    else:
        psi_0 = 0
    
    # Simulate the isometric trial with optimized parameters
    res = muscle.simulate('lengths_Benchmark' + str(k) + '.csv', t, get_l_isometric, get_v, get_z, l_init, write_output=0, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
    F = F0_max * res['F_TOT(t)']
    
    # Plot the result of the least-squares fitting
    plt.figure()
    plt.plot(t, isometric_force, label='Experimental isometric force')
    plt.plot(t, F, label='Model output with calibrated activation')
    plt.ylabel('Force')
    plt.xlabel('Time (s)')
    plt.xlim((0, 1.4))
    plt.legend()
    plt.show()
    
    return dt, z


def calibrate_random_activation_submaximal_Hatze(k):
    
    """
    Calibrates Hatze's muscle model for the trials with random stimulation rates of the Submaximal-Activation benchmark.
    The values of the normalized recruitment rate 'z' and the normalized avergae stimulation rate 'v' are optimized as piecewise affine (sections of 0.1s).
    Some parameters need to be removed to optimize the control parameters as piecewise constant.
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data measured during isometric trials.
    
    Parameters:
    -----------
    
    k : int
        4, 5 or 6, to designate the mean stimulation rate of the isometric trial that is calibrated (10, 20 or 30 Hz respectively)
    
    
    Output:
    -----------
    
    z : Numpy array of shape (20,)
        optimized value of the normalized recruitment rate (z(t) = z_i + t * z0_i)
    
    v : Numpy array of shape (15,)
        optimized value of the normalized average stimulation rate (v(t) = v_i + t * v0_i)
    
    z : Numpy array of shape (20,)
        optimized value of the normalized recruitment rate (z(t) = z_i + t * z0_i)
    
    v : Numpy array of shape (15,)
        optimized value of the normalized average stimulation rate (v(t) = v_i + t * v0_i)
    
    """
    freq_type = 'v'

    N = 10000
    t = np.linspace(0, 2, N)
    
    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05

    # Create muscle object
    muscle = create_muscle('Hatze', name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile)

    # Stimulation rates for constant stimulation trials
    freqs = [10, 20, 30]
    
    # Read the experimental data for the isometric trial
    path = '..\\biologicalBenchmark\\slowMuscle_submaximalActivation\\v_freq\\'
    displacement_offset = 0.004
    freq = [10, 20, 30][k-1]
    isometric_file = f'force_isometric_{freq_type}{freq}.dat'
    isometric_force_random_stim = read_data(path + isometric_file, 'force_force_SM', N)[1]
    
    muscle.max_stim_rate = 28
    
    def optimize_activation(params):
        
        z = params[:20]
        v = params[20:35]
        z0 = params[35:55]
        v0 = params[55:]

        print('z', z)
        print('v', v)
        print('z0', z0)
        print('v0', v0)
                
        l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        
        def get_l_isometric(t):
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
            return Lmt
        
        # Define both control parameters
        def get_v(t):
            if t >= 0 and t <= 1.5:
                i = int(10 * t)
                return v0[i] * t + v[i] * freqs[(k-1)%3] / muscle.max_stim_rate
            else:
                return 0
        
        def get_z(t):
            if t == 2:
                return 0
            i = int(10 * t)
            return z[i] + z0[i] * t
                
        # Define initial state
        n_0 = 0
        if n_0 == 1:
            psi_0 = muscle.c * get_v(0)
        else:
            psi_0 = 0
        
        res = muscle.simulate('submaximalActivationBenchmark_random' + str(k+1) + '.csv', t, get_l_isometric, get_v, get_z, l_init, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
        F = F0_max * res['F_TOT(t)']

        return F - isometric_force_random_stim
    
    print('\nStart optimization')
    x0 = 0.8 * np.hstack((np.ones(15), -np.ones(5), np.ones(15), np.zeros(35)))
    bounds = (np.hstack((-np.ones(20), np.zeros(15), -np.inf*np.ones(35))),
              np.hstack((np.ones(15), np.zeros(5), muscle.max_stim_rate / freqs[(k-1)%3] * np.ones(15), np.inf*np.ones(35))))
    res = sp.optimize.least_squares(optimize_activation, x0, bounds=bounds)
    
    z = res.x[:20]
    v = res.x[20:35] * freqs[(k-1)%3]
    z0 = res.x[35:55]
    v0 = res.x[55:]
    
    # Assess the optimization by comparing the experimental and simulated force profiles
    print('\nOptimized z', z)
    print('Optimized v', v)
    print('Optimized z0', z0)
    print('Optimized v0', v0)
    
    # Define the muscle length profile
    l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        
    def get_l_isometric(t):
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        return Lmt
    
    # Define the normalized average stimulation rate profile
    def get_v(t):
        if t >= 0 and t <= 1.5:
            i = int(10 * t)
            return v0[i] * t + v[i] / muscle.max_stim_rate
        else:
            return 0
    
    # Define the normalized recruitment rate profile
    def get_z(t):
        if t == 2:
            return 0
        i = int(10 * t)
        return z[i] + z0[i] * t
    
    # Define the initial state
    n_0 = 0
    if n_0 == 1:
        psi_0 = muscle.c * get_v(0)
    else:
        psi_0 = 0
    
    # Run the isometric trial with optimized parameters
    res = muscle.simulate('submaximalActivationBenchmark_random' + str(k+1) + '.csv', t, get_l_isometric, get_v, get_z, l_init, write_output=0, ignore_SE=0, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
    F = F0_max * res['F_TOT(t)']
    
    # Plot the result of the least-squares fitting
    plt.figure()
    plt.plot(t, isometric_force_random_stim, label='Experimental isometric force')
    plt.plot(t, F, label='Model output with calibrated activation')
    plt.ylabel('Force')
    plt.xlabel('Time (s)')
    plt.xlim((0, 2))
    plt.legend()
    plt.show()
    
    return z, v, z0, v0


def calibrate_activation_fast_Hatze(k):
    
    """
    Calibrates Hatze's muscle model for the isometric trials of cat caudofemoralis stimulated with different frequencies (see Brown 1999).
    The values of the normalized recruitment rate 'z' and the duration of the recruitment phase 'dt' are optimized.
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data measured during isometric trials.
    
    Parameters:
    -----------
    
    k : int
        1:5, to designate the mean stimulation rate & imposed length of the isometric trial that is calibrated.
    
    Output:
    -----------
    
    z : float
        optimized value of the normalized recruitment rate
    
    dt : float
        optimized value of the duration of the motor unit recruitment phase
    
    """

    N = 6000
    t = np.linspace(0, 0.6, N)
    
    # Muscle parameters from Brown 1998 (CF architecture)
    name = 'Cat caudofemoralis'
    alpha0 = 5
    L0_M = 0.056
    LT_slack = 0.017
    LT_opt = LT_slack * 1.08
    F0_max = 15.4
    fibre_type_profile = 0.95 # fast fibre

    # Create muscle object
    muscle = create_muscle('Hatze', name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, tendon=0)
    
    # Stimulation rates for constant stimulation trials
    freqs = [15, 30, 40, 50, 120]
    t_end = [0.46, 0.25, 0.18, 0.13, 0.09] 

    # Read the experimental data for the isometric trial
    exp_path = '..\\biologicalBenchmark\\fastMuscle\\iso\\'
    isometric_file = f'iso_f_{freqs[k-1]}_interp.npy'
    isometric_force = read_data(exp_path + isometric_file, 'force_force_npy', N)[1]

    muscle.max_stim_rate = freqs[-1]

    def optimize_activation(params):

        dt, z = params
                
        print('dt', dt)
        print('z', z)
        
        # Define the muscle length profile
        l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180)
        
        def get_l_isometric(t):
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180)
            return Lmt
        
        # Define the normalized average stimulation rate profile
        def get_v(t):
            if t > 0 and t <= t_end[k-1]:
                return freqs[k-1] / muscle.max_stim_rate
            else:
                return 0
        
        # Define the normalized recruitment rate profile
        def get_z(t):
            if t >= 0 and t < dt:
                return z
            elif t >= t_end[k-1]:
                return -1
            else:
                return 0
                
        # Define initial state
        n_0 = 0
        if n_0 == 1:
            psi_0 = muscle.c * get_v(0)
        else:
            psi_0 = 0
        
        # Simulate the isometric trial
        res = muscle.simulate('fast_Benchmark' + str(k) + '.csv', t, get_l_isometric, get_v, get_z, l_init, ignore_SE=1, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
        F = res['F_TOT(t)']

        return F - isometric_force
    
    print('\nStart optimization')
    res = sp.optimize.least_squares(optimize_activation, [0.08, 0.8], bounds=([0, 0.4], [0.4, 1]), diff_step=[0.01, 0.01])
    
    dt = res.x[0] 
    z = res.x[1]

    print('\nOptimized dt', dt)
    print('Optimized z', z)
    
    # Assess the optimization by comparing the experimental and simulated force profiles
    l_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180)
        
    def get_l_isometric(t):
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180)
        return Lmt
    
    # Define the optimized normalized average stimulation rate profile
    def get_v(t):
        if t >= 0 and t <= t_end[k-1]:
            return freqs[k-1] / muscle.max_stim_rate
        else:
            return 0
    
    # Define the optimized normalized recruitment rate profile
    def get_z(t):
        if t >= 0 and t < dt:
            return z
        elif t >= t_end[k-1]:
            return -1
        else:
            return 0
    
    # Define initial state
    n_0 = 0
    if n_0 == 1:
        psi_0 = muscle.c * get_v(0)
    else:
        psi_0 = 0
    
    # Simulate the isometric trial with optimized parameters
    res = muscle.simulate('fast_Benchmark' + str(k) + '.csv', t, get_l_isometric, get_v, get_z, l_init, write_output=0, ignore_SE=1, icbel=0, n_0=n_0, psi_0=psi_0, lbda_opt_shift=1)
    F = res['F_TOT(t)']
    
    # Plot the result of the least-squares fitting
    plt.figure()
    plt.plot(t, isometric_force, label='Experimental isometric force')
    plt.plot(t, F, label='Model output with calibrated activation')
    plt.ylabel('Force')
    plt.xlabel('Time (s)')
    plt.xlim((0, 0.6))
    plt.legend()
    plt.show()
    
    return dt, z



# =============================================================================
#     VIRTUAL MUSCLE - PARAMETERS CALIBRATION
# =============================================================================


def calibrate_maximal_activation_benchmark_VM():
    
    """
    Calibrates the 'Virtual Muscle' muscle model for the Maximal-Activation benchmark by optimizing the values of some parameters.
    The maximum isometric force, the maximum contracting velocities for both slow and fast fibres, and the recruitment threshold for fast fibres Ur are optimized.
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data, for the trial with 1 mm displacement.
    
    Output:
    -----------
    
    Ur : float
        optimized value of the recruitment threshold for fast fibres
    
    F0_max : float
        optimized value of the maximum isometric force
    
    Vmax_slow : float
        optimized value of the maximum contracting velocity of slow fibres
    
    Vmax_fast : float
        optimized value of the maximum contracting velocity of fast fibres
    
    """
    
    start = time.time()
    print("\nParameters optimization - Maximal Activation Benchmark: Virtual Muscle\n")
    
    # Number of integration steps
    N = 10000

    # Read displacement data
    path = '..\\biologicalBenchmark\\slowMuscle_maximalActivation\\'
    t, displacement = read_data(path + 'displacement.dat', 'displacement_M', N)[:2]
    
    # Muscle parameters
    name = 'Rat soleus'
    alpha0 = 6
    L0_M = 0.0171
    LT_slack = L0_M
    LT_opt = LT_slack * 1.08
    F0_max_0 = 1.17
    fibre_type_profile = 0.1       # Novak (Physiol. Res., 2010) or Fukutani (Scientific reports, 2023)
    f_05 = (35, 35)
    
    # Benchmark parameters for amplitudes of displacement
    max_displacement_amplitudes = 1
    displacement_offset = -0.002
        
    # Read experimental force data of the isometric trial
    exp_data = read_data(path + 'force_trial5.dat', 'force_M', N)[1]
    
    # Function to optimize by least-squares fitting
    def optimize_activation(params):

        Ur, F0_max, Vmax_slow, Vmax_fast = params
        
        Vmax_slow *= 10
        Vmax_fast *= 10
        
        print('Ur', Ur)
        print('F0_max', F0_max)
        print('Vmax_slow', Vmax_slow)
        print('Vmax_fast', Vmax_fast)
        
        # Create muscle object
        muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur, f_05=f_05)
        muscle.fibres_list['slow'].Vmax = Vmax_slow
        muscle.fibres_list['fast'].Vmax = Vmax_fast

        # Define the length profile of the musculotendon system throughout the simulation
        def get_muscle_length(t):
            i = max(round(N*t/2) - 1, 0)
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + max_displacement_amplitudes * 0.001 * displacement[i]
            return Lmt
        
        Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset
        
        # Define activation
        def get_activation(t):
            return 1
        
        # The muscle is initially fully activated
        U_eff_0 = 1
        f_eff_0 = 2
        f_int_0 = 2
        
        muscle.eta = 0.016
        
        F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max
        
        return F - exp_data
    
    print('\nStart optimization')
    res = sp.optimize.least_squares(optimize_activation, [0.8, F0_max_0, -1, -1], bounds=([0, 0, -2, -2], [1, 2, 0, 0]))#, diff_step=[0.001, 1e-8, 1e-8])
            
    Ur = res.x[0]
    F0_max = res.x[1]
    Vmax_slow = res.x[2] * 10
    Vmax_fast = res.x[3] * 10
            
    print('\nOptimized Ur', Ur)
    print('Optimized F0_max', F0_max)
    print('Optimized Vmax_slow', Vmax_slow)
    print('Optimized Vmax_fast', Vmax_fast)
    
    # Assess the optimization by comparing the experimental and simulated force profiles
    muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur)
    muscle.fibres_list['slow'].Vmax = Vmax_slow
    muscle.fibres_list['fast'].Vmax = Vmax_fast
    
    # Deduct inputs depending on the muscle model and compute the model output 
    def get_muscle_length(t):
        i = max(round(N*t/2) - 1, 0)
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset + max_displacement_amplitudes * 0.001 * displacement[i]
        return Lmt
    
    Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset
    
    # Define activation
    def get_activation(t):
        return 1
    
    # The muscle is initially fully activated
    U_eff_0 = 1
    f_eff_0 = 2
    f_int_0 = 2
    
    muscle.eta = 0.016
    
    # Run the isometric trial with optimized parameters
    F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max
    
    benchmark_output = F
    experimental_data = exp_data
    
    # Compute error between model output and experimental data according to Krylow & Sandercock (1997)
    errors = np.mean(np.absolute(benchmark_output - experimental_data)) / F0_max_0
    
    # Plot the result of the least-squares fitting
    plt.figure('Virtual Muscle - Parameters optimization - Maximal Activation Benchmark\n')
    plt.title('Virtual Muscle - Parameters activation - Error = ' + str(round(100 * errors, 1)) + '% - Maximal Activation Benchmark\n')
    plt.plot(t, benchmark_output / F0_max, label='Virtual Muscle')
    plt.plot(t, experimental_data / F0_max_0, linewidth=0.75, label='Krylow & Sandercock (1997)')
    plt.ylim((0, 1.5))
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized force')
    plt.legend()
    plt.tight_layout()
    
    duration = time.time() - start
    print("\nBenchmark computation time: " + str(round(duration) // 60) + ' min ' + str(round(duration) % 60) + ' s')
    
    return Ur, F0_max, Vmax_slow, Vmax_fast
    


def calibrate_activation_submaximal_VM(k):
    
    """
    Calibrates Hatze's muscle model for the trials with constant stimulation rates of the Submaximal-Activation benchmark.
    The values of the activation level and the recruitment threshold for fast fibres Ur are optimized.
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data measured during isometric trials.
    
    Parameters:
    -----------
    
    k : int
        1, 2 or 3, to designate the mean stimulation rate of the isometric trial that is calibrated (10, 20 or 30 Hz respectively)
    
    Output:
    -----------
    
    Ur : float
        optimized value of tthe recruitment threshold for fast fibres
    
    act : float
        optimized value of the activation level
    
    """
    
    freq_type = 'c'
    
    N = 10000
    t = np.linspace(0, 2, N)
    
    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05
    f_05 = (50, 50)
    
    # Create muscle object
    muscle = create_muscle('Virtual Muscle', name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, f_05=f_05)

    # Stimulation rates for constant stimulation trials
    freqs = [10, 20, 30]
    
    path = '..\\biologicalBenchmark\\slowMuscle_submaximalActivation\\c_freq\\'
    displacement_offset = 0.004
    freq = [10, 20, 30][k-1]
    isometric_file = f'force_isometric_{freq_type}{freq}.dat'
    isometric_force = read_data(path + isometric_file, 'force_force_SM', N)[1]

    # Function to optimize by least-squares fitting
    def optimize_activation(params):

        Ur = params[0]
        act = params[1]
        
        print('Ur', Ur)
        print('activation', act)
        
        # Create muscle object
        muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur, f_05=f_05)
                
        # Define the muscle length profile
        def get_muscle_length(t):
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
            return Lmt
        
        Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        
        # Define the activation profile
        def get_activation(t):
            if t <= 1.5:
                return act
            else:
                return 0
        
        # The muscle is initially fully activated
        U_eff_0 = 0
        f_eff_0 = 0
        f_int_0 = 0
        
        # Run the isometric trial
        F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max

        return F - isometric_force
    
    print('\nStart optimization')
    res = sp.optimize.least_squares(optimize_activation, [0.8, freqs[k-1] / 100], bounds=([0, 0], [1, 1]))
    
    Ur = res.x[0]
    act = res.x[1]
    
    print('Ur', Ur)
    print('act', act)
    
    # Create muscle object
    muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur, f_05=f_05)
        
    # Define the muscle length profile
    def get_muscle_length(t):
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        return Lmt
    
    Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
    
    # Define the activation profile
    def get_activation(t):
        if t <= 1.5:
            return act
        else:
            return 0
    
    # The muscle is initially fully activated
    U_eff_0 = 0
    f_eff_0 = 0
    f_int_0 = 0
            
    # Rune the isometric trial with optimized parameters
    F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max
    
    # Plot the result of the least-squares fitting
    plt.figure()
    plt.plot(t, isometric_force, label='Experimental isometric force')
    plt.plot(t, F, label='Model output with calibrated activation')
    plt.ylabel('Force')
    plt.xlabel('Time (s)')
    plt.xlim((0, 2))
    plt.legend()
    plt.show()
    
    return Ur, act



def calibrate_random_activation_submaximal_VM(k):
    
    """
    Calibrates the 'Virtual Muscle' muscle model for the trials with random stimulation rates of the Submaximal-Activation benchmark.
    The values of the recruitment threshold for fast fibres Ur and the activation level (as piecewise constant on sections of 0.1s) are optimized.
    Their values are optimized by fitting (least-squares) the force response of the model on the experimental force data measured during isometric trials.
    
    Parameters:
    -----------
    
    k : int
        4, 5 or 6, to designate the mean stimulation rate of the isometric trial that is calibrated (10, 20 or 30 Hz respectively)
    
    
    Output:
    -----------
    
    Act : Numpy array of shape (15,)
        optimized values of the piecewise constant activation level
    
    Ur : float
        optimized value of the recruitment threshold for fast fibres
    
    """
    
    freq_type = 'v'
    
    N = 10000
    t = np.linspace(0, 2, N)
    
    # Muscle parameters
    name = 'Cat soleus'
    alpha0 = 7.5
    L0_M = 0.030
    LT_slack = 0.065
    LT_opt = LT_slack * 1.08
    F0_max = 25.1
    fibre_type_profile = 0.05
    f_05 = (50, 50)

    # Stimulation rates for constant stimulation trials
    freqs = [10, 20, 30]
        
    # Read the experimental force profile for the isometric trial
    path = '..\\biologicalBenchmark\\slowMuscle_submaximalActivation\\v_freq\\'
    displacement_offset = 0.004
    freq = [10, 20, 30][k-1]
    isometric_file = f'force_isometric_{freq_type}{freq}.dat'
    isometric_force_random_stim = read_data(path + isometric_file, 'force_force_SM', N)[1]

    # Function to optimized by least-squares fitting
    def optimize_activation(params):
        
        Act = params[:15]
        Ur = params[-1]

        print('Ur', Ur)
        print('Act', Act)
        
        # Create muscle object
        muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur, f_05=f_05)
                
        # Define the muscle length profile 
        def get_muscle_length(t):
            Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
            return Lmt
        
        Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        
        # Define the activation profile
        def get_activation(t):
            if t <= 1.5:
                i = int(10*t)
                return Act[i]
            else:
                return 0
        
        # The muscle is initially fully activated
        U_eff_0 = 0
        f_eff_0 = 0
        f_int_0 = 0
                
        F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max

        return F - isometric_force_random_stim
    
    print('\nStart optimization')
    x0 = np.hstack((freqs[(k-1)%3] / 100 * np.ones(15), 0.8))
    bounds = (np.zeros(16), np.ones(16))
    res = sp.optimize.least_squares(optimize_activation, x0, bounds=bounds)
    
    Act = res.x[:15]
    Ur = res.x[-1]

    print('\nOptimized Ur', Ur)
    print('Optimized Act', Act)
        
    # Create muscle object
    muscle = create_muscle("Virtual Muscle", name, L0_M, LT_opt, LT_slack, alpha0, F0_max, fibre_type_profile, Ur=Ur, f_05=f_05)
            
    # Define the muscle length profile
    def get_muscle_length(t):
        Lmt = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
        return Lmt
    
    Lmt_init = LT_opt + L0_M * np.cos(alpha0 * np.pi / 180) + displacement_offset - 0.008
    
    # Define activation
    def get_activation(t):
        if t <= 1.5:
            i = int(10*t)
            return Act[i]
        else:
            return 0
    
    # The muscle is initially fully activated
    U_eff_0 = 0
    f_eff_0 = 0
    f_int_0 = 0
            
    # Simulate the isometric trial with optimized parameters
    F = muscle.simulate(t, get_activation, get_muscle_length, Lmt_init, U_eff_0=U_eff_0, f_eff_0=f_eff_0, f_int_0=f_int_0)[0] * F0_max

    # Plot the result of the least-squares fitting
    plt.figure()
    plt.plot(t, isometric_force_random_stim, label='Experimental isometric force')
    plt.plot(t, F, label='Model output with calibrated activation')
    plt.ylabel('Force')
    plt.xlabel('Time (s)')
    plt.xlim((0, 2))
    plt.legend()
    plt.show()
    
    return Act, Ur





if __name__ == '__main__':      
    
    ####################################################################
    # MAXIMAL ACTIVATION BENCHMARKS (Krilow 1997, rat soleus)
    ####################################################################

    # Maximal Activation Biological Benchmark
    # Hatze_benchmark_output, Hatze_experimental_data, Hatze_errors, muscle_Hatze = maximal_activation_biological_benchmark('Hatze')
    # VM_benchmark_output, VM_experimental_data, VM_errors, muscle_VM = maximal_activation_biological_benchmark('Virtual Muscle')
    
    # # Calibrate the muscle parameters for the maximal activation benchmark
    # params = calibrate_maximal_activation_benchmark_Hatze()
    # params = calibrate_maximal_activation_benchmark_VM()

    ###################################################################
    # SUBMAXIMAL ACTIVATION BENCHMARKS (Perreault 2003, cat soleus)
    ###################################################################

    # Run Submaximal Activation Biological Benchmark
    # benchmark_output_SM_Hatze, experimental_data_SM_Hatze, errors_SM_Hatze, muscle_SM_Hatze = submaximal_activation_biological_benchmark('Hatze')
    # benchmark_output_SM_VM, experimental_data_SM_VM, errors_SM_VM, muscle_SM_VM = submaximal_activation_biological_benchmark('Virtual Muscle')
    # benchmark_output_SMR_Hatze, experimental_data_SMR_Hatze, errors_SMR_Hatze, muscle_SMR_Hatze = submaximal_activation_biological_benchmark_random('Hatze')
    # benchmark_output_SMR_VM, experimental_data_SMR_VM, errors_SMR_VM, muscle_SMR_VM = submaximal_activation_biological_benchmark_random('Virtual Muscle')
    
    # Calibrate the activation parameters on isometric trials (Hatze)
    # dt_opti = []
    # z_opti = []
    # for k in [1, 2, 3]:
    #     params_opti = calibrate_activation_submaximal_Hatze(k)
    #     dt_opti.append(params_opti[0])
    #     z_opti.append(params_opti[1])

    # # Calibrate the activation parameters on isometric trials (Virtual Muscle)
    # Ur_opti = []
    # act_opti = []
    # for k in [1, 2, 3]:
    #     params_opti = calibrate_activation_submaximal_VM(k)
    #     Ur_opti.append(params_opti[0])
    #     act_opti.append(params_opti[1])
    # Calibrate the activation parameters on isometric trials (Hatze)

    # Calibrate the stimulation rate parameters for random activation with experimental isometric trials (Hatze)
    # z_opti = []
    # v_opti = []
    # z0_opti = []
    # v0_opti = []
    # for k in [1, 2, 3]:
    #     params_opti = calibrate_random_activation_submaximal_Hatze(k)
    #     z_opti.append(params_opti[0])
    #     v_opti.append(params_opti[1])
    #     z0_opti.append(params_opti[2])
    #     v0_opti.append(params_opti[3])
    # z_opti  = np.array(z_opti)
    # v_opti  = np.array(v_opti)
    # z0_opti = np.array(z0_opti)
    # v0_opti = np.array(v0_opti)
    # print('\nEnd of optimization')
    # print('v_opti')
    # print(v_opti)
    # print('z_opti')
    # print(z_opti)
    # print('v0_opti')
    # print(v0_opti)
    # print('z0_opti')
    # print(z0_opti)
    
    # # Calibrate the stimulation rate parameters for random activation with experimental isometric trials (Virtual Muscle)
    # params_opti = []
    # Ur_opti = []
    # Act_opti = []
    # for k in [1, 2, 3]:
    #     params_opti = calibrate_random_activation_submaximal_VM(k)
    #     Act_opti.append(params_opti[0])
    #     Ur_opti.append(params_opti[1])
    # Act_opti = np.array(Act_opti)
    # Ur_opti  = np.array(Ur_opti)
    # print('\nEnd of optimization')
    # print('Ur_opti')
    # print(Ur_opti)
    # print('Act_opti')
    # print(Act_opti)

    #########################################################################
    # LENGTH VARIATION BENCHMARKS (Kim 2015, cat soleus)
    #########################################################################

    # Run simulations
    # benchmark_output_SM_Hatze, experimental_data_SM_Hatze, errors_SM_Hatze, muscle_SM_Hatze = lengths_biological_benchmark('Hatze')

    import os
    os.chdir(r'C:\Users\z5517249\Dropbox\UNSW_Andrea_Luca_PhD\Code\Python_Scripts\pyHatze\Theo_code')

    # Calibrate activation parameters
    dt_opti = []
    z_opti = []
    for k in [1, 2, 3, 4]:
        params_opti = calibrate_lengths_Hatze(k)
        dt_opti.append(params_opti[0])
        z_opti.append(params_opti[1])

    print('Optimized dt', dt_opti)
    print('Optimized z', z_opti)
        
    #########################################################################
    # FAST MUSCLE BENCHMARKS (BROWN 1999, cat caudofemoralis)
    #########################################################################

    # Run simulations
    # benchmark_output_SM_Hatze, experimental_data_SM_Hatze, errors_SM_Hatze, muscle_SM_Hatze = fast_biological_benchmark_iso('Hatze')
    # benchmark_output_SM_Hatze, experimental_data_SM_Hatze, errors_SM_Hatze, muscle_SM_Hatze = fast_biological_benchmark_dyn('Hatze')

    # Calibrate activation parameters
    # dt_opti = []
    # z_opti = []
    # for k in [5]:
    #     params_opti = calibrate_activation_fast_Hatze(k)
    #     dt_opti.append(params_opti[0])
    #     z_opti.append(params_opti[1])

    # print('Optimized dt', dt_opti)
    # print('Optimized z', z_opti)    
    

    