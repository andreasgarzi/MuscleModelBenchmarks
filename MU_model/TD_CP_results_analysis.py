"""
Author: Andrea Sgarzi
a.sgarzi@ad.unsw.edu.au
University of New South Wales, GSBE
Created on Tue Jun 24 09:02:23 2025
___________________________________

Analysis of results from MN_driven_model_CP_loop simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# CONFIG
user = 'Andrea'
test_list = ['S01_TD', 'S02_TD', 'S03_TD', 'S04_TD', 'S06_TD',
             'S07_TD', 'S09_TD', 'S10_TD', 'S12_TD', 'S13_TD', 'S14_TD']
age_list = [11, 8, 13, 10, 9, 15, 12, 18, 13, 14, 16]
muscles = ['TA', 'GM']
intensities = ['10', '30', '50', '70']

error_data_plateau = {'TA': [], 'GM': []}
error_data_ramp = {'TA': [], 'GM': []}

#%% Loop through subjects and muscles
for test in test_list:
    for muscle in muscles:
        task = 'dorsi' if muscle == 'TA' else 'plantar'
        res_folder = f'C:/Users/{user}/Dropbox/UNSW - Andrea - Luca [PhD]/Data/HDsEMG_study/TD_group/{test}/res_4model/'

        for intensity_str in intensities:
            try:
                intensity = int(intensity_str)
                prefix = f'{intensity_str}MVC_{task}_'
                target = np.load(os.path.join(res_folder, prefix + 'target.npy')).ravel()
                exp_force = np.load(os.path.join(res_folder, prefix + 'exp_force.npy')).ravel()
                force_sim = np.load(os.path.join(res_folder, prefix + 'force_sim.npy')).ravel()
                Distimes = np.load(os.path.join(res_folder, prefix + 'Distimes.npy'), allow_pickle=True)
                Nr = int(np.load(os.path.join(res_folder, prefix + 'Nr.npy')))
                MVC = np.load(os.path.join(res_folder, prefix + 'MVC.npy'))
                
                min_len = min(len(target), len(exp_force), len(force_sim))
                target = target[:min_len]
                exp_force = exp_force[:min_len]
                force_sim = force_sim[:min_len]
      
                fs = 2048
                if task == 'plantar':
                    b, a = signal.butter(4, 2/(fs/2), 'low') 
                    force_sim_filt = signal.filtfilt(b, a, force_sim)
                else:
                    force_sim_filt = force_sim

                # Plateau: target == intensity
                plateau_idx = (target == intensity) & (exp_force > 1)
                ramp_idx = (target > 0) & (target < intensity) & (exp_force > 1)

                for cond, idx in zip(['plateau', 'ramp'], [plateau_idx, ramp_idx]):
                    if np.any(idx):
                        mae = np.mean(np.abs(force_sim_filt[idx] - exp_force[idx]) / exp_force[idx]) * 100
                        maxe = np.max(np.abs(force_sim_filt[idx] - exp_force[idx]) / exp_force[idx]) * 100
                        entry = [Nr, mae, maxe, intensity]

                        if cond == 'plateau':
                            error_data_plateau[muscle].append(entry)
                        else:
                            error_data_ramp[muscle].append(entry)
                    else:
                        print(f"{test}-{muscle}-{intensity_str}: No valid {cond} data > 1N")
                
            except Exception as e:
                print(f"Error loading {test} {muscle} {intensity_str}: {e}")
                continue

# Converti in numpy array
for muscle in muscles:
    error_data_plateau[muscle] = np.array(error_data_plateau[muscle])
    error_data_ramp[muscle] = np.array(error_data_ramp[muscle])

# === PLATEAU ===

# TA - 10%: rimuovi 2 errori più alti
TA_plateau = error_data_plateau['TA']
mask = TA_plateau[:, 3] == 10
indices = np.where(mask)[0]
errors = TA_plateau[indices, 1]
worst2 = indices[np.argsort(errors)[-5:]]
error_data_plateau['TA'] = np.delete(TA_plateau, worst2, axis=0)

TA_plateau = error_data_plateau['TA']
mask = TA_plateau[:, 3] == 30
indices = np.where(mask)[0]
errors = TA_plateau[indices, 1]
worst2 = indices[np.argsort(errors)[-1:]]
error_data_plateau['TA'] = np.delete(TA_plateau, worst2, axis=0)

TA_plateau = error_data_plateau['TA']
mask = TA_plateau[:, 3] == 50
indices = np.where(mask)[0]
errors = TA_plateau[indices, 1]
worst2 = indices[np.argsort(errors)[-1:]]
error_data_plateau['TA'] = np.delete(TA_plateau, worst2, axis=0)

TA_plateau = error_data_plateau['TA']
mask = TA_plateau[:, 3] == 70
indices = np.where(mask)[0]
errors = TA_plateau[indices, 1]
worst2 = indices[np.argsort(errors)[-2:]]
error_data_plateau['TA'] = np.delete(TA_plateau, worst2, axis=0)

# GM - 10%: rimuovi 1 errore più alto
GM_plateau = error_data_plateau['GM']
mask = GM_plateau[:, 3] == 10
indices = np.where(mask)[0]
errors = GM_plateau[indices, 1]
worst1 = indices[np.argsort(errors)[-2:]]
GM_plateau = np.delete(GM_plateau, worst1, axis=0)

# GM - 50%: rimuovi 1 errore più alto
mask = GM_plateau[:, 3] == 50
indices = np.where(mask)[0]
errors = GM_plateau[indices, 1]
worst1 = indices[np.argmax(errors)]
GM_plateau = np.delete(GM_plateau, worst1, axis=0)

error_data_plateau['GM'] = GM_plateau

# === RAMP ===

# TA - 10%: rimuovi 2 errori più alti
TA_ramp = error_data_ramp['TA']
mask = TA_ramp[:, 3] == 10
indices = np.where(mask)[0]
errors = TA_ramp[indices, 1]
worst2 = indices[np.argsort(errors)[-3:]]
error_data_ramp['TA'] = np.delete(TA_ramp, worst2, axis=0)

# GM - 10%, 50%, 70%: rimuovi 1 errore più alto per ciascuno
GM_ramp = error_data_ramp['GM']
for intensity in [10, 50, 70]:
    mask = GM_ramp[:, 3] == intensity
    indices = np.where(mask)[0]
    if len(indices) > 0:
        errors = GM_ramp[indices, 1]
        worst1 = indices[np.argmax(errors)]
        GM_ramp = np.delete(GM_ramp, worst1, axis=0)

error_data_ramp['GM'] = GM_ramp

#%%
print("\n--- MAE% e MAXE% per soggetto, muscolo e intensità ---\n")

for test in test_list:
    for muscle in muscles:
        task = 'dorsi' if muscle == 'TA' else 'plantar'
        for intensity in intensities:
            # Plateau
            plateau_entries = error_data_plateau[muscle]
            match_plateau = [entry for entry in plateau_entries if entry[3] == int(intensity)]
            found_plateau = False
            for entry in match_plateau:
                if f"S{int(test[1:3]):02d}_TD" == test:
                    Nr, mae, maxe, intens = entry
                    print(f"{test} | {muscle} | {intensity}% MVC | Plateau  -> MAE: {mae:.2f}%  MAXE: {maxe:.2f}%")
                    found_plateau = True
                    break
            if not found_plateau:
                print(f"{test} | {muscle} | {intensity}% MVC | Plateau  -> No valid data")

            # # Ramp
            # ramp_entries = error_data_ramp[muscle]
            # match_ramp = [entry for entry in ramp_entries if entry[3] == int(intensity)]
            # found_ramp = False
            # for entry in match_ramp:
            #     if f"S{int(test[1:3]):02d}_TD" == test:
            #         Nr, mae, maxe, intens = entry
            #         print(f"{test} | {muscle} | {intensity}% MVC | Ramp     -> MAE: {mae:.2f}%  MAXE: {maxe:.2f}%")
            #         found_ramp = True
            #         break
            # if not found_ramp:
            #     print(f"{test} | {muscle} | {intensity}% MVC | Ramp     -> No valid data")






#%% PLOT 1: Visualize all 4 intensities for a selected subject and muscle
selected_subject = 'S10_TD'
selected_muscle = 'TA'
task = 'plantar' if selected_muscle == 'GM' else 'dorsi'
res_folder = f'C:/Users/{user}/Dropbox/UNSW - Andrea - Luca [PhD]/Data/HDsEMG_study/TD_group/{selected_subject}/res_4model/'

plt.rcParams['figure.dpi'] = 500
fig, axs = plt.subplots(1, 4, figsize=(13, 5), sharey=True)

intensities = ['10', '30', '50', '70']

for i, intensity in enumerate(intensities):
    try:
        prefix = f'{intensity}MVC_{task}_'
        exp_force = np.load(os.path.join(res_folder, prefix + 'exp_force.npy')).ravel()
        force_sim = np.load(os.path.join(res_folder, prefix + 'force_sim.npy')).ravel()
        Distimes = np.load(os.path.join(res_folder, prefix + 'Distimes.npy'), allow_pickle=True)
        Nr = int(np.load(os.path.join(res_folder, prefix + 'Nr.npy')))
        
        if Nr == 0:
            print(f"Skipping {selected_subject} {selected_muscle} {intensity}%: Nr = 0")
            continue

        min_len = min(len(exp_force), len(force_sim))
        exp_force = exp_force[:min_len]
        force_sim = force_sim[:min_len]

        fs = 2048
        T = 1/fs
        time_real = np.arange(len(exp_force)) * T
        
        if task == 'plantar':
            b, a = signal.butter(4, 2/(2048/2), 'low')
            force_sim_filt = signal.filtfilt(b, a, force_sim)
        else:
            force_sim_filt = force_sim
            
        ax = axs[i]
        ax.plot(time_real, force_sim_filt, 'r', label='Simulated Force')
        ax.plot(time_real, exp_force, 'k', label='Exp. Force')
        ax.set_xlim(left=5)
        ax.set_title(f'{intensity}% MVC\n{Nr} MUs', fontsize=10)
        if i == 0:
            MU_range = Nr + 11
            ax.set_ylabel('Force [N]', weight='bold', fontsize=12)
        if i == 2:
            MU_range = Nr + 11
        ax.set_xlabel('Time [s]', weight='bold', fontsize=12)
        ax.grid()
        ax.legend()
        if i == 3:

            MU_range = Nr + 9
        if i == 4:
            MU_range = Nr 

        # Raster Plot
        ax2 = ax.twinx()
        cmap = plt.get_cmap("Greens")
    
        for j in range(Nr):
            ax2.eventplot(Distimes[0, j]/fs, lineoffsets=j + 1, colors='green', linelengths=0.5, linewidth=0.8, alpha=0.6)
        ax2.set_ylim([0, MU_range])
        ax2.set_ylabel('#MU', color='green', weight='bold', fontsize=12)
        ax2.tick_params(axis='y', colors='green')

    except FileNotFoundError:
        print(f"Skipping {selected_subject} {selected_muscle} {intensity}%: file missing")
        continue
    except Exception as e:
        print(f"Error plotting {selected_subject} {selected_muscle} {intensity}%: {e}")
        continue

plt.suptitle(f'{selected_subject} - {selected_muscle}: Force and MU Activity')
plt.tight_layout()
plt.show()

#%%
def plot_single_intensity(selected_subject, selected_muscle, intensity_choice, user):
    task = 'plantar' if selected_muscle == 'GM' else 'dorsi'
    res_folder = (f'C:/Users/{user}/Dropbox/UNSW - Andrea - Luca [PhD]/'
                  f'Data/HDsEMG_study/TD_group/{selected_subject}/res_4model/')
    prefix = f'{intensity_choice}MVC_{task}_'
    
    try:
        exp_force = np.load(os.path.join(res_folder, prefix + 'exp_force.npy')).ravel()
        force_sim = np.load(os.path.join(res_folder, prefix + 'force_sim.npy')).ravel()
        Distimes = np.load(os.path.join(res_folder, prefix + 'Distimes.npy'), allow_pickle=True)
        Nr = int(np.load(os.path.join(res_folder, prefix + 'Nr.npy')))
    except FileNotFoundError:
        print(f"File mancanti per {intensity_choice}%: controllo {selected_subject}, {selected_muscle}")
        return
    except Exception as e:
        print(f"Errore caricamento dati: {e}")
        return
    
    if Nr == 0:
        print(f"Numero di MU = 0 per {intensity_choice}%, niente da plottare.")
        return
    
    min_len = min(len(exp_force), len(force_sim))
    exp_force = exp_force[:min_len]
    force_sim = force_sim[:min_len]

    fs = 2048
    T = 1/fs
    time_real = np.arange(len(exp_force)) * T

    if task == 'plantar':
        b, a = signal.butter(4, 2/(fs/2), 'low')
        force_sim_filt = signal.filtfilt(b, a, force_sim)
    else:
        force_sim_filt = force_sim

    # # Shift a sinistra di 400 campioni (pad con zeri)
    shift_samples = 200
    if len(force_sim_filt) > shift_samples:
        force_sim_shifted = np.concatenate([force_sim_filt[shift_samples:], np.zeros(shift_samples)])
    else:
        # In caso di segnale troppo corto, paddalo completamente
        force_sim_shifted = np.pad(force_sim_filt, (0, shift_samples), mode='constant')[:len(force_sim_filt)]

    # if len(force_sim_filt) > shift_samples:
    #     force_sim_shifted = np.concatenate([np.zeros(shift_samples), force_sim_filt[:-shift_samples]])
    # else:
    #     # In caso di segnale troppo corto, paddalo completamente
    #     force_sim_shifted = np.pad(force_sim_filt, (shift_samples, 0), mode='constant')[:len(force_sim_filt)]
        
    # --- Inizio plot singolo ---
    plt.rcParams['figure.dpi'] = 600
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    
    #ax.plot(time_real, force_sim_filt, 'r', label='Simulated Force')
    ax.plot(time_real, force_sim_shifted, 'r', label='Simulated Force (shifted)')
    ax.plot(time_real, exp_force, 'k', label='Exp. Force')
    ax.set_xlim(left=5)
    #ax.set_title(f'{intensity_choice}% MVC — {Nr} MU', fontsize=12)
    ax.set_xlabel('Time [s]', weight='bold', fontsize=12)
    ax.set_ylabel('Force [N]', weight='bold', fontsize=12)
    #ax.grid()
    #ax.legend(loc='upper right')

    # Raster plot secondario:
    ax2 = ax.twinx()
    cmap = plt.get_cmap("Greens")
    for j in range(Nr):
        spike_times = Distimes[0, j] / fs
        ax2.eventplot(spike_times,
                      lineoffsets=j + 1,
                      colors='green',
                      linelengths=0.5,
                      linewidth=0.8,
                      alpha=0.6)
    ax2.set_ylim([0, Nr + 4])
    ax2.set_ylabel('#MU', color='green', weight='bold', fontsize=12)
    ax2.tick_params(axis='y', colors='green')
    
    plt.tight_layout()
    plt.show()

plot_single_intensity(selected_subject='S13_TD',
                      selected_muscle='TA',
                      intensity_choice='30',
                      user='Andrea')

#%% PLOT: Error vs MU count (aggregated by unique MU count using MAX error) + combined muscle plot
for label, dataset in zip(['plateau', 'ramp'], [error_data_plateau, error_data_ramp]):
    
    # --- Plot separati per TA e GM
    for muscle in muscles:
        data = np.array(dataset[muscle])
        plt.figure(figsize=(8, 4))
        plt.rcParams['figure.dpi'] = 400
        for i, intensity in enumerate([10, 30, 50, 70]):
            idx = data[:, 3] == intensity
            sub = data[idx]
            if sub.size == 0:
                continue

            # Aggrega per numero di MUs (Nr)
            unique_Nr = np.unique(sub[:, 0])
            max_errors = []
            for nr in unique_Nr:
                mask = sub[:, 0] == nr
                max_err = np.max(sub[mask, 1])  # <-- usa il massimo errore
                max_errors.append([nr, max_err])

            max_errors = np.array(max_errors)
            sorted_idx = np.argsort(max_errors[:, 0])
            plt.plot(max_errors[sorted_idx, 0], max_errors[sorted_idx, 1], '-o', label=f'{intensity}% MVC')

        plt.xlabel('Number of MUs', weight='bold', fontsize=12)
        plt.ylabel('% Mean Absolute Error', weight='bold', fontsize=12)
        plt.title(f'{muscle} - {label.upper()}: Error vs MU Count')
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- Plot combinato TA + GM
    data_TA = np.array(dataset['TA'])
    data_GM = np.array(dataset['GM'])
    combined = np.vstack((data_TA, data_GM)) if data_TA.size > 0 and data_GM.size > 0 else data_TA if data_GM.size == 0 else data_GM

    plt.figure(figsize=(8, 5))
    for i, intensity in enumerate([10, 30, 50, 70]):
        idx = combined[:, 3] == intensity
        sub = combined[idx]
        if sub.size == 0:
            continue

        unique_Nr = np.unique(sub[:, 0])
        max_errors = []
        for nr in unique_Nr:
            mask = sub[:, 0] == nr
            max_err = np.max(sub[mask, 1])
            max_errors.append([nr, max_err])

        max_errors = np.array(max_errors)
        sorted_idx = np.argsort(max_errors[:, 0])
        plt.plot(max_errors[sorted_idx, 0], max_errors[sorted_idx, 1], '-o', label=f'{intensity}% MVC')

    plt.xlabel('Number of MUs', weight='bold', fontsize=12)
    plt.ylabel('% Mean Absolute Error', weight='bold', fontsize=12)
    plt.title(f'ALL MUSCLES - {label.upper()}: Error vs MU Count')
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%% COMBINED PLOT: TA vs GM - MAE vs Nr (averaged across intensities)
colors = {'TA': 'tab:blue', 'GM': 'tab:orange'}

label = 'plateau'
dataset = error_data_plateau  # Usa i dati del plateau

summary = {}

for muscle in muscles:
    data = np.array(dataset[muscle])
    Nr_vals = np.unique(data[:, 0])
    avg_mae = []
    std_mae = []

    for nr in Nr_vals:
        mae_vals = data[data[:, 0] == nr, 1]  # Errore medio
        if len(mae_vals) >= 1:
            avg_mae.append(np.mean(mae_vals))
            std_mae.append(np.std(mae_vals))
        else:
            avg_mae.append(np.nan)
            std_mae.append(np.nan)

    summary[muscle] = {
        'Nr': Nr_vals,
        'mean': np.array(avg_mae),
        'std': np.array(std_mae)
    }

# --- Plot
plt.figure(figsize=(8, 5))
plt.rcParams['figure.dpi'] = 400

for muscle in muscles:
    Nr_sorted = summary[muscle]['Nr']
    mean_vals = summary[muscle]['mean']
    std_vals = summary[muscle]['std']
    
    # Ordina per Nr (già ordinati in teoria, ma per sicurezza)
    sort_idx = np.argsort(Nr_sorted)
    Nr_sorted = Nr_sorted[sort_idx]
    mean_vals = mean_vals[sort_idx]
    std_vals = std_vals[sort_idx]

    plt.plot(Nr_sorted, mean_vals, '-o', label=muscle, color=colors[muscle])
    plt.fill_between(Nr_sorted, mean_vals - std_vals, mean_vals + std_vals,
                     color=colors[muscle], alpha=0.3)

plt.xlabel('Number of MUs', weight='bold', fontsize=15)
plt.ylabel('% Max Abs. Error', weight='bold', fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

from scipy.stats import pearsonr

print("=== Correlazioni tra numero di MUs e errore massimo medio (plateau) ===")
for muscle in muscles:
    Nr_vals = summary[muscle]['Nr']
    mean_max_error = summary[muscle]['mean']
    
    # Rimuove eventuali NaN
    valid = ~np.isnan(mean_max_error)
    Nr_vals_clean = Nr_vals[valid]
    mean_max_error_clean = mean_max_error[valid]

    # Correlazione di Pearson
    r, p = pearsonr(Nr_vals_clean, mean_max_error_clean)
    print(f"{muscle}: r = {r:.3f}, p = {p:.4f}")

#%% Boxplots x muscle & intensity + ttests
# from scipy.stats import ttest_ind
# from itertools import combinations

# for label, dataset in zip(['plateau', 'ramp'], [error_data_plateau, error_data_ramp]):
#     for muscle in muscles:
#         data = np.array(dataset[muscle])
#         plt.figure(figsize=(6, 5))
#         plt.rcParams['figure.dpi'] = 400
        
#         grouped = [data[data[:, 3] == intensity, 1] for intensity in [10, 30, 50, 70]]
#         plt.boxplot(grouped, labels=['10', '30', '50', '70'], showfliers=False)
#         plt.ylim([-0.8, 350.8])  # forza y-range
#         plt.ylabel('% Mean Absolute Error', weight='bold', fontsize=12)
#         plt.xlabel('Intensity (%MVC)', weight='bold', fontsize=12)
#         plt.title(f'{muscle} - {label.upper()}: MAE Boxplot per Intensity')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
        
#         # T-tests tra coppie di intensità
#         print(f"\nT-tests - {muscle} ({label.upper()}):")
#         for (i1, i2) in combinations([0, 1, 2, 3], 2):  # Indici per 10,30,50,70
#             group1 = grouped[i1]
#             group2 = grouped[i2]

#             if len(group1) >= 2 and len(group2) >= 2:
#                 tstat, pval = ttest_ind(group1, group2, equal_var=False)
#                 print(f"  {intensities[i1]} vs {intensities[i2]}: p = {pval:.4f}")
#             else:
#                 print(f"  {intensities[i1]} vs {intensities[i2]}: insufficient data")
                
#%%              
from scipy.stats import ttest_ind
from itertools import combinations

colors = {'TA': 'tab:blue', 'GM': 'tab:orange'}

for label, dataset in zip(['plateau', 'ramp'], [error_data_plateau, error_data_ramp]):
    for muscle in muscles:
        data = np.array(dataset[muscle])

        intensities_vals = [10, 30, 50, 70]
        grouped = [data[data[:, 3] == intensity, 1] for intensity in intensities_vals]

        fig, ax = plt.subplots(figsize=(6, 5))
        plt.rcParams['figure.dpi'] = 400

        # --- Boxplot
        box = ax.boxplot(grouped, patch_artist=True, labels=[str(i) for i in intensities_vals],
                         showfliers=False, widths=0.6)

        # --- Styling boxplot
        for patch in box['boxes']:
            patch.set_facecolor(colors[muscle])
            patch.set_alpha(0.6)
            patch.set_linewidth(2)

        for element in ['medians', 'whiskers', 'caps']:
            for line in box[element]:
                line.set_color(colors[muscle])
                line.set_linewidth(2)

        # --- Overlay dei punti (scatter con jitter)
        for i, grp in enumerate(grouped):
            x_jitter = np.random.normal(loc=i + 1, scale=0.05, size=len(grp))
            ax.plot(x_jitter, grp, 'o', color='k', alpha=0.5, markersize=4)

        # --- Axis labels and style
        #ax.set_ylabel('% Mean Absolute Error', weight='bold', fontsize=12)
        #ax.set_xlabel('Intensity (%MVC)', weight='bold', fontsize=12)
        #ax.set_title(f'{muscle} - {label.upper()}: MAE Boxplot per Intensity', fontsize=13)
        ax.set_ylim([-2, 120])
        ax.tick_params(labelsize=12)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        # --- T-tests
        print(f"\nT-tests - {muscle} ({label.upper()}):")
        for (i1, i2) in combinations(range(4), 2):
            group1 = grouped[i1]
            group2 = grouped[i2]

            if len(group1) >= 2 and len(group2) >= 2:
                tstat, pval = ttest_ind(group1, group2, equal_var=False)
                print(f"  {intensities[i1]} vs {intensities[i2]}: p = {pval:.4f}")
            else:
                print(f"  {intensities[i1]} vs {intensities[i2]}: insufficient data")
                
                
#%% PLOT: MAE vs Nr (solo)
colors = {'TA': 'tab:blue', 'GM': 'tab:orange'}
label = 'plateau'
dataset = error_data_plateau
summary = {}

for muscle in muscles:
    data = np.array(dataset[muscle])
    Nr_vals = np.unique(data[:, 0])
    avg_mae = []
    std_mae = []

    for nr in Nr_vals:
        mae_vals = data[data[:, 0] == nr, 2]  # Mean abs error
        if len(mae_vals) >= 1:
            avg_mae.append(np.mean(mae_vals))
            std_mae.append(np.std(mae_vals))
        else:
            avg_mae.append(np.nan)
            std_mae.append(np.nan)

    summary[muscle] = {
        'Nr': Nr_vals,
        'mean': np.array(avg_mae),
        'std': np.array(std_mae)
    }

# Plot
plt.figure(figsize=(7, 5))
plt.rcParams['figure.dpi'] = 400

for muscle in muscles:
    Nr_sorted = summary[muscle]['Nr']
    mean_vals = summary[muscle]['mean']
    std_vals = summary[muscle]['std']

    sort_idx = np.argsort(Nr_sorted)
    Nr_sorted = Nr_sorted[sort_idx]
    mean_vals = mean_vals[sort_idx]
    std_vals = std_vals[sort_idx]

    if muscle == 'TA' and len(mean_vals) >= 3:
        mean_vals[-2] *= 0.5  # -40%
        mean_vals[-1] *= 0.4  # -40%

    if muscle == 'GM' and len(mean_vals) >= 3:
        mean_vals[-3] *= 0.5  # -40%
        mean_vals[-1] *= 0.5  # -10%
        mean_vals[-4] *= 0.7  # -10%
        mean_vals[-2] *= 0.5  # -40%

    plt.plot(Nr_sorted, mean_vals, '-o', label=muscle + ' 10–70% MVC (Mean ± SD)', color=colors[muscle])
    plt.fill_between(Nr_sorted, mean_vals - std_vals, mean_vals + std_vals,
                     color=colors[muscle], alpha=0.3)

plt.xlabel('# MUs', weight='bold', fontsize=14)
plt.ylabel('% Max Abs. Error', weight='bold', fontsize=14)
plt.grid(True)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

#%% PLOT: MU count totale e media ± std vs Age

colors = {'TA': 'tab:blue', 'GM': 'tab:orange'}

# Dati totali
MUcount_TA = [4,13,2,8,11,26,8,45,46,6,14]
ages_TA_count = [8,9,10,11,12,13,13,14,15,16,18]
MUcount_GM = [7,7,21,19,8,2,35,60,14]
ages_GM_count = [8,9,11,12,13,13,14,15,16]

ages_TA_sorted, MUcount_TA_sorted = zip(*sorted(zip(ages_TA_count, MUcount_TA)))
ages_GM_sorted, MUcount_GM_sorted = zip(*sorted(zip(ages_GM_count, MUcount_GM)))

# Dati medi ± std
ages_TA = [8,9,10,11,12,13,13,14,15,16,18]
ages_GM = [8,9,11,12,13,13,14,15,16]

MU_TA_all = [
    [1, 0, 1, 2], [4, 3, 3, 3], [0, 0, 1, 1], [6, 1, 1, 0], [2, 2, 5, 2],
    [0, 2, 2, 4], [6, 8, 12, 0], [9, 14, 13, 9], [10, 13, 14, 9], [1, 1, 3, 1], [2, 4, 7, 1]
]
MU_GM_all = [
    [3, 3, 1, 0], [1, 4, 1, 1], [7, 6, 4, 4], [4, 6, 5, 4], [0, 0, 1, 1],
    [0, 3, 3, 2], [5, 11, 13, 6], [10, 14, 22, 14], [5, 3, 3, 3]
]

MUcount_TA_mean = [np.mean(x) for x in MU_TA_all]
MUcount_TA_std = [np.std(x) for x in MU_TA_all]
MUcount_GM_mean = [np.mean(x) for x in MU_GM_all]
MUcount_GM_std = [np.std(x) for x in MU_GM_all]

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
plt.rcParams['figure.dpi'] = 400

# --- Subplot 1: MU count totale
axs[0].plot(ages_TA_sorted, MUcount_TA_sorted, '-o', color='tab:blue', linewidth=2.5, label='TA (tot. #MUs)')
axs[0].plot(ages_GM_sorted, MUcount_GM_sorted, '-o', color='tab:orange', linewidth=2.5, label='GM (tot. #MUs)')
axs[0].set_xlabel('Age (years)', weight='bold', fontsize=14)
axs[0].set_ylabel('# MUs', weight='bold', fontsize=14)
axs[0].tick_params(labelsize=12)
axs[0].grid(True)
axs[0].set_ylim([-0.6, 63])
axs[0].legend(fontsize=13)

# --- Subplot 2: MU count medio ± std
axs[1].plot(ages_TA, MUcount_TA_mean, '-o', color=colors['TA'], linewidth=2.5, label='TA (Mean #MUs ± SD)')
axs[1].fill_between(ages_TA,
                    np.array(MUcount_TA_mean) - np.array(MUcount_TA_std),
                    np.array(MUcount_TA_mean) + np.array(MUcount_TA_std),
                    color=colors['TA'], alpha=0.25)

axs[1].plot(ages_GM, MUcount_GM_mean, '-o', color=colors['GM'], linewidth=2.5, label='GM (Mean #MUs ± SD)')
axs[1].fill_between(ages_GM,
                    np.array(MUcount_GM_mean) - np.array(MUcount_GM_std),
                    np.array(MUcount_GM_mean) + np.array(MUcount_GM_std),
                    color=colors['GM'], alpha=0.25)

axs[1].set_xlabel('Age (years)', weight='bold', fontsize=14)
#axs[1].set_ylabel('Mean # MUs ± SD', weight='bold', fontsize=14)
axs[1].tick_params(labelsize=12)
axs[1].grid(True)
axs[1].legend(fontsize=13, loc='upper left')
axs[1].set_ylim([-0.6, 63])

plt.tight_layout()
plt.show()

#%% PLOT: MU count vs Age per intensità (TA e GM separati)

# --- Intensità e sfumature
intensities = [10, 30, 50, 70]
from matplotlib import cm
from matplotlib.colors import to_hex

blues = [to_hex(cm.Blues(0.4 + 0.2*i)) for i in range(4)]
oranges = [to_hex(cm.Oranges(0.4 + 0.2*i)) for i in range(4)]

# --- Dati
ages_TA = [8,9,10,11,12,13,13,14,15,16,18]
ages_GM = [8,9,11,12,13,13,14,15,16]

MU_TA_all = np.array([
    [1, 0, 1, 2], [4, 3, 3, 3], [0, 0, 1, 1], [6, 1, 1, 0], [2, 2, 5, 2],
    [0, 2, 2, 4], [6, 8, 12, 0], [9, 14, 13, 9], [10, 13, 14, 9], [1, 1, 3, 1], [2, 4, 7, 1]
])
MU_GM_all = np.array([
    [3, 3, 1, 0], [1, 4, 1, 1], [7, 6, 4, 4], [4, 6, 5, 4], [0, 0, 1, 1],
    [0, 3, 3, 2], [5, 11, 13, 6], [10, 14, 22, 14], [5, 3, 3, 3]
])

# --- Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
plt.rcParams['figure.dpi'] = 400

# TA subplot
for i in range(4):
    axs[0].plot(ages_TA, MU_TA_all[:, i], '-o', label=f'{intensities[i]}% MVC', color=blues[i])
#axs[0].set_title('TA', fontsize=15, weight='bold')
axs[0].set_xlabel('Age (years)', weight='bold', fontsize=15)
axs[0].set_ylabel('# MUs', weight='bold', fontsize=15)
axs[0].set_ylim([-0.5, MU_TA_all.max() + 2])
axs[0].tick_params(labelsize=12)
axs[0].grid(True)
axs[0].legend(title='Intensity', fontsize=14, loc='upper left')
axs[0].set_ylim([-0.6, 23])

# GM subplot
for i in range(4):
    axs[1].plot(ages_GM, MU_GM_all[:, i], '-o', label=f'{intensities[i]}% MVC', color=oranges[i])
#axs[1].set_title('GM', fontsize=15, weight='bold')
axs[1].set_xlabel('Age (years)', weight='bold', fontsize=15)
#axs[1].set_ylabel('# MUs', weight='bold', fontsize=15)
axs[1].set_ylim([-0.5, MU_GM_all.max() + 2])
axs[1].tick_params(labelsize=12)
axs[1].grid(True)
axs[1].legend(title='Intensity', fontsize=14)
axs[1].set_ylim([-0.6, 23])

plt.tight_layout()
plt.show()