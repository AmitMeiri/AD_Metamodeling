import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.linalg import expm
from bayesian_metamodeling.surrogates.backends import load_backend_model

def get_latest_model_path():
    base_dir = Path('tmp/surrogate_artifacts')
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError("No trained surrogate model found in tmp/surrogate_artifacts")
    # Sort by modification time of backend_payload.json
    subdirs.sort(key=lambda d: (d / 'backend_payload.json').stat().st_mtime if (d / 'backend_payload.json').exists() else 0)
    latest_dir = subdirs[-1]
    payload_path = latest_dir / 'backend_payload.json'
    print(f"Discovered latest model at: {payload_path}")
    return payload_path

def get_stage_probs_with_ci(stages, n_bootstrap=150):
    probs = []
    for val in [0, 1, 2]:
        probs.append(np.mean(stages == val))
    
    # Bootstrap to get 95% confidence intervals (uncertainty whiskers)
    boot_probs = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(stages, size=len(stages), replace=True)
        boot_probs.append([np.mean(boot_sample == val) for val in [0, 1, 2]])
    boot_probs = np.array(boot_probs)
    ci_lower = np.percentile(boot_probs, 2.5, axis=0)
    ci_upper = np.percentile(boot_probs, 97.5, axis=0)
    
    yerr = np.zeros((2, 3))
    for val_idx in range(3):
        yerr[0, val_idx] = max(0.0, probs[val_idx] - ci_lower[val_idx])
        yerr[1, val_idx] = max(0.0, ci_upper[val_idx] - probs[val_idx])
    return np.array(probs), yerr

def main():
    np.random.seed(42)
    
    # 1. Load training data to get winsorizing bounds
    print("Loading training data to synchronize bounds...")
    with open('ad_ode_training_data.json', 'r') as f:
        training_data = json.load(f)
    df_train = pd.DataFrame(training_data)
    ab_min, ab_max = df_train['amyloid_2yr'].min(), df_train['amyloid_2yr'].max()
    tau_min, tau_max = df_train['tau_2yr'].min(), df_train['tau_2yr'].max()
    mem_min, mem_max = df_train['memory_result_yr5'].min(), df_train['memory_result_yr5'].max()
    
    # 2. Load the raw ODE posterior traces (MCMC)
    print("Loading raw MCMC posterior samples...")
    with open('../ad_ode_demo_posterior.pkl', 'rb') as f:
        posterior = pickle.load(f)
    v_samples = posterior['v']          # (4000, 5, 5)
    wAge_samples = posterior['wAge']    # (4000, 5, 5)
    wAPOE_samples = posterior['wAPOE']  # (4000, 5, 5)
    v0_samples = posterior['v0']        # (4000, 5)
    c0_samples = posterior['c0']        # (4000, 10, 5)
    
    N_draws = v_samples.shape[0]        # 4000
    
    # 3. Load latest trained surrogate
    payload_path = get_latest_model_path()
    model = load_backend_model('sbi_npe', payload_path)
    
    # 4. Define patient archetypes
    archetypes = [
        {'id': 'A', 'age': 65.0, 'apoe4': 0, 'title': 'Subject A: Age 65, APOE4-'},
        {'id': 'B', 'age': 65.0, 'apoe4': 1, 'title': 'Subject B: Age 65, APOE4+'},
        {'id': 'C', 'age': 85.0, 'apoe4': 0, 'title': 'Subject C: Age 85, APOE4-'},
        {'id': 'D', 'age': 85.0, 'apoe4': 1, 'title': 'Subject D: Age 85, APOE4+'}
    ]
    
    subj_idx = 0 # Baseline subject to evaluate starting state c0
    
    # Setup premium plotting style
    sns.set_theme(style='ticks')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['text.color'] = '#2c3e50'
    plt.rcParams['axes.labelcolor'] = '#2c3e50'
    plt.rcParams['xtick.color'] = '#2c3e50'
    plt.rcParams['ytick.color'] = '#2c3e50'
    plt.rcParams['axes.titlecolor'] = '#2c3e50'
    
    # Create two separate figures for plotting
    fig_dyn, axes_dyn = plt.subplots(4, 3, figsize=(16, 14), gridspec_kw={'hspace': 0.4, 'wspace': 0.25})
    fig_cli, axes_cli = plt.subplots(4, 2, figsize=(11, 14), gridspec_kw={'hspace': 0.4, 'wspace': 0.25})
    
    for row_idx, arch in enumerate(archetypes):
        age = arch['age']
        apoe4 = arch['apoe4']
        print(f"Simulating archetype: {arch['title']}...")
        
        # True MCMC simulation
        ab_0yr_true = []
        tau_0yr_true = []
        ab_2yr_true = []
        tau_2yr_true = []
        ab_5yr_true = []
        tau_5yr_true = []
        mem_5yr_true = []
        tau_self_true = []
        ab_self_true = []
        ab_drive_tau_true = []
        
        for s in range(N_draws):
            v_mat = v_samples[s]
            wAge_mat = wAge_samples[s]
            wAPOE_mat = wAPOE_samples[s]
            v0_vec = v0_samples[s]
            c0_vec = c0_samples[s, subj_idx]
            
            # Compute velocity matrix A
            A = v_mat + age * wAge_mat + apoe4 * wAPOE_mat
            delta = c0_vec - v0_vec
            
            x0 = c0_vec
            x2 = expm(2.0 * A) @ delta + v0_vec
            x5 = expm(5.0 * A) @ delta + v0_vec
            
            ab_0yr_true.append(x0[0])
            tau_0yr_true.append(x0[1])
            ab_2yr_true.append(x2[0])
            tau_2yr_true.append(x2[1])
            ab_5yr_true.append(x5[0])
            tau_5yr_true.append(x5[1])
            mem_5yr_true.append(x5[3])
            
            tau_self_true.append(v_mat[1, 1])
            ab_self_true.append(v_mat[0, 0])
            ab_drive_tau_true.append(v_mat[1, 0])
            
        # Clip according to training dataset winsorization bounds
        ab_5yr_true = np.clip(ab_5yr_true, ab_min, ab_max)
        tau_5yr_true = np.clip(tau_5yr_true, tau_min, tau_max)
        mem_5yr_true = np.clip(mem_5yr_true, mem_min, mem_max)
        tau_self_true = np.array(tau_self_true)
        ab_self_true = np.array(ab_self_true)
        ab_drive_tau_true = np.array(ab_drive_tau_true)
        
        # True Clinical Stage
        clinical_stage_true = []
        for mem in mem_5yr_true:
            if mem < 0.5:
                stage = 0
            elif mem < 1.5:
                stage = 1
            else:
                stage = 2
            clinical_stage_true.append(stage)
        clinical_stage_true = np.array(clinical_stage_true)
        
        # Compute mean inputs for surrogate conditioning
        mean_ab_0yr = np.mean(ab_0yr_true)
        mean_tau_0yr = np.mean(tau_0yr_true)
        mean_ab_2yr = np.mean(ab_2yr_true)
        mean_tau_2yr = np.mean(tau_2yr_true)
        
        # Query Surrogate (SBI NPE)
        x_input = {
            'age_baseline': np.array([age]),
            'apoe4_status': np.array([apoe4]),
            'amyloid_baseline': np.array([mean_ab_0yr]),
            'tau_baseline': np.array([mean_tau_0yr]),
            'amyloid_2yr': np.array([mean_ab_2yr]),
            'tau_2yr': np.array([mean_tau_2yr])
        }
        
        # Draw 4,000 samples from trained NPE surrogate flow
        samples_surr = model.sample(x_input, n=N_draws, seed=42)[0]
        tau_self_surr = samples_surr[:, 0]
        ab_self_surr = samples_surr[:, 1]
        ab_drive_tau_surr = samples_surr[:, 2]
        mem_5yr_surr = samples_surr[:, 5]
        clinical_stage_surr = np.clip(np.round(samples_surr[:, 6]), 0, 2)
        
        # Calculate probabilities and CIs for clinical stages
        probs_true, yerr_true = get_stage_probs_with_ci(clinical_stage_true)
        probs_surr, yerr_surr = get_stage_probs_with_ci(clinical_stage_surr)
        
        # =====================================================================
        # FIGURE 1: SELF-DYNAMICS VALIDATION (Tau and Amyloid)
        # =====================================================================
        
        # Column 0: Tau Self-Dynamics density
        ax = axes_dyn[row_idx, 0]
        sns.kdeplot(data=tau_self_true, color='#7f8c8d', fill=True, alpha=0.3, label='ODE posterior (MCMC)', ax=ax)
        sns.kdeplot(data=tau_self_surr, color='#007aff', linewidth=2.5, label='SBI surrogate (NPE)', ax=ax)
        ax.set_title(f"Tau Self-Dynamics ($v_{{11}}$)\n{arch['title']}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        if row_idx == 0:
            ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=9, loc='upper left')
        sns.despine(ax=ax, trim=True)
        
        # Column 1: Amyloid Self-Dynamics density
        ax = axes_dyn[row_idx, 1]
        sns.kdeplot(data=ab_self_true, color='#7f8c8d', fill=True, alpha=0.3, label='ODE posterior (MCMC)', ax=ax)
        sns.kdeplot(data=ab_self_surr, color='#007aff', linewidth=2.5, label='SBI surrogate (NPE)', ax=ax)
        ax.set_title(f"Amyloid Self-Dynamics ($v_{{00}}$)\n{arch['title']}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        sns.despine(ax=ax, trim=True)
        
        # Column 2: Amyloid Drive Tau density
        ax = axes_dyn[row_idx, 2]
        sns.kdeplot(data=ab_drive_tau_true, color='#7f8c8d', fill=True, alpha=0.3, label='ODE posterior (MCMC)', ax=ax)
        sns.kdeplot(data=ab_drive_tau_surr, color='#007aff', linewidth=2.5, label='SBI surrogate (NPE)', ax=ax)
        ax.set_title(f"Amyloid $\\rightarrow$ Tau Drive ($v_{{10}}$)\n{arch['title']}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        sns.despine(ax=ax, trim=True)
        
        # =====================================================================
        # FIGURE 2: CLINICAL VALIDATION (Memory and Clinical Staging)
        # =====================================================================
        
        # Column 0: Predicted Memory @ 5 yr density
        ax = axes_cli[row_idx, 0]
        sns.kdeplot(data=mem_5yr_true, color='#7f8c8d', fill=True, alpha=0.3, label='ODE posterior (MCMC)', ax=ax)
        sns.kdeplot(data=mem_5yr_surr, color='#007aff', linewidth=2.5, label='SBI surrogate (NPE)', ax=ax)
        ax.set_title(f"Predicted Memory @ 5 yr\n{arch['title']}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Value (Worse $\\rightarrow$)')
        ax.set_ylabel('Density')
        if row_idx == 0:
            ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=9, loc='upper left')
        sns.despine(ax=ax, trim=True)
        
        # Column 1: Clinical Stage probabilities bar chart
        ax = axes_cli[row_idx, 1]
        stages = ['CN', 'MCI', 'Dementia']
        x_indices = np.arange(len(stages))
        width = 0.35
        
        ax.bar(x_indices - width/2, probs_true, width, yerr=yerr_true, label='MCMC', color='#7f8c8d', alpha=0.7, capsize=4, ecolor='#2c3e50')
        ax.bar(x_indices + width/2, probs_surr, width, yerr=yerr_surr, label='NPE', color='#007aff', alpha=0.7, capsize=4, ecolor='#2c3e50')
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(stages, fontsize=10)
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Clinical Stage Staging\n{arch['title']}", fontsize=11, fontweight='bold', pad=8)
        if row_idx == 0:
            ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=9, loc='upper right')
        sns.despine(ax=ax, trim=True)
        
    os.makedirs('surrogate_ode_plots', exist_ok=True)
    
    # Save Figure 1
    fig_dyn.suptitle("Individual Validation: Biological Self-Dynamics\nODE Posterior MCMC vs. Learned NPE Surrogate", fontsize=15, fontweight='bold', y=0.98)
    dyn_filename = 'surrogate_ode_plots/panel_d_ode_validation_dynamics.png'
    fig_dyn.savefig(dyn_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_dyn)
    print(f"Self-Dynamics validation saved successfully to {dyn_filename} !")
    
    # Save Figure 2
    fig_cli.suptitle("Individual Validation: Clinical Staging & Memory Outcomes\nODE Posterior MCMC vs. Learned NPE Surrogate", fontsize=15, fontweight='bold', y=0.98)
    cli_filename = 'surrogate_ode_plots/panel_d_ode_validation_clinical.png'
    fig_cli.savefig(cli_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_cli)
    print(f"Clinical outcomes validation saved successfully to {cli_filename} !")

if __name__ == '__main__':
    main()
