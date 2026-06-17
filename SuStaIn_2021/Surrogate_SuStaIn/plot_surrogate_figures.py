import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.patches as patches

# Ensure the bayesian_metamodeling package can be loaded
from bayesian_metamodeling.surrogates.backends import load_backend_model

# Define script base directory for robust relative paths
SCRIPT_DIR = Path(__file__).resolve().parent

def create_directory():
    os.makedirs(SCRIPT_DIR / 'surrogate_sustain_plots', exist_ok=True)

def load_data():
    data_path = SCRIPT_DIR / 'sustain_training_data.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    df_rows = []
    for row in data:
        merged = {**row['inputs'], **row['outputs']}
        df_rows.append(merged)
        
    df = pd.DataFrame(df_rows)
    return df

def get_latest_model_path():
    base_dir = SCRIPT_DIR / 'tmp' / 'surrogate_artifacts'
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No trained surrogate model found in {base_dir}")
    subdirs.sort(key=lambda d: (d / 'backend_payload.json').stat().st_mtime if (d / 'backend_payload.json').exists() else 0)
    latest_dir = subdirs[-1]
    payload_path = latest_dir / 'backend_payload.json'
    print(f"Discovered latest model at: {payload_path}")
    return payload_path

def generate_surrogate_data(df, model):
    # Extract inputs dynamically
    input_cols = [c for c in df.columns if 'region' in c]
    inputs = {c: df[c].values for c in input_cols}
    
    # Draw samples from the learned surrogate
    # model.sample returns shape [n_samples, batch_size, n_outputs]
    # Here n=1, so we reshape to [batch_size, n_outputs]
    samples = model.sample(inputs, n=1, seed=42)
    output_cols = [c for c in df.columns if 'prob_subtype' in c] + ['expected_stage']
    samples = samples.reshape(len(df), len(output_cols))
    
    surr_df = pd.DataFrame(samples, columns=output_cols)
    return surr_df

def set_premium_style():
    sns.set_theme(style='ticks')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['text.color'] = '#2c3e50'
    plt.rcParams['axes.labelcolor'] = '#2c3e50'
    plt.rcParams['xtick.color'] = '#2c3e50'
    plt.rcParams['ytick.color'] = '#2c3e50'
    plt.rcParams['axes.titlecolor'] = '#2c3e50'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_panel_a(df_orig, df_surr):
    set_premium_style()
    output_cols = [c for c in df_orig.columns if 'prob_subtype' in c] + ['expected_stage']
    num_plots = len(output_cols)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1: axes = [axes]
    
    for i, (ax, var) in enumerate(zip(axes, output_cols)):
        orig_data = df_orig[var]
        surr_data = df_surr[var]
        
        sns.histplot(orig_data, color='#7f8c8d', stat='density', alpha=0.4, label='SuStaIn (MCMC)', ax=ax, bins=35, edgecolor='none')
        sns.histplot(surr_data, color='#007aff', stat='density', element='step', fill=False, label='Surrogate (NPE)', ax=ax, bins=35, linewidth=2)
        
        title = var.replace('_', ' ').title()
        ax.set_title(title, fontweight='bold', pad=12)
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        
        # Subtype probabilities are bounded [0, 1]
        if 'prob' in var:
            ax.set_xlim(-0.05, 1.05)
            
        sns.despine(ax=ax, trim=True)
        
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=10)
    plt.suptitle("Panel A: Marginal Densities (Fidelity Evaluation)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'surrogate_sustain_plots' / 'panel_a_marginal.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_panel_b(df_orig, df_surr):
    set_premium_style()
    subtype_cols = [c for c in df_orig.columns if 'prob_subtype' in c]
    num_plots = len(subtype_cols)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1: axes = [axes]
    
    x_var = 'expected_stage'
    
    for i, (ax, y_var) in enumerate(zip(axes, subtype_cols)):
        # Hexbin for MCMC
        hb = ax.hexbin(df_orig[x_var], df_orig[y_var], gridsize=30, cmap='Greys', mincnt=1, bins='log', edgecolors='none', alpha=0.85)
        
        # Surrogate KDE contours
        sns.kdeplot(x=df_surr[x_var], y=df_surr[y_var], ax=ax, levels=[0.5, 0.9], colors=['#007aff'], linewidths=[1.5, 2.5], alpha=0.9)
        
        # Correlations
        r_orig, _ = pearsonr(df_orig[x_var], df_orig[y_var])
        r_surr, _ = pearsonr(df_surr[x_var], df_surr[y_var])
        
        ax.text(0.05, 0.95, f"MCMC r = {r_orig:.3f}\nSurrogate r = {r_surr:.3f}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=10, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlabel('Expected Stage', fontsize=11)
        ax.set_ylabel(y_var.replace('_', ' ').title(), fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        sns.despine(ax=ax, trim=True)

    plt.suptitle("Panel B: Classification Confidence (Pairwise Joint Distributions)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'surrogate_sustain_plots' / 'panel_b_pairwise.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_panel_c(model, df_orig):
    set_premium_style()
    # Panel C: Biomarker Sensitivity Analysis (Conditional Inference)
    # We will sweep the Z-score of specific regions from 0 to 5 while holding others at baseline (0.5),
    # to see how the surrogate shifts its subtype classification probabilities.
    
    z_sweep = np.linspace(0, 5, 20)
    baseline_z = 0.5
    
    # We sweep Region 0 (Entorhinal - typical limbic origin) and Region 6 (Precuneus - atypical neocortical origin)
    regions_to_sweep = [0, 6]
    region_names = ['Entorhinal (Region 0)', 'Precuneus (Region 6)']
    
    subtype_cols = [c for c in df_orig.columns if 'prob_subtype' in c]
    num_subtypes = len(subtype_cols)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, sweep_idx, region_name in zip(axes, regions_to_sweep, region_names):
        mean_probs = {s_col: [] for s_col in subtype_cols}
        
        for z in z_sweep:
            # Create a dictionary of inputs holding all at baseline, except the sweep target
            x_input = {}
            for i in range(7):
                if i == sweep_idx:
                    x_input[f'region_{i}_zscore'] = np.array([z])
                else:
                    x_input[f'region_{i}_zscore'] = np.array([baseline_z])
            
            # Draw samples
            samples = model.sample(x_input, n=500, seed=42)[0]
            
            # Subtype probabilities are the first `num_subtypes` columns
            for s_idx, s_col in enumerate(subtype_cols):
                mean_probs[s_col].append(np.mean(samples[:, s_idx]))
                
        # Plot the sweep results
        colors = ['#ff9500', '#007aff', '#34c759', '#ff3b30']
        for s_idx, s_col in enumerate(subtype_cols):
            ax.plot(z_sweep, mean_probs[s_col], label=s_col.replace('_', ' ').title(), 
                    linewidth=3, color=colors[s_idx % len(colors)])
            
        ax.set_xlabel(f'{region_name} Z-Score (Severity)', fontsize=12)
        ax.set_ylabel('Predicted Subtype Probability', fontsize=12)
        ax.set_title(f'Sensitivity to {region_name}', fontweight='bold', pad=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
        sns.despine(ax=ax, trim=True)
        
    plt.suptitle("Panel C: Biomarker Sensitivity Analysis (NPE Conditional Inference)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'surrogate_sustain_plots' / 'panel_c_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_directory()
    df_orig = load_data()
    payload_path = get_latest_model_path()
    model = load_backend_model('sbi_npe', payload_path)
    df_surr = generate_surrogate_data(df_orig, model)
    
    print("Plotting Panel A: Marginals...")
    plot_panel_a(df_orig, df_surr)
    
    print("Plotting Panel B: Pairwise...")
    plot_panel_b(df_orig, df_surr)
    
    print("Plotting Panel C: Sensitivity Analysis...")
    plot_panel_c(model, df_orig)
    
    print("Successfully generated figures in surrogate_sustain_plots/ !")
