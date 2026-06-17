import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure the bayesian_metamodeling package can be loaded
from bayesian_metamodeling.surrogates.backends import load_backend_model

# Define script base directory for robust relative paths
SCRIPT_DIR = Path(__file__).resolve().parent

def set_premium_style():
    sns.set_theme(style='ticks')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['text.color'] = '#2c3e50'
    plt.rcParams['axes.labelcolor'] = '#2c3e50'
    plt.rcParams['xtick.color'] = '#2c3e50'
    plt.rcParams['ytick.color'] = '#2c3e50'
    plt.rcParams['axes.titlecolor'] = '#2c3e50'
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def load_data():
    data_path = SCRIPT_DIR / 'sustain_training_data.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def get_latest_model_path():
    base_dir = SCRIPT_DIR / 'tmp' / 'surrogate_artifacts'
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No trained surrogate model found in {base_dir}")
    subdirs.sort(key=lambda d: (d / 'backend_payload.json').stat().st_mtime if (d / 'backend_payload.json').exists() else 0)
    latest_dir = subdirs[-1]
    payload_path = latest_dir / 'backend_payload.json'
    return payload_path

def find_archetypes(data):
    # Find four archetypes:
    # 1. Early stage subject
    # 2. Strong Subtype 0 subject
    # 3. Strong Subtype 1 subject
    # 4. Strong Subtype 2 subject
    
    stages = [x['outputs']['expected_stage'] for x in data]
    sub0_probs = [x['outputs']['prob_subtype_0'] for x in data]
    sub1_probs = [x['outputs']['prob_subtype_1'] for x in data]
    sub2_probs = [x['outputs']['prob_subtype_2'] for x in data]
    
    idx_early = np.argmin(stages)
    idx_sub0 = np.argmax(sub0_probs)
    idx_sub1 = np.argmax(sub1_probs)
    idx_sub2 = np.argmax(sub2_probs)
    
    archetypes = {
        'Early Stage (Mild/Prodromal)': data[idx_early],
        'Subtype 0 (Classic Limbic)': data[idx_sub0],
        'Subtype 1 (Atypical Neocortical)': data[idx_sub1],
        'Subtype 2 (Cortical-Leaning)': data[idx_sub2]
    }
    
    return archetypes

def validate_subject(model, subject, n_samples=2000):
    inputs = {k: np.array([v]) for k, v in subject['inputs'].items()}
    
    # Draw samples from the learned surrogate posterior
    # returns shape [n_samples, batch_size, n_outputs] -> reshape to [n_samples, n_outputs]
    samples = model.sample(inputs, n=n_samples, seed=42)[0]
    
    # Outputs: [prob_subtype_0, prob_subtype_1, prob_subtype_2, expected_stage]
    results = {}
    for i, key in enumerate(['prob_subtype_0', 'prob_subtype_1', 'prob_subtype_2', 'expected_stage']):
        surr_samples = samples[:, i]
        results[key] = {
            'true': subject['outputs'][key],
            'pred_mean': float(np.mean(surr_samples)),
            'pred_lower': float(np.percentile(surr_samples, 2.5)),
            'pred_upper': float(np.percentile(surr_samples, 97.5)),
            'samples': surr_samples
        }
    return results

def main():
    set_premium_style()
    data = load_data()
    payload_path = get_latest_model_path()
    model = load_backend_model('sbi_npe', payload_path)
    
    print("Finding patient archetypes...")
    archetypes = find_archetypes(data)
    
    # Run validation for each archetype
    results = {}
    for name, subject in archetypes.items():
        print(f"Validating: {name}...")
        results[name] = validate_subject(model, subject)
        
    # Generate visualization
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), gridspec_kw={'width_ratios': [1.5, 1]})
    
    colors_true = '#7f8c8d'  # MCMC
    colors_pred = '#007aff'  # Surrogate
    
    for row_idx, (name, res) in enumerate(results.items()):
        # LEFT PANEL: Subtype Probabilities comparison
        ax_left = axes[row_idx, 0]
        
        subtypes = ['Subtype 0', 'Subtype 1', 'Subtype 2']
        keys = ['prob_subtype_0', 'prob_subtype_1', 'prob_subtype_2']
        
        true_vals = [res[k]['true'] for k in keys]
        pred_means = [res[k]['pred_mean'] for k in keys]
        pred_lowers = [res[k]['pred_lower'] for k in keys]
        pred_uppers = [res[k]['pred_upper'] for k in keys]
        
        x = np.arange(len(subtypes))
        width = 0.35
        
        # Plot bars
        ax_left.bar(x - width/2, true_vals, width, label='SuStaIn (MCMC)', color=colors_true, alpha=0.6, edgecolor='none')
        
        # Plot predicted means with error bars representing 95% Credible Interval
        yerr = [
            [pred_means[i] - pred_lowers[i] for i in range(3)],
            [pred_uppers[i] - pred_means[i] for i in range(3)]
        ]
        ax_left.bar(x + width/2, pred_means, width, label='Surrogate (NPE)', color=colors_pred, alpha=0.8, edgecolor='none')
        ax_left.errorbar(x + width/2, pred_means, yerr=yerr, fmt='none', ecolor='#1d1d1f', elinewidth=2, capsize=4)
        
        ax_left.set_title(f"{name}: Subtype Classification", fontweight='bold')
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(subtypes)
        ax_left.set_ylabel('Probability')
        ax_left.set_ylim(-0.05, 1.05)
        ax_left.grid(True, axis='y', linestyle='--', alpha=0.3)
        sns.despine(ax=ax_left, trim=True)
        
        if row_idx == 0:
            ax_left.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
            
        # RIGHT PANEL: Expected Stage posterior distribution comparison
        ax_right = axes[row_idx, 1]
        stage_res = res['expected_stage']
        
        # Plot surrogate posterior density for stage
        sns.kdeplot(stage_res['samples'], ax=ax_right, fill=True, color=colors_pred, alpha=0.4, label='Surrogate Posterior Density', linewidth=2)
        
        # Plot true expected stage as a vertical dashed line
        ax_right.axvline(stage_res['true'], color='#ff3b30', linestyle='--', linewidth=2.5, label=f"MCMC Expected Stage ({stage_res['true']:.2f})")
        
        # Plot surrogate predicted mean expected stage as a solid vertical line
        ax_right.axvline(stage_res['pred_mean'], color=colors_pred, linestyle='-', linewidth=2, label=f"Surrogate Mean ({stage_res['pred_mean']:.2f})")
        
        ax_right.set_title(f"{name}: Disease Stage Estimation", fontweight='bold')
        ax_right.set_xlabel('Expected Stage')
        ax_right.set_ylabel('Density')
        ax_right.grid(True, axis='x', linestyle='--', alpha=0.3)
        sns.despine(ax=ax_right, trim=True)
        
        if row_idx == 0:
            ax_right.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, loc='upper left', fontsize=9)
            
    plt.suptitle("Panel D: Individual-Level Patient Validation Archetypes", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    os.makedirs(SCRIPT_DIR / 'surrogate_sustain_plots', exist_ok=True)
    save_path = SCRIPT_DIR / 'surrogate_sustain_plots' / 'panel_d_test_subjects.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated patient-level validation figures at: {save_path}")

if __name__ == '__main__':
    main()
