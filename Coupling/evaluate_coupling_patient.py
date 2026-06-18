import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import io
from contextlib import redirect_stdout
import re
import argparse

# NOTE: The hardcoded `sys.path.insert(0, "C:/Project/...")` paths that were previously here 
# have been intentionally removed. This script now relies natively on the fact that the 
# `bayesian_metamodeling` framework is installed in your Python environment (via `pip install -e .`), 
# which is the standard Python way of doing things. This ensures portability across different machines.
from bayesian_metamodeling.tutorial import run_mm_cli
from bayesian_metamodeling.surrogates.backends import load_backend_model

def get_latest_model_path(base_dir):
    base = Path(base_dir)
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No trained surrogate model found in {base_dir}")
    subdirs.sort(key=lambda d: (d / 'backend_payload.json').stat().st_mtime if (d / 'backend_payload.json').exists() else 0)
    return subdirs[-1] / 'backend_payload.json'

ode_model = load_backend_model('sbi_npe', get_latest_model_path('C:/Project/AD_Metamodeling/ode_2023/Surrogate/tmp/surrogate_artifacts'))
sustain_model = load_backend_model('sbi_npe', get_latest_model_path('C:/Project/AD_Metamodeling/SuStaIn_2021/Surrogate_SuStaIn/tmp/surrogate_artifacts'))

def get_uncoupled_samples(patient, n_draws=500):
    ode_x = {
        'age_baseline': np.array([patient['age']]),
        'apoe4_status': np.array([patient['apoe4']]),
        'amyloid_baseline': np.array([patient['amyloid_baseline']]),
        'tau_baseline': np.array([patient['tau_baseline']]),
        'amyloid_2yr': np.array([patient['amyloid_2yr']]),
        'tau_2yr': np.array([patient['tau_2yr']])
    }
    ode_samples = ode_model.sample(ode_x, n=n_draws, seed=42)[0]
    
    sustain_x = {f'region_{i}_zscore': np.array([patient['zscores'][i]]) for i in range(7)}
    sustain_samples = sustain_model.sample(sustain_x, n=n_draws, seed=42)[0]
    
    return ode_samples, sustain_samples

def run_coupled_samples(patient, patient_id, spec_path, ode_s, sus_s, n_draws=250):
    out_dir = Path("C:/Project/AD_Metamodeling/Coupling")
    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    with open(spec_path, 'r') as f:
        spec = json.load(f)
    
    spec["priors"] = [
        { "variable": "age_baseline", "distribution": { "kind": "normal", "loc": patient['age'], "scale": 0.01 } },
        { "variable": "apoe4_status", "distribution": { "kind": "normal", "loc": patient['apoe4'], "scale": 0.01 } }
    ]
    spec['priors'].append({ "variable": "amyloid_baseline", "distribution": { "kind": "normal", "loc": patient['amyloid_baseline'], "scale": 0.01 } })
    spec['priors'].append({ "variable": "tau_baseline", "distribution": { "kind": "normal", "loc": patient['tau_baseline'], "scale": 0.01 } })
    spec['priors'].append({ "variable": "amyloid_2yr", "distribution": { "kind": "normal", "loc": patient['amyloid_2yr'], "scale": 0.01 } })
    spec['priors'].append({ "variable": "tau_2yr", "distribution": { "kind": "normal", "loc": patient['tau_2yr'], "scale": 0.01 } })
    
    for i in range(7):
        spec["priors"].append({ "variable": f"region_{i}_zscore", "distribution": { "kind": "normal", "loc": patient['zscores'][i], "scale": 0.05 } })
        
    spec['priors'].append({"variable": "tau_self_dynamic", "distribution": {"kind": "normal", "loc": float(np.mean(ode_s[:, 0])), "scale": max(float(np.std(ode_s[:, 0])), 0.01)}})
    spec['priors'].append({"variable": "memory_result_yr5", "distribution": {"kind": "normal", "loc": float(np.mean(ode_s[:, 5])), "scale": max(float(np.std(ode_s[:, 5])), 0.01)}})
    spec['priors'].append({"variable": "clinical_stage_yr5", "distribution": {"kind": "normal", "loc": float(np.mean(ode_s[:, 6])), "scale": max(float(np.std(ode_s[:, 6])), 0.01)}})
    
    spec['priors'].append({"variable": "prob_subtype_0", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 0])), "scale": max(float(np.std(sus_s[:, 0])), 0.01)}})
    spec['priors'].append({"variable": "prob_subtype_1", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 1])), "scale": max(float(np.std(sus_s[:, 1])), 0.01)}})
    spec['priors'].append({"variable": "prob_subtype_2", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 2])), "scale": max(float(np.std(sus_s[:, 2])), 0.01)}})
    spec['priors'].append({"variable": "expected_stage", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 3])), "scale": max(float(np.std(sus_s[:, 3])), 0.01)}})
    
    spec_path_temp = scratch_dir / f"temp_spec_{patient_id}.json"
    spec_path_temp.write_text(json.dumps(spec, indent=2))
    spec_path_str = str(spec_path_temp).replace('\\', '/')
    
    f = io.StringIO()
    with redirect_stdout(f):
        run_mm_cli('meta', 'build', spec_path_str)
        run_mm_cli('meta', 'sample', spec_path_str, '--draws', str(n_draws), '--tune', '50', '--chains', '2', '--seed', '42')
    out = f.getvalue()
    
    matches = re.findall(r'sample_id=([a-f0-9]+)', out)
    if not matches:
        raise ValueError(f"Could not find sample_id in output: {out}")
    import bayesian_metamodeling
    framework_root = Path(bayesian_metamodeling.__file__).parent.parent.parent
    dataset_path = framework_root / "tmp" / "metamodel_samples" / sample_id / "samples_dataset.json"
    with open(dataset_path, 'r') as f_json:
        data = json.load(f_json)
        
    return data['variables']

def plot_kde(ax, uncoupled, coupled, title, xlabel, is_bar=False, bins=None):
    if is_bar:
        width = 0.35
        def get_probs(arr):
            arr = np.round(arr)
            probs = [np.mean(arr == v) for v in bins]
            return probs
        u_probs = get_probs(uncoupled)
        c_probs = get_probs(coupled)
        x = np.arange(len(bins))
        ax.bar(x - width/2, u_probs, width, color='#007aff', alpha=0.7, label='Uncoupled')
        ax.bar(x + width/2, c_probs, width, color='#e74c3c', alpha=0.7, label='Coupled')
        ax.set_xticks(x)
        ax.set_xticklabels(['CN', 'MCI', 'Dementia'])
        ax.set_ylabel('Probability')
    else:
        sns.kdeplot(data=uncoupled, color='#007aff', fill=True, alpha=0.3, label='Uncoupled', ax=ax)
        sns.kdeplot(data=coupled, color='#e74c3c', fill=True, alpha=0.3, label='Coupled', ax=ax)
        ax.set_ylabel('Density')
        
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel)
    sns.despine(ax=ax, trim=True)

def evaluate_all(patients, spec_path):
    results = {}
    for patient in patients:
        print(f"Evaluating {patient['id']}...")
        ode_s, sus_s = get_uncoupled_samples(patient, n_draws=500)
        c_vars = run_coupled_samples(patient, patient['id'], spec_path, ode_s, sus_s, n_draws=250)
        
        results[patient['id']] = {
            'uncoupled_ode': ode_s,
            'uncoupled_sustain': sus_s,
            'coupled': c_vars
        }
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate patient coupling")
    parser.add_argument('--spec_path', type=str, 
                        default="C:/Project/AD_Metamodeling/Coupling/coupling_specs/metamodel_coupling.json",
                        help="Path to the master JSON coupling specification template.")
    args = parser.parse_args()
    
    patients = [
        {
            'id': 'P1', 
            'title': 'Subject 1: Fast-Growth Limbic Contradiction\n(APOE4-, Fast Growth Tau, Limbic Z-scores)', 
            'age': 60.0, 
            'apoe4': 0.0, 
            'amyloid_baseline': -0.5,
            'tau_baseline': 0.0,
            'amyloid_2yr': -0.2,
            'tau_2yr': 50.0,
            'zscores': [2.81, 1.58, 3.47, 0.8, 2.2, 0.4, 1.5]
        },
        {
            'id': 'P2', 
            'title': 'Subject 2: The Atypical Explosion\n(APOE4+, Extreme Growth Tau, Atypical Z-scores)', 
            'age': 65.0, 
            'apoe4': 1.0, 
            'amyloid_baseline': -0.98,
            'tau_baseline': 0.28,
            'amyloid_2yr': 0.58,
            'tau_2yr': 260.0,
            'zscores': [2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5]
        }
    ]
    
    print(f"Running evaluation using spec: {args.spec_path}")
    results = evaluate_all(patients, args.spec_path)
    
    sns.set_theme(style='ticks')
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axes = plt.subplots(6, 2, figsize=(14, 26), gridspec_kw={'hspace': 0.6, 'wspace': 0.3})
    
    for col, p in enumerate(patients):
        res = results[p['id']]
        ode_s, sus_s, c_vars = res['uncoupled_ode'], res['uncoupled_sustain'], res['coupled']
        
        u_tau = ode_s[:, 0]
        u_mem = ode_s[:, 5]  # memory_result_yr5 is output index 5
        u_clin = np.clip(np.round(ode_s[:, 6]), 0, 2)  # clinical_stage_yr5 is output index 6
        
        u_prob0 = sus_s[:, 0]
        u_prob1 = sus_s[:, 1]
        u_stage = sus_s[:, 3]
        
        def ext(name):
            return np.array(c_vars[name]).flatten()
            
        c_tau = ext('tau_self_dynamic')
        c_mem = ext('memory_result_yr5')
        c_clin = np.clip(np.round(ext('clinical_stage_yr5')), 0, 2)
        c_prob0 = ext('prob_subtype_0')
        c_prob1 = ext('prob_subtype_1')
        c_stage = ext('expected_stage')
        
        print(f"\n--- NUMERICAL RESULTS FOR {p['id']} ---")
        print(f"UNCOUPLED SuStaIn - S0: {np.mean(u_prob0):.3f}, S1: {np.mean(u_prob1):.3f}, Stage: {np.mean(u_stage):.3f}")
        print(f"UNCOUPLED ODE     - Tau: {np.mean(u_tau):.3f}, Mem: {np.mean(u_mem):.3f}, Clin: {np.mean(u_clin):.3f}")
        print(f"COUPLED SuStaIn   - S0: {np.mean(c_prob0):.3f}, S1: {np.mean(c_prob1):.3f}, Stage: {np.mean(c_stage):.3f}")
        print(f"COUPLED ODE       - Tau: {np.mean(c_tau):.3f}, Mem: {np.mean(c_mem):.3f}, Clin: {np.mean(c_clin):.3f}")
        print("-------------------------------------")
        
        ax = axes[0, col]
        sns.kdeplot(x=u_tau, y=u_mem, color='#007aff', alpha=0.5, fill=True, ax=ax, label='Uncoupled')
        sns.kdeplot(x=c_tau, y=c_mem, color='#e74c3c', alpha=0.5, fill=True, ax=ax, label='Coupled')
        ax.set_title(f"{p['title']}\n\nJoint: Tau Rate vs Memory", fontweight='bold', fontsize=12)
        ax.set_xlabel('Tau Self Dynamic Rate')
        ax.set_ylabel('Memory Deficit (Lower is Better)')
        
        plot_kde(axes[1, col], u_prob0, c_prob0, "SuStaIn Subtype 0 (Atypical) Probability", "Probability", False)
        plot_kde(axes[2, col], u_prob1, c_prob1, "SuStaIn Subtype 1 (Limbic) Probability", "Probability", False)
        plot_kde(axes[3, col], u_tau, c_tau, "ODE Tau Self Dynamic", "Rate", False)
        if col == 0:
            axes[3, col].legend()
            
        plot_kde(axes[4, col], u_clin, c_clin, "ODE Clinical Stage", "Stage", True, [0, 1, 2])
        if col == 0:
            axes[4, col].legend()
            
        plot_kde(axes[5, col], u_stage, c_stage, "SuStaIn Expected Stage", "Spatial Stage Severity", False)
        
    fig.suptitle("Patient-Specific Evaluation: Uncoupled vs Coupled Metamodel", fontsize=18, fontweight='bold', y=0.94)
    plt.savefig('C:/Project/AD_Metamodeling/Coupling/patient_coupling_evaluation.png', dpi=300, bbox_inches='tight')
    print("Done!")

if __name__ == '__main__':
    main()
