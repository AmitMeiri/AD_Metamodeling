"""
Isolated Hyperparameter Exploration & Tuning Runner
===================================================
This script systematically runs candidate hyperparameter configurations for the 
metamodel coupling in an isolated environment (Coupling/experiments/), comparing 
the coupled results against the expectations in Coupling_Test.md and the baseline.
"""

import os
import json
import re
import io
import shutil
from pathlib import Path
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# Ensure working directory is set to framework root for run_mm_cli
os.chdir("C:/Project/metamodeler_codex_scaffold_docs-develop")

from bayesian_metamodeling.tutorial import run_mm_cli
from bayesian_metamodeling.surrogates.backends import load_backend_model
import bayesian_metamodeling

# Setup paths
EXP_DIR = Path("C:/Project/AD_Metamodeling/Coupling/experiments")
SPECS_DIR = EXP_DIR / "specs"
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR = EXP_DIR / "plots"

for p in [SPECS_DIR, RESULTS_DIR, PLOTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Helper to find the latest trained surrogate payloads
def get_latest_model_path(base_dir):
    base = Path(base_dir)
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No trained surrogate model found in {base_dir}")
    subdirs.sort(key=lambda d: (d / 'backend_payload.json').stat().st_mtime if (d / 'backend_payload.json').exists() else 0)
    return subdirs[-1] / 'backend_payload.json'

ode_model_path = get_latest_model_path('C:/Project/AD_Metamodeling/ode_2023/Surrogate/tmp/surrogate_artifacts')
sustain_model_path = get_latest_model_path('C:/Project/AD_Metamodeling/SuStaIn_2021/Surrogate_SuStaIn/tmp/surrogate_artifacts')

ode_model = load_backend_model('sbi_npe', ode_model_path)
sustain_model = load_backend_model('sbi_npe', sustain_model_path)

ode_art_path = ode_model_path.parent / "artifact.json"
sustain_art_path = sustain_model_path.parent / "artifact.json"

# Patient test profiles from Coupling_Test.md
patients = [
    {
        'id': 'patient1_conflict',
        'name': 'Subject 1: Fast-Growth Limbic Contradiction',
        'desc': 'APOE4-, Fast Growth Tau, Limbic Z-scores',
        'age': 60.0,
        'apoe4': 0,
        'amyloid_baseline': -0.5,
        'tau_baseline': 0.0,
        'amyloid_2yr': -0.2,
        'tau_2yr': 50.0,
        'zscores': [2.81, 1.58, 3.47, 0.8, 2.2, 0.4, 1.5]
    },
    {
        'id': 'patient2_aligned',
        'name': 'Subject 2: The Atypical Explosion',
        'desc': 'APOE4+, Extreme Growth Tau, Atypical Z-scores',
        'age': 65.0,
        'apoe4': 1,
        'amyloid_baseline': -0.98,
        'tau_baseline': 0.28,
        'amyloid_2yr': 0.58,
        'tau_2yr': 260.0,
        'zscores': [2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5]
    }
]

def get_uncoupled_samples(patient, n_draws=1000):
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

def build_spec(cfg, out_path):
    spec = {
      "schema_version": "1.0",
      "name": f"coupling_{cfg['id']}",
      "ppl_backend": "pymc",
      "surrogate_refs": [
        str(ode_art_path).replace('\\', '/'),
        str(sustain_art_path).replace('\\', '/')
      ],
      "variables": [
        { "name": "age_baseline", "type": "scalar" },
        { "name": "apoe4_status", "type": "scalar" },
        { "name": "amyloid_baseline", "type": "scalar" },
        { "name": "tau_baseline", "type": "scalar" },
        { "name": "amyloid_2yr", "type": "scalar" },
        { "name": "tau_2yr", "type": "scalar" },
        { "name": "tau_self_dynamic", "type": "scalar" },
        { "name": "amyloid_self_dynamic", "type": "scalar" },
        { "name": "amyloid_drive_tau", "type": "scalar" },
        { "name": "memory_result_yr5", "type": "scalar" },
        { "name": "clinical_stage_yr5", "type": "scalar" },
        { "name": "clinical_stage_baseline", "type": "scalar" },
        { "name": "region_0_zscore", "type": "scalar" },
        { "name": "region_1_zscore", "type": "scalar" },
        { "name": "region_2_zscore", "type": "scalar" },
        { "name": "region_3_zscore", "type": "scalar" },
        { "name": "region_4_zscore", "type": "scalar" },
        { "name": "region_5_zscore", "type": "scalar" },
        { "name": "region_6_zscore", "type": "scalar" },
        { "name": "prob_subtype_0", "type": "scalar" },
        { "name": "prob_subtype_1", "type": "scalar" },
        { "name": "prob_subtype_2", "type": "scalar" },
        { "name": "expected_stage", "type": "scalar" },
        { "name": "suStIn_global_tau", "type": "scalar" }
      ],
      "priors": [
        { "variable": "age_baseline", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "apoe4_status", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "amyloid_baseline", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "tau_baseline", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "amyloid_2yr", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "tau_2yr", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_0_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_1_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_2_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_3_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_4_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_5_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_6_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } }
      ],
      "couplings": [
        {
          "kind": "gaussian_link",
          "source": "expected_stage,prob_subtype_0,prob_subtype_1,prob_subtype_2",
          "target": "clinical_stage_baseline",
          "transform": { 
              "kind": "sustain_to_ode_stage",
              "alpha": cfg.get("alpha", 0.4),
              "midpoint_limbic": cfg.get("midpoint_limbic", 10.0),
              "midpoint_atypical": cfg.get("midpoint_atypical", 15.0)
          },
          "sigma": cfg.get("sigma_stage", 0.2)
        },
        {
          "kind": "deterministic",
          "source": "region_0_zscore,region_1_zscore,region_2_zscore,region_3_zscore,region_4_zscore,region_5_zscore,region_6_zscore",
          "target": "suStIn_global_tau",
          "transform": { "kind": "sum" }
        },
        {
          "kind": "directional_potential",
          "source": "prob_subtype_0,prob_subtype_1,prob_subtype_2",
          "target": "tau_self_dynamic",
          "transform": { "kind": "velocity_modifier_score", "weights": cfg.get("vel_weights", [1.0, -0.3, 0.5]) },
          "sigma": cfg.get("sigma_velocity", 2.0)
        },
        {
          "kind": "gaussian_link",
          "source": "apoe4_status,tau_self_dynamic,tau_baseline,memory_result_baseline,tau_2yr",
          "target": "prob_subtype_0,prob_subtype_1,prob_subtype_2",
          "transform": { 
              "kind": "clinical_subtype_scorer",
              "beta": cfg.get("beta", 1.0),
              "w_apoe": cfg.get("w_apoe", 1.0),
              "v_scale": cfg.get("v_scale", 1.0),
              "w_mem": cfg.get("w_mem", 1.0)
          },
          "sigma": cfg.get("sigma_subtype", 0.25)
        }
      ]
    }
    with open(out_path, 'w') as f:
        json.dump(spec, f, indent=2)

def run_coupled_samples(patient, cfg, spec_path, ode_s, sus_s, n_draws=1500):
    with open(spec_path, 'r') as f:
        spec = json.load(f)
    
    spec["priors"] = [
        { "variable": "age_baseline", "distribution": { "kind": "normal", "loc": patient['age'], "scale": 0.01 } },
        { "variable": "apoe4_status", "distribution": { "kind": "normal", "loc": patient['apoe4'], "scale": 0.01 } },
        { "variable": "amyloid_baseline", "distribution": { "kind": "normal", "loc": patient['amyloid_baseline'], "scale": 0.01 } },
        { "variable": "tau_baseline", "distribution": { "kind": "normal", "loc": patient['tau_baseline'], "scale": 0.01 } },
        { "variable": "amyloid_2yr", "distribution": { "kind": "normal", "loc": patient['amyloid_2yr'], "scale": 0.01 } },
        { "variable": "tau_2yr", "distribution": { "kind": "normal", "loc": patient['tau_2yr'], "scale": 0.01 } }
    ]
    for i in range(7):
        spec["priors"].append({ "variable": f"region_{i}_zscore", "distribution": { "kind": "normal", "loc": patient['zscores'][i], "scale": 0.05 } })
        
    spec['priors'].append({"variable": "tau_self_dynamic", "distribution": {"kind": "normal", "loc": float(np.mean(ode_s[:, 0])), "scale": max(float(np.std(ode_s[:, 0])), 0.01)}})
    spec['priors'].append({"variable": "memory_result_baseline", "distribution": {"kind": "normal", "loc": float(np.mean(ode_s[:, 3])), "scale": max(float(np.std(ode_s[:, 3])), 0.01)}})
    spec['priors'].append({"variable": "clinical_stage_baseline", "distribution": {"kind": "normal", "loc": float(np.mean(ode_s[:, 4])), "scale": max(float(np.std(ode_s[:, 4])), 0.01)}})
    spec['priors'].append({"variable": "prob_subtype_0", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 0])), "scale": max(float(np.std(sus_s[:, 0])), 0.01)}})
    spec['priors'].append({"variable": "prob_subtype_1", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 1])), "scale": max(float(np.std(sus_s[:, 1])), 0.01)}})
    spec['priors'].append({"variable": "prob_subtype_2", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 2])), "scale": max(float(np.std(sus_s[:, 2])), 0.01)}})
    spec['priors'].append({"variable": "expected_stage", "distribution": {"kind": "normal", "loc": float(np.mean(sus_s[:, 3])), "scale": max(float(np.std(sus_s[:, 3])), 0.01)}})
    
    spec_path_temp = SPECS_DIR / f"temp_spec_{cfg['id']}_{patient['id']}.json"
    with open(spec_path_temp, 'w') as f:
        json.dump(spec, f, indent=2)
    spec_path_str = str(spec_path_temp).replace('\\', '/')
    
    f_out = io.StringIO()
    with redirect_stdout(f_out):
        run_mm_cli('meta', 'build', spec_path_str)
        run_mm_cli('meta', 'sample', spec_path_str, '--draws', str(n_draws), '--tune', '500', '--chains', '2', '--seed', '42')
    out = f_out.getvalue()
    
    matches = re.findall(r'sample_id=([a-f0-9]+)', out)
    if not matches:
        raise ValueError(f"Could not find sample_id in output: {out}")
    sample_id = matches[-1]
    
    framework_root = Path(bayesian_metamodeling.__file__).parent.parent.parent
    dataset_path = framework_root / "tmp" / "metamodel_samples" / sample_id / "samples_dataset.json"
    with open(dataset_path, 'r') as f_json:
        data = json.load(f_json)
        
    return data['variables']

def plot_experiment_figure(patient_data_list, cfg, out_plot_path):
    fig = plt.figure(figsize=(13, 22))
    fig.suptitle(f"Patient-Specific Evaluation [{cfg['id']}]: Uncoupled vs Coupled\n{cfg['desc']}", 
                 fontsize=12, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.38)
    
    BLUE = '#4C8EDA'
    RED = '#E05252'
    
    def plot_kde(ax, uncoupled, coupled, title, xlabel, is_bar=False, bins=None):
        if is_bar:
            width = 0.35
            def get_probs(arr):
                arr = np.round(arr)
                return [np.mean(arr == v) for v in bins]
            u_probs = get_probs(uncoupled)
            c_probs = get_probs(coupled)
            x_ind = np.arange(len(bins))
            ax.bar(x_ind - width/2, u_probs, width, label='Uncoupled', color=BLUE, alpha=0.7)
            ax.bar(x_ind + width/2, c_probs, width, label='Coupled', color=RED, alpha=0.7)
            ax.set_xticks(x_ind)
            ax.set_xticklabels(['CN', 'MCI', 'Dementia'], fontsize=8)
            ax.set_ylabel("Probability", fontsize=8)
            ax.set_ylim(0, 1.05)
        else:
            def kde_plot(data, color):
                try:
                    kde = gaussian_kde(data)
                    x_min = data.min() - abs(data.min()) * 0.2
                    x_max = data.max() + abs(data.max()) * 0.2
                    xs = np.linspace(x_min, x_max, 200)
                    ys = kde(xs)
                    ax.fill_between(xs, ys, alpha=0.5, color=color)
                    ax.plot(xs, ys, color=color, lw=1.5)
                except Exception:
                    pass
            kde_plot(uncoupled, BLUE)
            kde_plot(coupled, RED)
            ax.set_ylabel("Density", fontsize=8)
            
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)

    for col_idx, (p_info, u_ode, u_sus, c_vars) in enumerate(patient_data_list):
        # 1. 2D Joint
        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_title(f"{p_info['name']}\n({p_info['desc']})\nJoint: Tau Rate vs Memory", fontsize=9, fontweight='bold')
        try:
            ax.scatter(u_ode[:300, 0], u_ode[:300, 3], color=BLUE, alpha=0.25, s=10, label='Uncoupled')
            ax.scatter(np.array(c_vars['tau_self_dynamic'])[:300], np.array(c_vars['memory_result_yr5'])[:300], color=RED, alpha=0.25, s=10, label='Coupled')
        except Exception:
            pass
        ax.set_xlabel("Tau Self Dynamic Rate", fontsize=7)
        ax.set_ylabel("Memory Deficit", fontsize=7)
        ax.tick_params(labelsize=7)
        
        # 2. Subtype 0
        ax = fig.add_subplot(gs[1, col_idx])
        plot_kde(ax, u_sus[:, 0], np.array(c_vars['prob_subtype_0']), "SuStaIn Subtype 0 (Atypical) Probability", "Probability")
        
        # 3. Subtype 1
        ax = fig.add_subplot(gs[2, col_idx])
        plot_kde(ax, u_sus[:, 1], np.array(c_vars['prob_subtype_1']), "SuStaIn Subtype 1 (Limbic) Probability", "Probability")
        
        # 4. Tau rate
        ax = fig.add_subplot(gs[3, col_idx])
        plot_kde(ax, u_ode[:, 0], np.array(c_vars['tau_self_dynamic']), "ODE Tau Self Dynamic", "Rate")
        
        # 5. Clinical Stage
        ax = fig.add_subplot(gs[4, col_idx])
        plot_kde(ax, u_ode[:, 4], np.array(c_vars['clinical_stage_baseline']), "ODE Clinical Stage", "Stage", is_bar=True, bins=[0, 1, 2])
        
        # 6. Spatial Stage
        ax = fig.add_subplot(gs[5, col_idx])
        plot_kde(ax, u_sus[:, 3], np.array(c_vars['expected_stage']), "SuStaIn Expected Stage", "Spatial Stage Severity")

    legend_elements = [
        Line2D([0], [0], color=BLUE, lw=2, label='Uncoupled'),
        Line2D([0], [0], color=RED, lw=2, label='Coupled')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 0.97), framealpha=0.85)
    plt.savefig(out_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

# Define exploration configurations
configurations = [
    {
        "id": "cfg_00_baseline",
        "desc": "Baseline (Current settings: w_apoe=1.0, v_scale=1.0, beta=1.0, sig_vel=2.0)",
        "w_apoe": 1.0, "v_scale": 1.0, "w_mem": 1.0, "beta": 1.0,
        "sigma_subtype": 0.25, "sigma_velocity": 2.0, "vel_weights": [1.0, -0.3, 0.5],
        "sigma_stage": 0.20, "alpha": 0.4, "midpoint_limbic": 10.0, "midpoint_atypical": 15.0
    },
    {
        "id": "cfg_01_tau_delta_apoe05",
        "desc": "Tau Delta + APOE 0.5, Beta 1.5, Sig_vel 1.0, Sig_sub 0.20",
        "w_apoe": 0.5, "v_scale": 1.0, "w_mem": 1.0, "beta": 1.5,
        "sigma_subtype": 0.20, "sigma_velocity": 1.0, "vel_weights": [1.5, -0.5, 0.8],
        "sigma_stage": 0.20, "alpha": 0.4, "midpoint_limbic": 10.0, "midpoint_atypical": 14.5
    },
    {
        "id": "cfg_02_tau_delta_apoe035",
        "desc": "Tau Delta + APOE 0.35, Beta 1.8, Sig_vel 0.8, Sig_sub 0.18",
        "w_apoe": 0.35, "v_scale": 1.0, "w_mem": 1.0, "beta": 1.8,
        "sigma_subtype": 0.18, "sigma_velocity": 0.8, "vel_weights": [1.5, -0.5, 1.0],
        "sigma_stage": 0.18, "alpha": 0.4, "midpoint_limbic": 10.0, "midpoint_atypical": 14.0
    },
    {
        "id": "cfg_03_tau_delta_tight",
        "desc": "Tau Delta + APOE 0.25, Beta 2.0, Sig_vel 0.6, Sig_sub 0.15",
        "w_apoe": 0.25, "v_scale": 1.2, "w_mem": 1.0, "beta": 2.0,
        "sigma_subtype": 0.15, "sigma_velocity": 0.6, "vel_weights": [1.8, -0.6, 1.0],
        "sigma_stage": 0.15, "alpha": 0.4, "midpoint_limbic": 9.5, "midpoint_atypical": 13.5
    },
    {
        "id": "cfg_04_balanced_optimal",
        "desc": "Tau Delta + APOE 0.40, Beta 1.6, Sig_vel 0.8, Sig_sub 0.20",
        "w_apoe": 0.40, "v_scale": 1.0, "w_mem": 1.0, "beta": 1.6,
        "sigma_subtype": 0.20, "sigma_velocity": 0.8, "vel_weights": [1.5, -0.4, 0.8],
        "sigma_stage": 0.18, "alpha": 0.4, "midpoint_limbic": 10.0, "midpoint_atypical": 14.0
    },
    {
        "id": "cfg_05_gentle_optimal",
        "desc": "Tau Delta + APOE 0.45, Beta 1.4, Sig_vel 1.0, Sig_sub 0.22",
        "w_apoe": 0.45, "v_scale": 0.8, "w_mem": 1.0, "beta": 1.4,
        "sigma_subtype": 0.22, "sigma_velocity": 1.0, "vel_weights": [1.2, -0.3, 0.6],
        "sigma_stage": 0.20, "alpha": 0.4, "midpoint_limbic": 10.0, "midpoint_atypical": 14.5
    }
]

def evaluate_metrics(s1_unc_sus, s1_unc_ode, s1_c_vars, s2_unc_sus, s2_unc_ode, s2_c_vars):
    # Subject 1 Metrics (Conflict resolution)
    s1_u_s1 = float(np.mean(s1_unc_sus[:, 1]))
    s1_c_s1 = float(np.mean(s1_c_vars['prob_subtype_1']))
    s1_s1_drop = s1_u_s1 - s1_c_s1  # Should be positive (Limbic drops)
    
    s1_u_s0 = float(np.mean(s1_unc_sus[:, 0]))
    s1_c_s0 = float(np.mean(s1_c_vars['prob_subtype_0']))
    s1_s0_rise = s1_c_s0 - s1_u_s0  # Should be positive (Atypical rises)
    
    s1_c_stage = float(np.mean(s1_c_vars['clinical_stage_baseline']))
    s1_stage_score = 1.0 - abs(s1_c_stage - 0.8) / 0.8 # Target ~0.8 (MCI boundary)
    
    # Subject 2 Metrics (Mutual amplification & Atypical dominance)
    s2_c_s0 = float(np.mean(s2_c_vars['prob_subtype_0'])) # Target >= 0.75
    s2_c_s1 = float(np.mean(s2_c_vars['prob_subtype_1'])) # Target <= 0.15
    
    s2_u_tau = float(np.mean(s2_unc_ode[:, 0]))
    s2_c_tau = float(np.mean(s2_c_vars['tau_self_dynamic']))
    s2_tau_boost = s2_c_tau - s2_u_tau # Should be positive
    
    s2_c_stage = float(np.mean(s2_c_vars['clinical_stage_baseline'])) # Target >= 1.25
    
    # Composite Score (0-100)
    score_s1 = (
        (1.0 if s1_s1_drop > 0 else 0.0) * 15 +
        (1.0 if s1_s0_rise > 0 else 0.0) * 15 +
        max(0.0, min(1.0, s1_stage_score)) * 10
    )
    score_s2 = (
        min(1.0, s2_c_s0 / 0.75) * 25 +
        max(0.0, (0.50 - s2_c_s1) / 0.35) * 15 +
        (1.0 if s2_tau_boost > 0 else 0.0) * 10 +
        min(1.0, s2_c_stage / 1.30) * 10
    )
    
    total_score = round(score_s1 + score_s2, 1)
    
    return {
        "Total_Score": total_score,
        "S1_Subtype1_Drop": round(s1_s1_drop, 3),
        "S1_Subtype0_Rise": round(s1_s0_rise, 3),
        "S1_Clin_Stage": round(s1_c_stage, 3),
        "S2_Subtype0_Prob": round(s2_c_s0, 3),
        "S2_Subtype1_Prob": round(s2_c_s1, 3),
        "S2_Tau_Boost": round(s2_tau_boost, 4),
        "S2_Clin_Stage": round(s2_c_stage, 3)
    }

def main():
    print("=" * 60)
    print("STARTING HYPERPARAMETER TUNING RUN")
    print("=" * 60)
    
    print("\nDrawing uncoupled baseline samples for patients...")
    s1_u_ode, s1_u_sus = get_uncoupled_samples(patients[0])
    s2_u_ode, s2_u_sus = get_uncoupled_samples(patients[1])
    
    leaderboard = []
    runs_results = {}
    
    for cfg in configurations:
        cfg_id = cfg['id']
        print(f"\nEvaluating Configuration: {cfg_id}...")
        print(f"  Description: {cfg['desc']}")
        
        spec_path = SPECS_DIR / f"spec_{cfg_id}.json"
        build_spec(cfg, spec_path)
        
        print("  Sampling Subject 1 (Conflict)...")
        s1_c_vars = run_coupled_samples(patients[0], cfg, spec_path, s1_u_ode, s1_u_sus, n_draws=1500)
        
        print("  Sampling Subject 2 (Amplification)...")
        s2_c_vars = run_coupled_samples(patients[1], cfg, spec_path, s2_u_ode, s2_u_sus, n_draws=1500)
        
        metrics = evaluate_metrics(s1_u_sus, s1_u_ode, s1_c_vars, s2_u_sus, s2_u_ode, s2_c_vars)
        metrics["Config_ID"] = cfg_id
        metrics["Description"] = cfg["desc"]
        
        print(f"  -> Total Alignment Score: {metrics['Total_Score']}/100")
        print(f"     S1 Subtype 1 Drop: {metrics['S1_Subtype1_Drop']}, S1 Subtype 0 Rise: {metrics['S1_Subtype0_Rise']}")
        print(f"     S2 Subtype 0 Prob: {metrics['S2_Subtype0_Prob']} (Target >= 0.75), S2 Subtype 1 Prob: {metrics['S2_Subtype1_Prob']} (Target <= 0.15)")
        print(f"     S2 Tau Boost: {metrics['S2_Tau_Boost']}, S2 Clinical Stage: {metrics['S2_Clin_Stage']}")
        
        leaderboard.append(metrics)
        
        patient_data_list = [
            (patients[0], s1_u_ode, s1_u_sus, s1_c_vars),
            (patients[1], s2_u_ode, s2_u_sus, s2_c_vars)
        ]
        plot_path = PLOTS_DIR / f"exp_{cfg_id}_evaluation.png"
        plot_experiment_figure(patient_data_list, cfg, plot_path)
        print(f"  -> Saved plot: {plot_path}")
        
        runs_results[cfg_id] = {
            "config": cfg,
            "metrics": metrics
        }
        
    df_leaderboard = pd.DataFrame(leaderboard).sort_values(by="Total_Score", ascending=False)
    csv_path = RESULTS_DIR / "tuning_leaderboard.csv"
    df_leaderboard.to_csv(csv_path, index=False)
    
    with open(RESULTS_DIR / "tuning_summary.json", "w") as f:
        json.dump(runs_results, f, indent=2)
        
    print("\n" + "=" * 60)
    print("TUNING COMPLETE! LEADERBOARD SUMMARY:")
    print("=" * 60)
    print(df_leaderboard[["Config_ID", "Total_Score", "S1_Subtype1_Drop", "S1_Subtype0_Rise", "S2_Subtype0_Prob", "S2_Subtype1_Prob", "S2_Tau_Boost"]].to_string(index=False))
    
    winner_cfg_id = df_leaderboard.iloc[0]["Config_ID"]
    winner_plot_src = PLOTS_DIR / f"exp_{winner_cfg_id}_evaluation.png"
    winner_plot_dst = PLOTS_DIR / "winner_evaluation.png"
    shutil.copy(winner_plot_src, winner_plot_dst)
    print(f"\nWinner configuration: {winner_cfg_id} (Score: {df_leaderboard.iloc[0]['Total_Score']}/100)")
    print(f"Winner plot saved to: {winner_plot_dst}")

if __name__ == "__main__":
    main()
