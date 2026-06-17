import pickle
import numpy as np
import json
from scipy.linalg import expm

with open('../ad_ode_demo_posterior.pkl', 'rb') as f:
    posterior = pickle.load(f)

v_samples = posterior['v']
wAge_samples = posterior['wAge']
wAPOE_samples = posterior['wAPOE']
c0_samples = posterior['c0']
v0_samples = posterior['v0']

np.random.seed(42)
N_total = 4000

# First pass: simulate trajectories and extract raw values
ages = []
apoe4s = []
tau_selfs = []
ab_selfs = []
ab_drive_taus = []
ab_0yrs_raw = []
tau_0yrs_raw = []
mem_0yrs_raw = []
ab_2yrs_raw = []
tau_2yrs_raw = []
mem_5yrs_raw = []

for s in range(N_total):
    age = np.random.uniform(60, 90)
    apoe4 = int(np.random.rand() > 0.5)
    subj_idx = np.random.randint(10)
    
    # Calculate baseline self-dynamics directly from v_samples
    tau_self = v_samples[s, 1, 1]
    ab_self = v_samples[s, 0, 0]
    ab_drive_tau = v_samples[s, 1, 0]
    
    # Calculate subject-specific velocity matrix for ODE trajectory
    A = v_samples[s] + age * wAge_samples[s] + apoe4 * wAPOE_samples[s]
    delta = c0_samples[s, subj_idx] - v0_samples[s]
    
    # Baseline state
    x0 = c0_samples[s, subj_idx]
    
    # 2-year state
    x2 = expm(2.0 * A) @ delta + v0_samples[s]
    
    # 5-year state
    x5 = expm(5.0 * A) @ delta + v0_samples[s]
    
    ab_0yr = x0[0]
    tau_0yr = x0[1]
    mem_0yr = x0[3]
    ab_2yr = x2[0]
    tau_2yr = x2[1]
    mem_5yr = x5[3]  # Memory state at year 5
    
    ages.append(age)
    apoe4s.append(apoe4)
    tau_selfs.append(tau_self)
    ab_selfs.append(ab_self)
    ab_drive_taus.append(ab_drive_tau)
    ab_0yrs_raw.append(ab_0yr)
    tau_0yrs_raw.append(tau_0yr)
    mem_0yrs_raw.append(mem_0yr)
    ab_2yrs_raw.append(ab_2yr)
    tau_2yrs_raw.append(tau_2yr)
    mem_5yrs_raw.append(mem_5yr)

# Winsorize target parameters at 1st and 99th percentiles to handle unstable ODE draws
p1_mem0 = np.percentile(mem_0yrs_raw, 1)
p99_mem0 = np.percentile(mem_0yrs_raw, 99)
mem_0yrs_clean = np.clip(mem_0yrs_raw, p1_mem0, p99_mem0)

p1_mem5 = np.percentile(mem_5yrs_raw, 1)
p99_mem5 = np.percentile(mem_5yrs_raw, 99)
mem_5yrs_clean = np.clip(mem_5yrs_raw, p1_mem5, p99_mem5)

p1_ab0 = np.percentile(ab_0yrs_raw, 1)
p99_ab0 = np.percentile(ab_0yrs_raw, 99)
ab_0yrs_clean = np.clip(ab_0yrs_raw, p1_ab0, p99_ab0)

p1_tau0 = np.percentile(tau_0yrs_raw, 1)
p99_tau0 = np.percentile(tau_0yrs_raw, 99)
tau_0yrs_clean = np.clip(tau_0yrs_raw, p1_tau0, p99_tau0)

p1_ab2 = np.percentile(ab_2yrs_raw, 1)
p99_ab2 = np.percentile(ab_2yrs_raw, 99)
ab_2yrs_clean = np.clip(ab_2yrs_raw, p1_ab2, p99_ab2)

p1_tau2 = np.percentile(tau_2yrs_raw, 1)
p99_tau2 = np.percentile(tau_2yrs_raw, 99)
tau_2yrs_clean = np.clip(tau_2yrs_raw, p1_tau2, p99_tau2)

# Decide continuous clinical stage scores
def get_expected_stage(mem_score):
    eta = 2.0 * mem_score
    c1, c2 = 0.5, 1.5
    # Using scipy.special.expit could be better, but np.exp works fine here.
    # To prevent overflow:
    p_cn = 1.0 / (1.0 + np.exp(np.clip(eta - c1, -100, 100)))
    p_ad = 1.0 / (1.0 + np.exp(np.clip(c2 - eta, -100, 100)))
    p_mci = 1.0 - p_cn - p_ad
    return p_mci + 2.0 * p_ad

clinical_stage_0yrs = []
clinical_stage_5yrs = []
for s in range(N_total):
    clinical_stage_0yrs.append(get_expected_stage(mem_0yrs_clean[s]))
    clinical_stage_5yrs.append(get_expected_stage(mem_5yrs_clean[s]))

data_out = []
for s in range(N_total):
    data_out.append({
        'age_baseline': float(ages[s]),
        'apoe4_status': int(apoe4s[s]),
        'amyloid_baseline': float(ab_0yrs_clean[s]),
        'tau_baseline': float(tau_0yrs_clean[s]),
        'amyloid_2yr': float(ab_2yrs_clean[s]),
        'tau_2yr': float(tau_2yrs_clean[s]),
        'tau_self_dynamic': float(tau_selfs[s]),
        'amyloid_self_dynamic': float(ab_selfs[s]),
        'amyloid_drive_tau': float(ab_drive_taus[s]),
        'memory_result_baseline': float(mem_0yrs_clean[s]),
        'clinical_stage_baseline': float(clinical_stage_0yrs[s]),
        'memory_result_yr5': float(mem_5yrs_clean[s]),
        'clinical_stage_yr5': float(clinical_stage_5yrs[s])
    })

with open('ad_ode_training_data.json', 'w') as f:
    json.dump(data_out, f)

print(f'Generated {N_total} rows in ad_ode_training_data.json')
print(f'Winsorized thresholds - Memory Baseline: [{p1_mem0:.3f}, {p99_mem0:.3f}], Memory 5-Yr: [{p1_mem5:.3f}, {p99_mem5:.3f}]')
