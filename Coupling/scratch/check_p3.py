import numpy as np
import evaluate_coupling_patient as ec

p = {
    'id': 'P3', 
    'title': 'P3: Severe Aligned\n(APOE4+, Severe Am, Severe Z-scores)', 
    'age': 85.0, 
    'apoe4': 1.0, 
    'amyloid_5yr': 0.5,  # Severe disease range
    'tau_5yr': 850.0,  # Dementia range
    'zscores': [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
}

ode_s, sus_s = ec.get_uncoupled_samples(p, 500)
c_vars = ec.run_coupled_samples(p, 'P3', 250)

print('Uncoupled stage:', np.mean(ode_s[:,4]), 'Coupled:', np.mean(c_vars['clinical_stage']))
print('Uncoupled prob0:', np.mean(sus_s[:,0]), 'Coupled:', np.mean(c_vars['prob_subtype_0']))
print('Uncoupled tau self:', np.mean(ode_s[:,0]), 'Coupled:', np.mean(c_vars['tau_self_dynamic']))
