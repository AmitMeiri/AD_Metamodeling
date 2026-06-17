import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, "C:/Project/metamodeler_codex_scaffold_docs-develop/src")
from bayesian_metamodeling.surrogates.backends import load_backend_model

def get_latest_model_path(base_dir):
    base = Path(base_dir)
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda d: (d / 'backend_payload.json').stat().st_mtime if (d / 'backend_payload.json').exists() else 0)
    return subdirs[-1] / 'backend_payload.json'

ode_model = load_backend_model('sbi_npe', get_latest_model_path('C:/Project/ode_2023/tmp/surrogate_artifacts'))
x={'age_baseline':np.array([75.0]),'apoe4_status':np.array([1.0]),'amyloid_5yr':np.array([3.0]),'tau_5yr':np.array([3.0])}
s = ode_model.sample(x,n=100,seed=42)[0]
print("ODE predictions for 75, apoe4=1, am=3, tau=3:")
print("tau_self:", np.mean(s[:,0]))
print("mem_5yr:", np.mean(s[:,3]))
print("stage:", np.mean(np.round(s[:,4])))

x2={'age_baseline':np.array([85.0]),'apoe4_status':np.array([1.0]),'amyloid_5yr':np.array([10.0]),'tau_5yr':np.array([10.0])}
s2 = ode_model.sample(x2,n=100,seed=42)[0]
print("ODE predictions for 85, apoe4=1, am=10, tau=10:")
print("tau_self:", np.mean(s2[:,0]))
print("mem_5yr:", np.mean(s2[:,3]))
print("stage:", np.mean(np.round(s2[:,4])))
