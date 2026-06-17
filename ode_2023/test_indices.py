import pickle
import numpy as np
from scipy.linalg import expm

with open('ad_ode_demo_posterior.pkl', 'rb') as f:
    pos = pickle.load(f)

v = pos['v']
wAge = pos['wAge']
wAPOE = pos['wAPOE']
v0 = pos['v0']
c0 = pos['c0']

np.random.seed(42)
N_draws = 50
tau_selfs = []
ab_selfs = []
mem_5yrs = []

for s in range(N_draws):
    age = np.random.uniform(60, 90)
    apoe4 = int(np.random.rand() > 0.5)
    subj_idx = np.random.randint(10)
    
    A = v[s] + age * wAge[s] + apoe4 * wAPOE[s]
    
    tau_self = A[1, 1]
    ab_self = A[0, 0]
    
    delta = c0[s, subj_idx] - v0[s]
    xt = expm(5.0 * A) @ delta + v0[s]
    mem_5yr = xt[3] # Memory index
    
    tau_selfs.append(tau_self)
    ab_selfs.append(ab_self)
    mem_5yrs.append(mem_5yr)

print('Tau Self Dynamics: min:', min(tau_selfs), 'max:', max(tau_selfs), 'mean:', np.mean(tau_selfs))
print('Abeta Self Dynamics: min:', min(ab_selfs), 'max:', max(ab_selfs), 'mean:', np.mean(ab_selfs))
print('Mem 5yr: min:', min(mem_5yrs), 'max:', max(mem_5yrs), 'mean:', np.mean(mem_5yrs))
