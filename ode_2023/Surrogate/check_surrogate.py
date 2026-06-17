import json
import numpy as np
import pandas as pd
from pathlib import Path
from bayesian_metamodeling.surrogates.backends import load_backend_model

# Load training data
with open('ad_ode_training_data.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

print("Training Data Summary:")
print(df.describe())

# Load trained surrogate
model_path = Path(r'tmp\surrogate_artifacts\b64bbc634735471396b99d492cef831d\backend_payload.json')
model = load_backend_model('sbi_npe', model_path)

# Generate surrogate data for a few inputs
inputs = {
    'age_baseline': df['age_baseline'].values[:100],
    'apoe4_status': df['apoe4_status'].values[:100]
}
samples = model.sample(inputs, n=1, seed=42)
samples = samples.reshape(100, 3)
df_surr = pd.DataFrame(samples, columns=['tau_self_dynamic', 'amyloid_self_dynamic', 'memory_cognitive_test_result'])

print("\nSurrogate Predictions (100 samples) Summary:")
print(df_surr.describe())

# Test conditional points
conditions = [
    {'age_baseline': 65, 'apoe4_status': 0},
    {'age_baseline': 65, 'apoe4_status': 1},
    {'age_baseline': 85, 'apoe4_status': 0},
    {'age_baseline': 85, 'apoe4_status': 1}
]

print("\nConditional Statistics (2000 samples each):")
for cond in conditions:
    x_input = {
        'age_baseline': np.array([cond['age_baseline']]),
        'apoe4_status': np.array([cond['apoe4_status']])
    }
    cond_samples = model.sample(x_input, n=2000, seed=42)[0]
    mean = np.mean(cond_samples, axis=0)
    std = np.std(cond_samples, axis=0)
    print(f"Age {cond['age_baseline']}, APOE4 {cond['apoe4_status']}:")
    print(f"  Tau: Mean={mean[0]:.4f}, Std={std[0]:.4f}")
    print(f"  Amyloid: Mean={mean[1]:.4f}, Std={std[1]:.4f}")
    print(f"  Memory: Mean={mean[2]:.4f}, Std={std[2]:.4f}")
