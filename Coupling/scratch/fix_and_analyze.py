import json
import re

def fix_re():
    # Example to show if re.findall is better
    out = "sample_id=abc\nsample_id=def"
    match = re.search(r'sample_id=([a-f0-9]+)', out)
    print("search:", match.group(1))
    matches = re.findall(r'sample_id=([a-f0-9]+)', out)
    print("findall:", matches[-1])

import pandas as pd
import numpy as np
import pickle

def analyze_training_data():
    try:
        with open('C:/Project/AD_Metamodeling/ode_2023/Surrogate/ad_ode_training_data.json', 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            print("\n--- ODE Training Data Ranges ---")
            for col in ['amyloid_baseline', 'tau_baseline', 'amyloid_2yr', 'tau_2yr', 'memory_result_yr5', 'clinical_stage_yr5']:
                if col in df.columns:
                    print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    except Exception as e:
        print("Error loading ODE data:", e)

    try:
        with open('C:/Project/AD_Metamodeling/SuStaIn_2021/sim_zscore_output/tau_sim_10k_simulated_data.pickle', 'rb') as f:
            data = pickle.load(f)
            # data typically has ['data', 'subtype', 'stage']
            if isinstance(data, dict) and 'data' in data:
                zscores = data['data']
                subtype = data['subtype']
                stage = data['stage']
                print("\n--- SuStaIn Subtype 1 (Limbic) Mean Z-scores at Stage 10 ---")
                idx = (subtype == 1) & (stage == 10)
                if np.any(idx):
                    print(np.mean(zscores[idx], axis=0))
                
                print("\n--- SuStaIn Subtype 0 (Atypical) Mean Z-scores at Stage 10 ---")
                idx = (subtype == 0) & (stage == 10)
                if np.any(idx):
                    print(np.mean(zscores[idx], axis=0))
    except Exception as e:
        print("Error loading SuStaIn data:", e)

if __name__ == '__main__':
    fix_re()
    analyze_training_data()
