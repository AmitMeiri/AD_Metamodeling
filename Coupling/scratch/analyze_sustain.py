import numpy as np
import pandas as pd
import json

print("--- SuStaIn Limbic Subtype Analysis ---")
sustain_sim = np.load('C:/Project/AD_Metamodeling/SuStaIn_2021/sim_zscore_output/tau_sim.npy')
sustain_info = pd.read_csv('C:/Project/AD_Metamodeling/SuStaIn_2021/sim_zscore_output/tau_sim_ground_truth.csv')

df = sustain_info
print("Value counts of subtypes:")
print(df['subtype'].value_counts())

for sub_id, sub_name in zip([0, 1, 2], ['Atypical', 'Limbic', 'Typical']):
    print(f"\nMean z-scores for Subtype {sub_id} ({sub_name}):")
    sub_df = df[df['subtype'] == sub_id]
    for stage in range(1, 22, 5):
        st = sub_df[sub_df['stage'] == stage]
        if len(st) > 0:
            idx = st.index.values
            z = sustain_sim[idx].mean(axis=0)
            print(f"Stage {stage:2d}: {[round(x, 2) for x in z]}")
