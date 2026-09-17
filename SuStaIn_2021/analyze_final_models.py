import os
import pickle
import pandas as pd
import numpy as np

# Load classifications
df_class = pd.read_csv('real_data_output/real_data_classification.csv')
print("Total classified scans:", len(df_class))
print("\nPredicted Subtype distribution (6-subtype solution):")
print(df_class['Pred_Subtype'].value_counts(dropna=False).sort_index())

# Load 4-subtype model
with open('real_data_output/pickle_files/adni_tau_real_subtype3.pickle', 'rb') as f:
    model_4 = pickle.load(f)

# Load 6-subtype model
with open('real_data_output/pickle_files/adni_tau_real_subtype5.pickle', 'rb') as f:
    model_6 = pickle.load(f)

LOBES = [
    'Left_MTL', 'Right_MTL',
    'Left_LatTemp', 'Right_LatTemp',
    'Left_Parietal', 'Right_Parietal',
    'Left_Occipital', 'Right_Occipital',
    'Left_Frontal', 'Right_Frontal'
]

# Print 4-subtype event sequence summary
print("\n" + "="*50)
print("4-SUBTYPE MODEL TRAJECTORIES (VOGEL CANONICAL COMPARISON)")
print("="*50)

ml_seq_4 = model_4['ml_sequence_EM']
ml_f_4 = model_4['ml_f_EM']

for s in range(4):
    print(f"\n--- Subtype {s+1} (Prevalence: {ml_f_4[s]*100:.1f}%) ---")
    seq = ml_seq_4[s, :]
    # Decode event indices: event_idx // 3 is lobe, event_idx % 3 is z-score (2, 5, 10)
    z_map = {0: 'Z=2 (Mild)', 1: 'Z=5 (Moderate)', 2: 'Z=10 (Severe)'}
    early_events = []
    for rank, ev in enumerate(seq[:10]):
        lobe_idx = int(ev // 3)
        z_idx = int(ev % 3)
        early_events.append(f"{rank+1}. {LOBES[lobe_idx]} [{z_map[z_idx]}]")
    print("Earliest 10 Events:")
    for e in early_events:
        print("  ", e)

# Print 6-subtype event sequence summary
print("\n" + "="*50)
print("6-SUBTYPE MODEL TRAJECTORIES (DATA-DRIVEN CVIC MINIMUM)")
print("="*50)

ml_seq_6 = model_6['ml_sequence_EM']
ml_f_6 = model_6['ml_f_EM']

for s in range(6):
    print(f"\n--- Subtype {s+1} (Prevalence: {ml_f_6[s]*100:.1f}%) ---")
    seq = ml_seq_6[s, :]
    z_map = {0: 'Z=2', 1: 'Z=5', 2: 'Z=10'}
    early_events = []
    for rank, ev in enumerate(seq[:8]):
        lobe_idx = int(ev // 3)
        z_idx = int(ev % 3)
        early_events.append(f"{rank+1}. {LOBES[lobe_idx]} [{z_map[z_idx]}]")
    print("Earliest 8 Events:")
    for e in early_events:
        print("  ", e)

# Cross-reference with Vogel Ground Truth where available
vogel_valid = df_class[df_class['VOGEL_SUBTYPE'].notna()]
print(f"\nOverlapping scans with Vogel ground truth labels: {len(vogel_valid)}")
print("\nCross-tabulation (Predicted Subtype vs Vogel Subtype):")
print(pd.crosstab(vogel_valid['Pred_Subtype'], vogel_valid['VOGEL_SUBTYPE']))
