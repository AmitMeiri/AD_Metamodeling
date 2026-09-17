import os
import pickle
import pandas as pd
import numpy as np

# Load original merged table
df_tau = pd.read_csv('UCBERKELEY_TAU_6MM_06Sep2026.csv', low_memory=False)
df_master = pd.read_csv('ADNI_MASTER_ALL_SUBJECTS.csv', low_memory=False)
merged = pd.merge(df_tau, df_master[['PTID', 'VISCODE2', 'DX', 'VOGEL_SUBTYPE']].drop_duplicates(), on=['PTID', 'VISCODE2'], how='inner')

LOBES = {
    'Left_MTL': ['CTX_LH_ENTORHINAL', 'CTX_LH_PARAHIPPOCAMPAL', 'LEFT_AMYGDALA', 'LEFT_HIPPOCAMPUS'],
    'Right_MTL': ['CTX_RH_ENTORHINAL', 'CTX_RH_PARAHIPPOCAMPAL', 'RIGHT_AMYGDALA', 'RIGHT_HIPPOCAMPUS'],
    'Left_LatTemp': ['CTX_LH_INFERIORTEMPORAL', 'CTX_LH_MIDDLETEMPORAL', 'CTX_LH_SUPERIORTEMPORAL', 'CTX_LH_TRANSVERSETEMPORAL', 'CTX_LH_BANKSSTS', 'CTX_LH_TEMPORALPOLE', 'CTX_LH_FUSIFORM'],
    'Right_LatTemp': ['CTX_RH_INFERIORTEMPORAL', 'CTX_RH_MIDDLETEMPORAL', 'CTX_RH_SUPERIORTEMPORAL', 'CTX_RH_TRANSVERSETEMPORAL', 'CTX_RH_BANKSSTS', 'CTX_RH_TEMPORALPOLE', 'CTX_RH_FUSIFORM'],
    'Left_Parietal': ['CTX_LH_INFERIORPARIETAL', 'CTX_LH_SUPERIORPARIETAL', 'CTX_LH_PRECUNEUS', 'CTX_LH_POSTCENTRAL', 'CTX_LH_SUPRAMARGINAL', 'CTX_LH_ISTHMUSCINGULATE', 'CTX_LH_POSTERIORCINGULATE', 'CTX_LH_PARACENTRAL'],
    'Right_Parietal': ['CTX_RH_INFERIORPARIETAL', 'CTX_RH_SUPERIORPARIETAL', 'CTX_RH_PRECUNEUS', 'CTX_RH_POSTCENTRAL', 'CTX_RH_SUPRAMARGINAL', 'CTX_RH_ISTHMUSCINGULATE', 'CTX_RH_POSTERIORCINGULATE', 'CTX_RH_PARACENTRAL'],
    'Left_Occipital': ['CTX_LH_LATERALOCCIPITAL', 'CTX_LH_LINGUAL', 'CTX_LH_CUNEUS', 'CTX_LH_PERICALCARINE'],
    'Right_Occipital': ['CTX_RH_LATERALOCCIPITAL', 'CTX_RH_LINGUAL', 'CTX_RH_CUNEUS', 'CTX_RH_PERICALCARINE'],
    'Left_Frontal': ['CTX_LH_SUPERIORFRONTAL', 'CTX_LH_ROSTRALMIDDLEFRONTAL', 'CTX_LH_CAUDALMIDDLEFRONTAL', 'CTX_LH_PARSOPERCULARIS', 'CTX_LH_PARSTRIANGULARIS', 'CTX_LH_PARSORBITALIS', 'CTX_LH_LATERALORBITOFRONTAL', 'CTX_LH_MEDIALORBITOFRONTAL', 'CTX_LH_PRECENTRAL', 'CTX_LH_FRONTALPOLE', 'CTX_LH_ROSTRALANTERIORCINGULATE', 'CTX_LH_CAUDALANTERIORCINGULATE'],
    'Right_Frontal': ['CTX_RH_SUPERIORFRONTAL', 'CTX_RH_ROSTRALMIDDLEFRONTAL', 'CTX_RH_CAUDALMIDDLEFRONTAL', 'CTX_RH_PARSOPERCULARIS', 'CTX_RH_PARSTRIANGULARIS', 'CTX_RH_PARSORBITALIS', 'CTX_RH_LATERALORBITOFRONTAL', 'CTX_RH_MEDIALORBITOFRONTAL', 'CTX_RH_PRECENTRAL', 'CTX_RH_FRONTALPOLE', 'CTX_RH_ROSTRALANTERIORCINGULATE', 'CTX_RH_CAUDALANTERIORCINGULATE']
}
lobe_names = list(LOBES.keys())

for lobe, rois in LOBES.items():
    suvr_cols = [r + '_SUVR' for r in rois]
    vol_cols = [r + '_VOLUME' for r in rois]
    for c in suvr_cols + vol_cols:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')
    total_vol = merged[vol_cols].sum(axis=1)
    merged[lobe + '_SUVR'] = (merged[suvr_cols].values * merged[vol_cols].values).sum(axis=1) / total_vol

cn_mask = merged['DX'] == 'CN'
for lobe in lobe_names:
    cn_mean = merged.loc[cn_mask, lobe + '_SUVR'].mean()
    cn_std = merged.loc[cn_mask, lobe + '_SUVR'].std()
    merged[lobe + '_Z'] = ((merged[lobe + '_SUVR'] - cn_mean) / cn_std).clip(lower=0)

z_cols = [lobe + '_Z' for lobe in lobe_names]
valid_mask = merged[z_cols].notna().all(axis=1)
analysis_df = merged[valid_mask].copy().reset_index(drop=True)

# Load 4-subtype model (subtype3.pickle)
with open('real_data_output/pickle_files/adni_tau_real_subtype3.pickle', 'rb') as f:
    m4 = pickle.load(f)

# Load 6-subtype model (subtype5.pickle)
with open('real_data_output/pickle_files/adni_tau_real_subtype5.pickle', 'rb') as f:
    m6 = pickle.load(f)

# Function to build human-readable DataFrame
def build_clean_classification_csv(analysis_df, model_data, num_subtypes, output_filename, subtype_names_map):
    df = analysis_df.copy()
    
    ml_sub = np.ravel(model_data['ml_subtype'])
    ml_stg = np.ravel(model_data['ml_stage'])
    prob_sub = np.ravel(model_data['prob_ml_subtype'])
    prob_stg = np.ravel(model_data['prob_ml_stage'])
    
    # 1-based indexing for subtypes: Subtype 1 to num_subtypes.
    # When stage is 0 (healthy / no abnormal tau), label as Subtype 0 (Stage 0 / Tau-Negative).
    sub_1based = []
    sub_label = []
    
    for s, stg in zip(ml_sub, ml_stg):
        if np.isnan(stg) or stg == 0:
            sub_1based.append(0)
            sub_label.append("Stage 0 (Tau-Negative / Baseline)")
        else:
            sub_num = int(s) + 1
            sub_1based.append(sub_num)
            sub_label.append(subtype_names_map.get(sub_num, f"Subtype {sub_num}"))
            
    df['Subtype_Number'] = sub_1based
    df['Subtype_Label'] = sub_label
    df['Subtype_Probability'] = prob_sub
    df['Stage_Number'] = [int(x) if not np.isnan(x) else 0 for x in ml_stg]
    df['Stage_Probability'] = prob_stg
    
    cols = ['PTID', 'VISCODE2', 'DX', 'VOGEL_SUBTYPE', 'Subtype_Number', 'Subtype_Label', 'Subtype_Probability', 'Stage_Number', 'Stage_Probability'] + z_cols
    out_path = os.path.join('real_data_output', output_filename)
    df[cols].to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print("Distribution of Subtype_Number:")
    print(df['Subtype_Number'].value_counts().sort_index())
    print("-" * 50)

# Map names for 4-subtypes
MAP_4 = {
    1: "Subtype 1: Limbic-Predominant",
    2: "Subtype 2: MTL-Sparing / Cortical",
    3: "Subtype 3: Lateral Temporal",
    4: "Subtype 4: Posterior / Frontoparietal"
}

# Map names for 6-subtypes
MAP_6 = {
    1: "Subtype 1: Limbic (MTL-Restricted)",
    2: "Subtype 2: Limbic (MTL + Parietal)",
    3: "Subtype 3: Lateral Temporal (Left-Dominant)",
    4: "Subtype 4: Frontoparietal / Cortical",
    5: "Subtype 5: Lateral Temporal (Right-Dominant)",
    6: "Subtype 6: Lateral Temporal (Temporoparietal)"
}

print("Generating 4-subtype classification CSV...")
build_clean_classification_csv(analysis_df, m4, 4, 'real_data_classification_4subtypes.csv', MAP_4)

print("Generating 6-subtype classification CSV...")
build_clean_classification_csv(analysis_df, m6, 6, 'real_data_classification_6subtypes.csv', MAP_6)
