import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import KFold

# Suppress warnings to keep console output clean
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Setup pySuStaIn Path
# -----------------------------------------------------------------------------
# We need to import ZscoreSustain from the pySuStaIn library folder.
# The library is located in a subfolder named 'pySuStaIn'.
sys.path.append(os.path.join(os.path.dirname(__file__), 'pySuStaIn'))
from pySuStaIn.ZscoreSustain import ZscoreSustain

def main():
    # =========================================================================
    # STEP 1: LOAD AND MERGE DATA
    # =========================================================================
    print("Loading data...")
    # df_tau contains the raw PET scan measurements (SUVR) and brain volumes.
    df_tau = pd.read_csv('UCBERKELEY_TAU_6MM_06Sep2026.csv', low_memory=False)
    # df_master contains the clinical diagnoses (DX), demographics, and Vogel labels.
    df_master = pd.read_csv('ADNI_MASTER_ALL_SUBJECTS.csv', low_memory=False)

    print("Merging tables...")
    # We merge on Patient ID (PTID) and Visit Code (VISCODE2) to match each scan
    # with its corresponding clinical diagnosis at that specific time point.
    merged = pd.merge(df_tau, df_master[['PTID', 'VISCODE2', 'DX', 'VOGEL_SUBTYPE']].drop_duplicates(), on=['PTID', 'VISCODE2'], how='inner')

    # =========================================================================
    # STEP 2: DEFINE REGIONS OF INTEREST (ROIs)
    # =========================================================================
    # Using all 68 individual Desikan-Killiany (DK) brain regions would make the 
    # model too complex (overparameterization). To fix this, Vogel et al. (2021) 
    # grouped them into 10 bilateral lobar regions (5 for the left brain, 5 for the right).
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

    print("Computing volume-weighted lobar SUVRs...")
    # Because some brain regions are larger than others, a simple average of the SUVR 
    # (Tau accumulation score) is inaccurate. We must weight each region by its volume.
    for lobe, rois in LOBES.items():
        suvr_cols = [r + '_SUVR' for r in rois]
        vol_cols = [r + '_VOLUME' for r in rois]
        
        # Convert to numeric in case the CSV loaded them as strings
        for c in suvr_cols + vol_cols:
            merged[c] = pd.to_numeric(merged[c], errors='coerce')
            
        total_vol = merged[vol_cols].sum(axis=1)
        weighted_suvr = (merged[suvr_cols].values * merged[vol_cols].values).sum(axis=1) / total_vol
        merged[lobe + '_SUVR'] = weighted_suvr

    # =========================================================================
    # STEP 3: Z-SCORE NORMALIZATION AGAINST HEALTHY CONTROLS
    # =========================================================================
    print("Computing Z-scores based on Cognitively Normal (CN) subjects...")
    # SuStaIn models disease *progression*, which means we need to know how abnormal 
    # a patient's scan is compared to a healthy baseline. 
    # We use patients with diagnosis 'CN' (Cognitively Normal) to establish this baseline.
    cn_mask = merged['DX'] == 'CN'
    for lobe in lobe_names:
        cn_mean = merged.loc[cn_mask, lobe + '_SUVR'].mean()
        cn_std = merged.loc[cn_mask, lobe + '_SUVR'].std()
        
        # Z-score formula: (Value - Mean) / Standard_Deviation
        merged[lobe + '_Z'] = (merged[lobe + '_SUVR'] - cn_mean) / cn_std
        
        # SuStaIn assumes that pathology only accumulates (it doesn't go backwards).
        # We clip negative Z-scores to 0 to ensure the values are strictly positive.
        merged[lobe + '_Z'] = merged[lobe + '_Z'].clip(lower=0)

    # Filter out any rows that have missing data (NaNs) in the Z-score columns
    z_cols = [lobe + '_Z' for lobe in lobe_names]
    valid_mask = merged[z_cols].notna().all(axis=1)
    analysis_df = merged[valid_mask].copy()
    
    print(f"Data ready: {len(analysis_df)} scans valid for modeling.")

    # =========================================================================
    # STEP 4: CONFIGURE SUSTAIN PARAMETERS
    # =========================================================================
    data = analysis_df[z_cols].values
    N = len(lobe_names)
    
    # Z_vals: The severity thresholds the algorithm will look for.
    # We use Z=2 (Mild), Z=5 (Moderate), and Z=10 (Severe) for every lobe.
    # This means the algorithm tracks 30 total events (10 lobes * 3 thresholds).
    Z_vals = np.array([[2, 5, 10]] * N)
    
    # Z_max: The absolute maximum Z-score we expect to see. Capped at 15.
    Z_max = np.array([15] * N)

    output_folder = os.path.join(os.getcwd(), 'real_data_output')
    os.makedirs(output_folder, exist_ok=True)

    # N_S_max sets the maximum number of subtypes the algorithm is allowed to look for.
    # We increased this to 6 so the algorithm has room to discover up to 6 distinct trajectories.
    N_S_max = 6
    
    # Initialize the Z-score version of the SuStaIn algorithm
    sustain = ZscoreSustain(
        data,
        Z_vals,
        Z_max,
        lobe_names,
        N_startpoints=5,          # Reduced from 10 to 5 to halve the number of chains
        N_S_max=N_S_max,          
        N_iterations_MCMC=1000,   # Reduced from 5000. 1000 is enough for CVIC comparison locally
        output_folder=output_folder,
        dataset_name='adni_tau_real',
        use_parallel_startpoints=True # ENABLE PARALLEL PROCESSING to use all CPU cores!
    )

    # =========================================================================
    # STEP 5: RUN CROSS-VALIDATION (TO FIND OPTIMAL NUMBER OF SUBTYPES)
    # =========================================================================
    print("Step 1: Running Cross-Validation to determine optimal number of subtypes (CVIC)...")
    # We use 5-Fold Cross Validation as requested to ensure maximum statistical reliability.
    N_folds = 5
    cv = KFold(n_splits=N_folds, shuffle=True, random_state=42)
    test_idxs = [test for train, test in cv.split(data)]
    
    # This step can take hours, as it trains models from 1 to 6 subtypes across 5 folds!
    # It outputs CVIC (Cross-Validation Information Criterion) - a score where lower is better.
    CVIC, loglike_matrix = sustain.cross_validate_sustain_model(test_idxs)
    
    # The optimal number of subtypes is the one that achieved the lowest CVIC score.
    optimal_subtypes = np.argmin(CVIC) + 1
    print(f"\n======================================")
    print(f"Data-driven discovery complete!")
    print(f"Optimal number of subtypes determined by CVIC: {optimal_subtypes}")
    print(f"CVIC Values: {CVIC}")
    print(f"======================================\n")

    # =========================================================================
    # STEP 6: TRAIN FINAL MODEL & SAVE CLASSIFICATIONS
    # =========================================================================
    print(f"Step 2: Training final SuStaIn model with the optimal number of subtypes ({optimal_subtypes})...")
    # Now that we mathematically proved how many subtypes exist, we run the algorithm 
    # one final time on the FULL dataset using only that optimal number.
    sustain.N_S_max = optimal_subtypes
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm(plot=True)

    print("Step 3: Saving classification results...")
    # The algorithm assigns every patient to their most likely Subtype and Stage.
    analysis_df['Pred_Subtype'] = np.ravel(ml_subtype)           # The subtype the patient belongs to
    analysis_df['Pred_Subtype_Prob'] = np.ravel(prob_ml_subtype) # How confident the model is (0.0 to 1.0)
    analysis_df['Pred_Stage'] = np.ravel(ml_stage)               # How far along the disease trajectory they are
    analysis_df['Pred_Stage_Prob'] = np.ravel(prob_ml_stage)     # Confidence of the stage assignment

    out_csv = os.path.join(output_folder, 'real_data_classification.csv')
    cols_to_save = ['PTID', 'VISCODE2', 'DX', 'VOGEL_SUBTYPE', 'Pred_Subtype', 'Pred_Subtype_Prob', 'Pred_Stage', 'Pred_Stage_Prob'] + z_cols
    analysis_df[cols_to_save].to_csv(out_csv, index=False)
    
    print(f"Successfully saved classifications to {out_csv}")
    print("Execution Finished!")

if __name__ == '__main__':
    main()
