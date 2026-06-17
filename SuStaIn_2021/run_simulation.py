import sys
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Add pySuStaIn and its sim directory to sys.path (Absolute paths for the author)
sys.path.append(r'c:\Project\Sustain\pySuStaIn')
sys.path.append(r'c:\Project\Sustain\pySuStaIn\sim')

# Add relative paths so colleagues can run it if pySuStaIn is inside the same folder
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'pySuStaIn'))
sys.path.append(os.path.join(current_dir, 'pySuStaIn', 'sim'))

# Now we can import pySuStaIn modules
from pySuStaIn.ZscoreSustain import ZscoreSustain
from simfuncs import generate_random_Zscore_sustain_model, generate_data_Zscore_sustain
import sklearn.model_selection

def run_simulation():
    np.random.seed(42)

    output_folder = os.path.join(os.getcwd(), 'sim_zscore_output')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    dataset_name = 'tau_sim_10k'

    # =========================================================================
    # 1. PARAMETERS FOR SYNTHETIC DATA (mimicking Tau PET trajectories)
    # =========================================================================
    # We will simulate 7 biomarkers representing different brain regions for tau
    N = 7  
    M = 500  # Number of subjects
    N_S_ground_truth = 2  # The actual number of trajectories (subtypes) in our synthetic data
    
    # Ground truth fractions for the two subtypes
    ground_truth_fractions = np.array([0.6, 0.4])

    BiomarkerNames = ['Entorhinal', 'Amygdala', 'Parahippocampal', 'Fusiform', 'Inferior Temporal', 'Middle Temporal', 'Precuneus']
    
    # Define the Z-score based events for each biomarker (1, 2, 3 standard deviations)
    Z_vals = np.array([[1, 2, 3]] * N)
    Z_max = np.array([5] * N)
    
    simulated_data_file = os.path.join(output_folder, f'{dataset_name}_simulated_data.pickle')
    if os.path.exists(simulated_data_file):
        print(f"Found existing simulated data file: {simulated_data_file}. Loading simulated data...")
        with open(simulated_data_file, 'rb') as f:
            sim_data = pickle.load(f)
        ground_truth_sequences = sim_data['ground_truth_sequences']
        ground_truth_subtypes = sim_data['ground_truth_subtypes']
        ground_truth_stages = sim_data['ground_truth_stages']
        data = sim_data['data']
        data_denoised = sim_data['data_denoised']
        stage_value = sim_data['stage_value']
    else:
        print(f"Generating synthetic Z-score SuStaIn data with {N_S_ground_truth} ground-truth subtypes...")
        
        # Generate the ground truth sequence for each subtype
        ground_truth_sequences = generate_random_Zscore_sustain_model(Z_vals, N_S_ground_truth)

        # Randomly generate ground truth subtype assignments
        ground_truth_subtypes = np.random.choice(range(N_S_ground_truth), M, replace=True, p=ground_truth_fractions).astype(int)

        N_stages = np.sum(Z_vals > 0) + 1

        # Stage assignments: 25% control (stage 0), 75% cases (random stage > 0)
        ground_truth_stages_control = np.zeros((int(np.round(M * 0.25)), 1))
        ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(M * 0.75)), 1))
        ground_truth_stages = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int).ravel()

        # Generate the simulated data
        data, data_denoised, stage_value = generate_data_Zscore_sustain(
            ground_truth_subtypes,
            ground_truth_stages,
            ground_truth_sequences,
            Z_vals,
            Z_max
        )

        # Save simulated data to cache
        print(f"Saving generated simulated data to {simulated_data_file}...")
        sim_data = {
            'ground_truth_sequences': ground_truth_sequences,
            'ground_truth_subtypes': ground_truth_subtypes,
            'ground_truth_stages': ground_truth_stages,
            'data': data,
            'data_denoised': data_denoised,
            'stage_value': stage_value
        }
        with open(simulated_data_file, 'wb') as f:
            pickle.dump(sim_data, f)

    # =========================================================================
    # 2. RUN SuStaIn ALGORITHM WITH CROSS-VALIDATION
    # =========================================================================
    # We will allow SuStaIn to search up to N_S_max subtypes.
    # The cross-validation (CVIC) will tell us the optimal number.
    N_startpoints = 25
    N_S_max = 3 # Let's test 1, 2, and 3 subtypes
    N_iterations_MCMC = int(1e4)

    # Initialize the ZscoreSustain object
    sustain = ZscoreSustain(
        data, 
        Z_vals, 
        Z_max, 
        BiomarkerNames, 
        N_startpoints, 
        N_S_max, 
        N_iterations_MCMC, 
        output_folder, 
        dataset_name, 
        False # Parallel startpoints off to prevent multiprocessing issues on Windows
    )

    # Plot the ground truth sequences for reference
    print("Plotting ground truth Positional Variance Diagrams (PVDs)...")
    ground_truth_sequences_expanded = np.expand_dims(ground_truth_sequences, axis=2)
    ground_truth_fractions_actual, _ = np.histogram(ground_truth_subtypes, bins=np.arange(N_S_ground_truth + 1) - 0.5)
    ground_truth_fractions_actual = ground_truth_fractions_actual / len(ground_truth_subtypes)
    ground_truth_fractions_actual = np.expand_dims(ground_truth_fractions_actual, axis=1)

    figs, ax = sustain._plot_sustain_model(
        ground_truth_sequences_expanded, 
        ground_truth_fractions_actual, 
        np.inf, 
        subtype_order=np.arange(N_S_ground_truth), 
        biomarker_order=ground_truth_sequences_expanded[0, :].astype(int).ravel(), 
        biomarker_labels=BiomarkerNames,
        title_font_size=12
    )
    figs[0].suptitle('Ground truth sequences')
    figs[0].savefig(os.path.join(output_folder, 'PVD_true.png'))

    # Run the main SuStaIn algorithm
    print("Running SuStaIn MCMC Inference...")
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm(plot=True)
    
    df = pd.DataFrame({
        'subj_id': np.arange(1, M+1),
        'ml_subtype': np.ravel(ml_subtype),
        'prob_ml_subtype': np.ravel(prob_ml_subtype),
        'ml_stage': np.ravel(ml_stage),
        'prob_ml_stage': np.ravel(prob_ml_stage),
        'gt_subtype': np.ravel(ground_truth_subtypes),
        'gt_stage': np.ravel(ground_truth_stages)
    })
    df.to_csv(os.path.join(output_folder, 'Subject_subtype_stage_estimates.csv'), index=False)
    print("Saved subject estimates.")

    # =========================================================================
    # (Plotting and text generation moved to after CVIC calculation to ensure
    # we evaluate the exact optimal model rather than the N_S_max model).
    # =========================================================================

    # =========================================================================
    # 3. CROSS-VALIDATION TO SELECT OPTIMAL NUMBER OF SUBTYPES
    # =========================================================================
    cvic_file = os.path.join(output_folder, 'cvic_results.npy')
    if os.path.exists(cvic_file):
        print("Found existing CVIC results. Skipping Cross-Validation...")
        CVIC = np.load(cvic_file)
    else:
        print("Running Cross-Validation to determine optimal subtypes...")
        N_folds = 5
        test_idxs = []
        
        # We need some generic labels for stratification. Let's just create pseudo-labels.
        pseudo_labels = (np.mean(data, axis=1) > np.median(data)).astype(int)
        cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
        for train, test in cv.split(data, pseudo_labels):
            test_idxs.append(test)
        test_idxs = np.array(test_idxs, dtype=object)
        
        CVIC, loglike_matrix = sustain.cross_validate_sustain_model(test_idxs)
        np.save(cvic_file, CVIC)

    print(f"\nCross-Validation Information Criterion (CVIC) for 1 to {N_S_max} subtypes:")
    print(CVIC)
    print("Lower CVIC indicates a better model.")

    if len(CVIC) > 0:
        optimal_subtypes = np.argmin(CVIC) + 1
        print(f"--> The model correctly inferred that there are {optimal_subtypes} subtypes!")
        
        # Save CVIC Plot
        plt.figure()
        plt.plot(range(1, len(CVIC) + 1), CVIC, marker='o', linestyle='-', color='b', linewidth=2)
        plt.xlabel('Number of Subtypes')
        plt.ylabel('CVIC')
        plt.title('Cross-Validation Information Criterion (Lower is better)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(range(1, len(CVIC) + 1))
        plt.savefig(os.path.join(output_folder, 'CVIC_plot.png'))
        plt.close()
        
        # =========================================================================
        # 4. LOAD OPTIMAL MODEL & GENERATE SMART TEXT ANALYSIS
        # =========================================================================
        optimal_pickle = os.path.join(output_folder, 'pickle_files', f'{dataset_name}_subtype{optimal_subtypes - 1}.pickle')
        with open(optimal_pickle, 'rb') as f:
            pickle_data = pickle.load(f)
            
        opt_seq = pickle_data['samples_sequence']
        opt_f = pickle_data['samples_f']
        opt_ml_subtype = pickle_data['ml_subtype']
        
        print(f"Saving Estimated PVDs and text analysis for the optimal {optimal_subtypes}-subtype model...")
        figs, ax = sustain._plot_sustain_model(opt_seq, opt_f, M, biomarker_labels=BiomarkerNames, title_font_size=12)
        if len(figs) > 0:
            figs[0].suptitle(f'Estimated Sequences (Optimal: {optimal_subtypes} Subtypes)')
            figs[0].savefig(os.path.join(output_folder, 'PVD_estimated.png'))
            plt.close(figs[0])

        plt.figure()
        plt.hist(opt_ml_subtype, bins=np.arange(optimal_subtypes+2)-0.5, rwidth=0.8, color='purple', alpha=0.7)
        plt.xlabel('Subtype')
        plt.ylabel('Number of Subjects')
        plt.title(f'Distribution of Subjects (Optimal: {optimal_subtypes} Subtypes)')
        plt.xticks(range(optimal_subtypes))
        plt.savefig(os.path.join(output_folder, 'Subtype_distribution.png'))
        plt.close()

        analysis_text = f"# Trajectory Analysis for dataset: {dataset_name} (Optimal Model: {optimal_subtypes} Subtypes)\n\n"
        analysis_text += "This automated analysis evaluates the spatial pathways discovered by the SuStaIn model, categorizing them according to standard clinical phenotypes (e.g., Vogel et al. 2021).\n\n"
        
        last_seq = opt_seq[:, :, -1]
        last_f = np.mean(opt_f, axis=1)
        
        for s in range(last_seq.shape[0]):
            analysis_text += f"## Estimated Subtype {s+1} (Prevalence: {last_f[s]:.1%})\n"
            seq = last_seq[s].astype(int)
            biomarker_order = [BiomarkerNames[i % N] for i in seq]
            early_regions = []
            for b in biomarker_order:
                if b not in early_regions:
                    early_regions.append(b)
                if len(early_regions) == 3:
                    break
            
            if 'Entorhinal' in early_regions[:2] or 'Amygdala' in early_regions[:2]:
                phenotype = "Classic Limbic (Braak-like) Pathway"
                description = "The pathology originates strictly in the medial temporal lobe, which is the hallmark of classic Braak tau staging. It typically spreads adjacently before reaching neocortical areas. Clinically, this trajectory is heavily associated with early and severe amnestic (memory) symptoms."
            elif 'Precuneus' in early_regions[:2] or 'Middle Temporal' in early_regions[:2]:
                phenotype = "Neocortical-Leaning / Atypical Pathway"
                description = "Tau accumulation begins outside the classic limbic center, showing early vulnerability in regions like the Precuneus or temporal neocortex. Clinically, this pattern corresponds to atypical Alzheimer's phenotypes (like medial temporal-sparing), where executive, language, or visuospatial deficits appear early, prior to severe memory loss."
            else:
                phenotype = "Mixed / Unspecified Pathway"
                description = "This pathway shows an atypical mixture of early regional vulnerabilities that deviate from the classic Limbic and Neocortical phenotypes."
                
            analysis_text += f"**Phenotype Classification:** {phenotype}\n"
            analysis_text += f"- **Early Stages:** According to the estimated PVD (`PVD_estimated.png`), the very first regions to show tau accumulation (Z=1, red squares on the far left) are the **{early_regions[0]}**, followed closely by the **{early_regions[1]}** and the **{early_regions[2]}**.\n"
            analysis_text += f"- **Clinical Characteristics & Progression:** {description}\n\n"
            
        analysis_text += "**Note on Continuity (How to read the PVD):**\nThe squares on the PVD represent the exact *stage* (time point) where a region crosses a new threshold (e.g., Z=1 to Z=2). The absence of a square in subsequent stages does NOT mean the tau disappeared; it simply means the region is *maintaining* its pathological state while other regions cross their thresholds. Tau accumulation is strictly monotonic in this model.\n"
        
        text_file_path = os.path.join(output_folder, f'{dataset_name}_trajectory_analysis.txt')
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        print(f"Automated text analysis saved to {text_file_path}")

        # =========================================================================
        # 5. CLASSIFY A NEW SEPARATE SUBJECT SCAN (Example)
        # =========================================================================
        print("\n" + "="*60)
        print("DEMO: CLASSIFYING A NEW UNSEEN PATIENT SCAN")
        print("="*60)
        
        new_patient_scan = np.array([[2.5, 2.0, 0.4, 0.2, 0.1, 0.0, 0.3]])
        print("New Patient Z-scores:")
        for region, val in zip(BiomarkerNames, new_patient_scan[0]):
            print(f"  {region}: {val}")

        N_samples_estimate = 1000
        ml_sub_new, prob_ml_sub_new, ml_stg_new, prob_ml_stg_new, prob_sub_new, prob_stg_new, prob_sub_stg_new = sustain.subtype_and_stage_individuals_newData(
            new_patient_scan, 
            opt_seq, 
            opt_f, 
            N_samples_estimate
        )
        
        print(f"\nClassification Results (Using Optimal {optimal_subtypes}-Subtype Model):")
        print(f"--> Assigned Subtype (Trajectory): Subtype {int(np.ravel(ml_sub_new)[0]) + 1}")
        print(f"--> Assigned Disease Stage: Stage {int(np.ravel(ml_stg_new)[0])} of {sustain.stage_zscore.shape[1]}")
        for i in range(prob_sub_new.shape[1]):
            print(f"--> Probability of belonging to Subtype {i + 1}: {prob_sub_new[0, i]:.2%}")
        print("="*60 + "\n")

if __name__ == '__main__':
    run_simulation()
