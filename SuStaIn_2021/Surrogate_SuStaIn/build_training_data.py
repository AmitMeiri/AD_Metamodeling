import os
import pickle
import json
import numpy as np

def main():
    # Paths
    base_dir = r"C:\Project\Simulating_SuStaIn"
    output_dir = os.path.join(base_dir, "Surrogate_SuStaIn")
    os.makedirs(output_dir, exist_ok=True)
    
    sim_data_path = os.path.join(base_dir, "sim_zscore_output", "tau_sim_10k_simulated_data.pickle")
    cvic_path = os.path.join(base_dir, "sim_zscore_output", "cvic_results.npy")
    out_json_path = os.path.join(output_dir, "sustain_training_data.json")

    # 1. Determine Optimal Number of Subtypes using CVIC
    print(f"Loading CVIC results from {cvic_path}...")
    cvic = np.load(cvic_path)
    optimal_subtypes = np.argmin(cvic) + 1
    print(f"Optimal number of subtypes determined by CVIC: {optimal_subtypes}")

    # 2. Load Simulated Inputs
    print(f"Loading simulated data from {sim_data_path}...")
    with open(sim_data_path, 'rb') as f:
        sim_data = pickle.load(f)
        
    data_denoised = sim_data['data_denoised']
    M, N_regions = data_denoised.shape

    # 3. Load Optimal Model Outputs
    model_filename = f"tau_sim_10k_subtype{optimal_subtypes - 1}.pickle"
    model_path = os.path.join(base_dir, "sim_zscore_output", "pickle_files", model_filename)
    print(f"Loading optimal trained model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    prob_subtype_stage = model_data['prob_subtype_stage'] # shape: [M, N_stages + 1, N_subtypes]
    N_stages = prob_subtype_stage.shape[1] - 1
    N_subtypes = prob_subtype_stage.shape[2]
    
    if N_subtypes != optimal_subtypes:
        raise ValueError(f"Mismatch: Loaded model has {N_subtypes} subtypes but CVIC indicated {optimal_subtypes}.")
    
    # Marginalize over stages to get discrete subtype probabilities
    prob_subtype = np.sum(prob_subtype_stage, axis=1) # shape: [M, N_subtypes]
    
    # Marginalize over subtypes to get stage probabilities
    prob_stage = np.sum(prob_subtype_stage, axis=2) # shape: [M, N_stages + 1]
    
    # Compute expected stage: dot product with stage indices
    stage_indices = np.arange(N_stages + 1)
    expected_stage = np.dot(prob_stage, stage_indices) # shape: [M]

    # Format into JSON
    print(f"Structuring {M} samples...")
    data_out = []
    
    for i in range(M):
        sample = {
            "inputs": {
                f"region_{j}_zscore": float(data_denoised[i, j]) for j in range(N_regions)
            },
            "outputs": {
                f"prob_subtype_{s}": float(prob_subtype[i, s]) for s in range(N_subtypes)
            }
        }
        sample["outputs"]["expected_stage"] = float(expected_stage[i])
        data_out.append(sample)

    with open(out_json_path, 'w') as f:
        json.dump(data_out, f, indent=4)
        
    print(f"Successfully saved {M} training samples to {out_json_path}")

if __name__ == "__main__":
    main()
