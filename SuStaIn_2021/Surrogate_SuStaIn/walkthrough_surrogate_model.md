# Surrogate Model Walkthrough: Training and Sampling (SuStaIn)

This document explains how we built the specific surrogate model for the SuStaIn framework, how to train it, and how to sample from it.

---

## 0. What is this Surrogate Model?

The **Surrogate Model** is a fast, neural network-based approximation of the full, computationally heavy **SuStaIn MCMC Algorithm**.

The original SuStaIn model uses Markov Chain Monte Carlo (MCMC) to explore a massive mathematical space to determine a subject's disease stage and subtype based on cross-sectional brain scans. While highly accurate, running this full Bayesian inference for new patients is extremely slow. 

The goal of this surrogate is to "learn" the mathematical behavior of the SuStaIn MCMC. By training a Neural Posterior Estimator (NPE) on the thousands of generated posterior samples, the surrogate can instantly map a new patient's regional brain pathology directly to their exact stage and subtype probability distribution, effectively bypassing the slow inference engine.

---

## 1. How Our Specific Model Was Built

Before training the surrogate, we defined exactly what the neural network should learn. The goal was to map a patient's spatial tau pathology directly to their subtype classifications.

This architecture is defined inside the **`sustain_surrogate.json`** configuration file.

### The Chosen Variables
We configured the surrogate with the following architecture:

**Inputs (The "Conditioning" Variables):**
What information do we feed the model to make a prediction?
1. `region_{0..6}_zscore`: The continuous tau PET Z-scores across the 7 specific brain regions (e.g., Entorhinal, Amygdala, Parahippocampal). 

*Note: Clinical demographics like Age and APOE4 are implicitly marginalized out by this specific surrogate, allowing predictions based purely on spatial pathology.*

**Outputs (The Target Parameters):**
What is the model trying to predict? 

> **Probabilistic Output:** The surrogate model does *not* output a single deterministic classification. It learns the **full probability density function** of the MCMC. It outputs a probability distribution (thousands of samples) representing the model's confidence in the classification.

1. `prob_subtype_{0..S-1}`: The discrete marginal probability of the subject belonging to each of the discovered disease subtypes.
2. `expected_stage`: The continuous expected disease stage for the patient.

### Preparing the Data
The script **`build_training_data.py`** is responsible for creating the dataset. It dynamically opens the optimal MCMC `.pickle` file, extracts the denoised Z-scores (inputs) and the probability arrays (outputs), and saves them into a clean JSON file (`sustain_training_data.json`).

### Editing Variables
If you wish to expand the surrogate model with new variables (e.g., tracking a specific demographic feature like Age, or an aggregated "Total Tau" metric):
1. **Edit `build_training_data.py`**: Extract/calculate the new variable and append it as a key/value pair into `sample["inputs"]` or `sample["outputs"]`.
2. **Dynamic Preprocessing**: No changes needed to `preprocess_surrogate.py`! The script is fully dynamic. It inspects the JSON keys and automatically injects them into the `sustain_surrogate.json` config arrays before triggering training.
3. **Retrain**: Run extraction and preprocessing again.

---

## 2. Training the Surrogate

### How to Train
Once the data is ready, you can train the NPE surrogate model from scratch by running:
```bash
python preprocess_surrogate.py
```

### What Happens Under the Hood
The script programmatically triggers the external `bayesian_metamodeling` library to train a **Masked Autoregressive Flow (MAF)** neural network. 

The network looks at the 7 input Z-scores and learns to predict the complex probability distribution of the outputs. Instead of waiting hours for an MCMC algorithm to explore the probability space, the neural network learns the *exact mathematical shape* of that posterior distribution. 

> [!IMPORTANT]
> **Artifact Storage:** Once trained, the weights of this neural network are saved as an artifact in the `Surrogate_SuStaIn/tmp/surrogate_artifacts/<UNIQUE_HASH>/` directory in a file called `backend_payload.json`. **Always ensure you are loading the model from the most recently created hash folder** if you train multiple versions!

---

## 3. Using the Trained Surrogate (Sampling)

### How to Sample
To bypass the slow PySuStaIn MCMC sampling in production, you can load the trained surrogate for millisecond-level inference. This is demonstrated in `validate_test_subject.py`.

```python
import numpy as np
from bayesian_metamodeling.surrogates.backends import load_backend_model

# 1. Load the trained surrogate payload (Use the LATEST hash folder in tmp/)
model = load_backend_model('sbi_npe', 'tmp/surrogate_artifacts/10562c940a6c43d39851925cebb2cb49/backend_payload.json')

# 2. Define your new patient's regional Z-scores
new_patient_scan = {
    'region_0_zscore': np.array([2.5]),
    'region_1_zscore': np.array([2.0]),
    'region_2_zscore': np.array([0.4]),
    'region_3_zscore': np.array([0.2]),
    'region_4_zscore': np.array([0.1]),
    'region_5_zscore': np.array([0.0]),
    'region_6_zscore': np.array([0.3])
}

# 3. Draw fast probabilistic samples from the learned mapping
# Returns an array of shape [n_samples, batch_size, n_outputs]
samples = model.sample(new_patient_scan, n=1000, seed=42)[0]

# 4. Calculate expected values from the distribution
predicted_stage = np.mean(samples[:, -1])
predicted_subtype_0 = np.mean(samples[:, 0])

print(f"Predicted Stage: {predicted_stage:.2f}")
print(f"Prob Subtype 0: {predicted_subtype_0:.1%}")
```

### The Theory: How Does Sampling Work?
The saved surrogate model is the actual **trained weights of the Normalizing Flow neural network**. 

When you pass `new_patient_scan` into the `model.sample()` function:
1. The neural network processes the specific patient Z-scores.
2. It evaluates its layers to output the exact parameters that define a custom probability distribution tailored specifically to that patient's pathology.
3. The function then mathematically draws `n=1000` random samples from that newly generated distribution, giving you a full, mathematically rigorous probability landscape instantly.
