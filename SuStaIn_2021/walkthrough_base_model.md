# SuStaIn Base Model: Technical Walkthrough

This document explains the theoretical background of the SuStaIn model, as well as the technical details, parameters, and variable structures of the pySuStaIn Alzheimer's Disease progression implementation.

For high-level directory structure and requirements, refer to the main `README_SuStaIn.md`.

## Table of Contents
1. [Background: Alzheimer's Disease and Tau Trajectories](#1-background-alzheimers-disease-and-tau-trajectories)
2. [Why the Vogel et al. (2021) Paper Matters](#2-why-the-vogel-et-al-2021-paper-matters)
3. [How Trajectories are Inferred from Data](#3-how-trajectories-are-inferred-from-data)
4. [Simulation Example and Analysis](#4-simulation-example-and-analysis)
5. [Practical Model Implementation: Data & Trained Variables](#5-practical-model-implementation-data--trained-variables)

---

## 1. Background: Alzheimer's Disease and Tau Trajectories
Alzheimer's disease (AD) is characterized pathologically by two main protein aggregates in the brain: amyloid-beta plaques and tau tangles. While amyloid accumulation occurs early and diffusely, tau pathology is more closely tied to neurodegeneration and cognitive decline. 

Tau spreads through the brain in a spatiotemporal pattern, historically thought to be a single, uniform pathway (described by "Braak staging"). A "tau trajectory" refers to the progressive sequence in which tau pathology appears and intensifies across different brain regions over time. 

## 2. Why the Vogel et al. (2021) Paper Matters
In the Vogel et al. (2021) paper on Tau PET data, the researchers used the SuStaIn (Subtype and Stage Inference) algorithm to demonstrate that tau does NOT follow a single trajectory in all Alzheimer's patients. 

Instead, they discovered that there are multiple distinct spatiotemporal trajectories (subtypes) of tau progression. Some patients experience tau accumulation starting in the limbic system (classic), while others might show early cortical or temporal involvement. 

This matters tremendously because:
- It explains the clinical heterogeneity of Alzheimer's disease (why patients have different symptoms at different times).
- It allows for personalized medicine and better patient stratification in clinical trials.
- It provides a way to "stage" a patient based on their specific subtype using cross-sectional (single timepoint) brain scans, inferring their past and predicting their future progression.

## 3. How Trajectories are Inferred from Data
In the real world, longitudinal data (scanning the same patient over decades) is extremely rare and expensive. SuStaIn solves this by using *cross-sectional data*—a snapshot of many different patients at various random stages of the disease.

The model assumes that if a disease progresses in a specific sequence (e.g., Region A -> Region B -> Region C), then in a large population, we should frequently see patients with pathology in A, patients with A and B, and patients with A, B, and C. We should *rarely* see patients with C but not A. By applying machine learning (MCMC and EM clustering), SuStaIn uncovers these probabilistic sequences from noisy snapshots.

## 4. Simulation Example and Analysis
To understand this practically, we ran a simulation mimicking Tau PET biomarker data across 7 brain regions (Entorhinal, Amygdala, Parahippocampal, Fusiform, Inferior Temporal, Middle Temporal, Precuneus).

**What we did:**
We generated synthetic cross-sectional data for 500 subjects, explicitly embedding 2 distinct ground-truth "subtypes" of progression. We then passed this snapshot data into the pySuStaIn algorithm without telling it how many subtypes existed.

**What running the simulation means:**
Running the simulation means tasking the algorithm to:
1. Figure out the optimal number of trajectories (subtypes).
2. Determine the order of regional tau accumulation for each subtype.
3. Assign each of the 500 subjects to their most likely subtype and stage.

**Observations and Inference from the Simulated Data:**
1. **Optimal Number of Subtypes:** We used Cross-Validation Information Criterion (CVIC) to evaluate models with 1, 2, and 3 subtypes. The simulation results yielded the lowest CVIC score for the **3-subtype model**. This shows that despite explicitly embedding 2 ground-truth subtypes during data generation, the MCMC inference on this specific noisy 10k dataset mathematically favored a 3-trajectory fit to account for the variance.
2. **Positional Variance Diagrams (PVDs):** The generated PVD plots (e.g., `PVD_true.png`) visualize the uncertainty and ordering of the events. The x-axis represents the SuStaIn Stage (the progression timeline), and the y-axis represents the brain regions. The color intensity shows the confidence of an event happening at that stage.
3. **Subject Staging and Subtyping:** The output plots (`Stage_estimate_boxplots.png` and `Subtype_estimate_histograms.png`) show how accurately the model classified the 500 random subjects. A tight diagonal alignment in the stage boxplots indicates that the cross-sectional staging highly correlates with actual disease severity.

> [!WARNING]
> **What we CAN and CANNOT infer from simulations:**
> From the simulated data, we can validate that the SuStaIn algorithm is highly robust at untangling mixed, noisy cross-sectional data into distinct longitudinal sequences. However, because it's a simulation based on Z-scores, we cannot infer clinical truths (like actual tau biology). The simulation proves the *mathematical capability* of the inference engine, which Vogel et al. then applied to real-world biological data to make groundbreaking clinical discoveries.

---

## 5. Practical Model Implementation: Data & Trained Variables

This section explains the exact structure of the cached `.pickle` files, the priors applied, and how the posterior probabilities are computed and used in the model.

### 5.1 Simulated Data Pickle (`{dataset_name}_simulated_data.pickle`)
This file is generated on the first run of the script. It contains a Python dictionary with the following keys, preserving the exact synthetic dataset for consistency:
* **`data`** *(numpy array, shape `M x N`)*: Simulated continuous Z-score data for $M=500$ subjects across $N=7$ brain regions. These represent noisy cross-sectional tau measurements.
* **`data_denoised`** *(numpy array, shape `M x N`)*: The clean simulated Z-scores before measurement noise was added.
* **`stage_value`** *(numpy array, shape `M x 1`)*: The underlying progression stage assigned to each subject during data generation.
* **`ground_truth_sequences`** *(numpy array, shape `N_subtypes x N_stages`)*: The true spatiotemporal Z-score event progression sequence generated for each of the $2$ ground-truth subtypes ($N_{\text{stages}} = 21$).
* **`ground_truth_subtypes`** *(numpy array, shape `M`)*: Integers (`0` or `1`) representing the true subtype assignment of each subject.
* **`ground_truth_stages`** *(numpy array, shape `M`)*: Integers (`0` to `21`) representing the true disease stage of each subject.

### 5.2 Trained Model Pickle (`{dataset_name}_subtype{s}.pickle`)
This file contains the learned variables and estimated uncertainties for a fitted SuStaIn model with $s+1$ subtypes:
* **`samples_sequence`** *(numpy array, shape `N_subtypes x N_stages x MCMC_runs`)*: MCMC samples representing the event sequence (Z-score crossing order) for each subtype.
* **`samples_f`** *(numpy array, shape `N_subtypes x MCMC_runs`)*: MCMC samples representing the learned subtype prevalence fractions.
* **`samples_likelihood`** *(numpy array, shape `MCMC_runs`)*: Log-likelihood values for the model across all MCMC iterations.
* **`ml_subtype`** *(numpy array, shape `M x 1`)*: The most likely subtype assignment (`0` to `s`) for each subject.
* **`prob_ml_subtype`** *(numpy array, shape `M x 1`)*: The posterior probability of belonging to the assigned subtype.
* **`ml_stage`** *(numpy array, shape `M x 1`)*: The most likely disease stage (`0` to `21`) for each subject.
* **`prob_ml_stage`** *(numpy array, shape `M x 1`)*: The posterior probability of belonging to the assigned stage.
* **`prob_subtype`** *(numpy array, shape `M x N_subtypes`)*: The marginal posterior probability of each subtype for every subject.
* **`prob_stage`** *(numpy array, shape `M x N_stages + 1`)*: The marginal posterior probability of each disease stage (0 to 21) for every subject.
* **`prob_subtype_stage`** *(numpy array, shape `M x N_stages + 1 x N_subtypes`)*: The full **joint posterior probability distribution** over both subtype and stage for every subject.
* **`ml_sequence_EM`** *(numpy array, shape `N_subtypes x N_stages`)*: Maximum likelihood sequence of events from the Expectation-Maximization step.
* **`ml_f_EM`** *(numpy array, shape `N_subtypes`)*: Maximum likelihood subtype prevalence fractions from the EM step.

### 5.3 Priors & Posteriors
* **The Prior:**
  * **Stage Prior:** SuStaIn applies a **uniform prior over stages** ($P(\text{Stage } k \mid \text{Subtype } s) = \frac{1}{N_{\text{stages}} + 1}$). A subject is assumed a priori to be equally likely to be in any disease stage.
  * **Subtype Prior:** The prior probability of a subject belonging to subtype $s$ is the learned subtype prevalence fraction, $f_s$.
* **The Posterior:**
  * The **individual joint posterior probability** of belonging to subtype $s$ at stage $k$ given subject data $x$ is computed using Bayes' rule:
    $$P(s, k \mid x) = \frac{P(x \mid s, k) \cdot P(s)}{\sum_{s'} \sum_{k'} P(x \mid s', k') \cdot P(s')}$$
    where $P(x \mid s, k)$ is the likelihood of the subject's Z-scores under the linear model at stage $k$ of subtype $s$, and $P(s) = f_s$ is the subtype prevalence fraction.
  * These individual posteriors are averaged across all MCMC sequence samples $S$ and fractions $f$ to account for model uncertainty, and are saved directly in the **`prob_subtype_stage`** key inside the trained pickle.
  * **Marginal Posteriors** over subtypes alone (`prob_subtype`) or stages alone (`prob_stage`) are obtained by summing (marginalizing) the joint posterior over stages or subtypes, respectively.

### 5.4 Patient-Specific Prior Adjustment (Post-hoc Re-weighting)
If you have clinical or genetic prior knowledge about a new patient (e.g., they are highly likely to have a specific subtype or be in a late disease stage), you can **adjust the posteriors post-hoc** using Bayes' rule without modifying the library code. 

Multiply the default joint posterior by the ratio of the new prior to the default prior, and re-normalize:

$$\text{Posterior}_{\text{new}}(s, k \mid x) \propto \text{Posterior}_{\text{default}}(s, k \mid x) \cdot \frac{\text{Prior}_{\text{new}}(s, k)}{\text{Prior}_{\text{default}}(s, k)}$$

Where $\text{Prior}_{\text{default}}(s, k) = \frac{f_s}{N_{\text{stages}} + 1}$.

#### Implementation Recipe in Python:
```python
# 1. Run default classification to get the posteriors
_, _, _, _, _, _, prob_subtype_stage = sustain.subtype_and_stage_individuals_newData(
    new_patient_scan, opt_seq, opt_f, N_samples_estimate
)

# 2. Define your new prior matrix (shape: N_stages + 1 x N_subtypes)
P_new_prior = np.zeros((N_stages + 1, N_subtypes))
P_new_prior[15:22, 0] = 0.8 / 7  # High prior for late stages (15-21) of Subtype 1
P_new_prior += 0.2 / ((N_stages + 1) * N_subtypes)  # Small safety epsilon for other states

# 3. Calculate default prior matrix (Uniform stage prior * Population fractions)
f_population = np.mean(opt_f, axis=1)
P_old_prior = f_population[None, :] / float(N_stages + 1)

# 4. Re-weight the joint posterior for the first subject (index 0)
patient_posterior = prob_subtype_stage[0, :, :]
updated_posterior = patient_posterior * (P_new_prior / P_old_prior)
updated_posterior /= np.sum(updated_posterior)  # Re-normalize

# 5. Extract updated subtype and stage assignments
new_subtype_idx = np.argmax(np.sum(updated_posterior, axis=0))
new_stage_idx = np.argmax(updated_posterior[:, new_subtype_idx])
```

### 5.5 Code Examples: Training & Inference

Below are brief practical code snippets demonstrating how the base model is trained and how it is used to sample outputs for a new patient.

**Training the Model:**
```python
# Initialize the SuStaIn model object with your cross-sectional data
sustain = ZscoreSustain(
    data, Z_vals, Z_max, BiomarkerNames, 
    N_startpoints=25, N_S_max=3, N_iterations_MCMC=10000, 
    output_folder=output_folder, dataset_name=dataset_name, use_parallel_startpoints=False
)

# Run the MCMC inference algorithm to discover subtypes and stages
samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm()
```

**Classifying a New Patient:**
Once the model is trained, you can classify new unseen patients without re-running the heavy MCMC algorithm. You simply use the optimal parameter matrices (`opt_seq` and `opt_f`) loaded from the trained `.pickle` file.

```python
# 1. Define new patient's regional Tau Z-scores (must match the model's region order)
# Example: ['Entorhinal', 'Amygdala', 'Parahippocampal', 'Fusiform', 'Inferior Temporal', 'Middle Temporal', 'Precuneus']
new_patient_scan = np.array([[2.5, 2.0, 0.4, 0.2, 0.1, 0.0, 0.3]])

# 2. Run inference to classify the new patient
N_samples_estimate = 1000  # Number of samples to draw from the MCMC sequence for stability
ml_sub_new, prob_ml_sub_new, ml_stg_new, prob_ml_stg_new, prob_sub_new, prob_stg_new, prob_sub_stg_new = sustain.subtype_and_stage_individuals_newData(
    new_patient_scan, 
    opt_seq,  # Loaded from pickle 'samples_sequence'
    opt_f,    # Loaded from pickle 'samples_f'
    N_samples_estimate
)

# prob_sub_new contains the posterior probability distribution of the patient's subtype
# ml_stg_new contains the most likely predicted stage
```
