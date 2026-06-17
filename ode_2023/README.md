# AD ODE 2023 Metamodeling & Surrogate Pipeline

This directory contains the continuous-time Ordinary Differential Equation (ODE) model of Alzheimer's Disease progression, originally introduced in the paper:
> **"A personalized continuous-time model of Alzheimer's disease progression"** (Bossa & Sahli, 2023).  
> [Nature Scientific Reports - Article Link](https://www.nature.com/articles/s41598-023-29383-5#data-availability)

Alongside the original model code and cached posteriors, this directory includes a **Surrogate Model** implementation designed to replace computationally expensive Monte Carlo simulations with an instant, probability-based Neural Posterior Estimator.

## About the Model

The original ODE model uses longitudinal patient data to map the continuous-time cascade of Alzheimer's disease biomarkers via differential equations. 

**How the model works:**
1. **Training (Learning the Equations):** During the initial training phase (via MCMC), the model computes a probability distribution for every coefficient in the differential equations. Each coefficient represents exactly how one biological variable influences another.
2. **Causal Constraints:** The researchers explicitly built biological facts into these equations. For example, the model dictates that Amyloid Beta drives Tau pathology, and Tau pathology drives Memory decline. Importantly, it enforces that the reverse is not true (e.g., worsening memory does *not* cause Tau to accumulate; the coefficient for that is constrained to zero).
3. **Personalization:** Once the population-level coefficient distributions are established, the model can take a specific patient's current data and fit an updated, personalized parameter distribution just for them. 
4. **Prediction:** *Only after* calculating these personalized differential equation parameters can the model simulate the patient's trajectory forward in time to output forecasts for clinical classification and future biomarkers.

## Model Prediction Outputs & Clinical Staging

At any given future forecast time step (e.g., a 5-year forecast), the ODE model generates two primary types of outputs:
1. **Biomarker Forecasts (Continuous)**: Predictions of the patient's continuous physical state vector $x(t)$, which models 5 latent dimensions: Amyloid-Beta ($A\beta$), Tau, and three cognitive factors (Language, Memory, and Praxis).
2. **Clinical Staging (Ordered Logistic GLM)**: Rather than relying on a single memory score, the original model determines clinical diagnosis using an **ordered logistic Generalized Linear Model (GLM)** based on the entire cognitive and biomarker state.

### Mathematical Formulation
The linear predictor $\eta$ represents the patient's overall disease load, calculated as a weighted combination of their biomarker values:
$$\eta = \sum_{j=1}^{5} x_j(t) \cdot \beta_{j}$$
Using this linear predictor, the model calculates the probability of the patient belonging to each of three ordered clinical categories (1 = Cognitively Normal, 2 = Mild Cognitive Impairment, 3 = Dementia/AD) using cutpoint thresholds $c_1$ and $c_2$:
- **Probability of Cognitively Normal (CN)**:
  $$P(CN) = \text{sigmoid}(c_1 - \eta)$$
- **Probability of Mild Cognitive Impairment (MCI)**:
  $$P(MCI) = \text{sigmoid}(c_2 - \eta) - \text{sigmoid}(c_1 - \eta)$$
- **Probability of Dementia (AD)**:
  $$P(AD) = 1 - \text{sigmoid}(c_2 - \eta) = \text{sigmoid}(\eta - c_2)$$

### Continuous Representation
To represent this probabilistic classification as a continuous variable suitable for normalizing flows and regression links, we compute the **Expected Clinical Stage Score** ($CS$) in the range $[0.0, 2.0]$:
$$CS = 0 \cdot P(CN) + 1 \cdot P(MCI) + 2 \cdot P(AD) = P(MCI) + 2 \cdot P(AD)$$
This expected score smoothly maps the patient's condition to clinical categories:
- $[0.0, 0.5]$: Cognitively Normal (CN)
- $(0.5, 1.5]$: Mild Cognitive Impairment (MCI)
- $(1.5, 2.0]$: Dementia (AD)

## Directory Overview

This folder contains three distinct sets of files:
1. **The Base ODE Scripts:** Original files to run and simulate the ODE model on data.
2. **The Base ODE Artifacts:** Pre-calculated posterior parameter matrices from rigorous Markov Chain Monte Carlo (MCMC) training, meaning the heavy computational lifting is already done.
3. **The Surrogate Module (`Surrogate/`):** A sub-directory dedicated to **Surrogate Modeling**. Because simulating the full ODE using standard MCMC takes hours or days, we have built a Neural Posterior Estimation (NPE) surrogate. This neural network acts as a drop-in replacement: instead of explicitly calculating differential equations, it instantly outputs the exact mathematical probability distribution of a patient's future state, achieving the same fidelity in milliseconds.

## File & Folder Map

### Base ODE Files
* `ad_ode_demo_posterior.pkl`: The cached, pre-trained MCMC posterior parameters of the model.
* `ad_ode_model.stan`: The mathematical core of the original ODE model, programmed in Stan.
* `reproduce_figures.py`: Generates the figures from the original 2023 paper using simulated data.
* `test_indices.py`: A utility script for testing and understanding how to extract biological parameters from the raw matrix posterior.

### Surrogate Files (`Surrogate/` directory)
* **[`walkthrough.md`](Surrogate/walkthrough.md)**: **Start here!** A detailed guide explaining the theory behind Neural Posterior Estimation, how to train the surrogate, and how to sample from it.
* `build_training_data.py`: Extracts and prepares data from the original ODE posterior into a JSON format suitable for neural network training.
* `ad_ode_training_data.json`: The resulting 4,000-sample dataset generated by the script above, mapping inputs (like Age and Genetics) to target outputs.
* `check_surrogate.py`: A quick-check utility script for validating output shapes and generating basic summary statistics from the surrogate.
* `ad_ode_surrogate.json`: The configuration metadata that maps inputs, outputs, and hyperparameters for the surrogate trainer.
* `preprocess_surrogate.py`: The main trainer script. It consumes the data and config, invokes the metamodeling library, and fits the flow-based NPE surrogate.
* `tmp/surrogate_artifacts/`: **This is where the trained surrogate model is saved!** The `bayesmm` library natively caches the trained Neural Posterior Estimator (NPE) networks here. Validation scripts will automatically look here to load the model.
* `validate_ode_subject.py`: Validates the trained surrogate (loaded from `tmp/`) by tracking simulated patient archetypes over a 15-year forecast and comparing clinical conversion probabilities (MCI/Dementia) against the original ODE.
* `plot_surrogate_figures.py`: Generates publication-ready 1D and 2D density plots proving the statistical correlation accuracy of the surrogate.
* `surrogate_ode_plots/`: The output directory where the validation and plotting scripts save all of their final generated PNG figures.

## The Biomarker Parameter Matrix ($v$)

If you wish to extract different variables to train a new surrogate, you must understand the underlying 5x5 interaction matrix $v$ found in `ad_ode_demo_posterior.pkl`. The model tracks 5 latent variables, where $v[\text{row}, \text{column}]$ represents the influence of the column on the row.

**Note on Variable Directionality:** All 5 latent variables track *disease burden* (so that a **higher** value represents **worse** pathology/impairment):
* **Cognitive Domains (Index 2, 3, 4)**: A higher continuous value naturally indicates worse cognitive impairment (e.g., higher error scores on ADAS-Cog test components).
* **Tau (Index 1)**: A higher value represents worse neurodegeneration (as cell damage releases Tau into the CSF).
* **Amyloid-Beta ($A\beta$) (Index 0)**:
  * In raw clinical assays, a **higher** CSF level is actually **healthy/good** (meaning $A\beta_{1-42}$ remains soluble in the CSF and is not building up as plaques in the brain; a low CSF level is abnormal).
  * However, for the mathematical equations of the ODE model, the raw inputs are preprocessed via a decreasing transformation that measures the **distance from normal values** (mapping healthy values $\ge 1700\text{ pg/mL}$ to $0.0$, and lower, pathological values closer to $1.0$).
  * Therefore, in the ODE state vector $x_{A\beta}$ and parameter matrix $v$, a **higher** value represents **greater pathology severity** (worse disease burden).

| Index | Biomarker / Domain |
| :---: | :--- |
| **0** | Amyloid Beta ($A\beta$) |
| **1** | Tau |
| **2** | Cognitive Domain 1 (e.g., Language) |
| **3** | Cognitive Domain 2 (ADAS-Cog Memory Factor) |
| **4** | Cognitive Domain 3 (e.g., Praxis) |


**Examples of extracting interactions:**
* `v[:, 0, 0]` = Amyloid self-dynamics
* `v[:, 1, 1]` = Tau self-dynamics
* `v[:, 1, 0]` = Amyloid driving Tau accumulation
* `v[:, 3, 1]` = Tau driving Memory decline

## Requirements & Prerequisites

To run this repository successfully, please install the public dependencies listed in `requirements.txt`. You can install them globally or in a virtual environment using:
```bash
pip install -r requirements.txt
```

### Dependency Overview:
* **Base ODE:** Requires `cmdstanpy` (and a working Stan C++ compiler backend), `scipy`, `pandas`, `numpy`, `matplotlib`, and `seaborn`.
* **Surrogate Model (Private Library):** The surrogate scripts require the `bayesian_metamodeling` library to be installed in your Python environment. Please note that this is a private/internal library and is *not* available via public PIP. You must ensure it is installed manually from your internal source before running the surrogate scripts.

*(Note: The surrogate model natively stores its trained artifacts within the `Surrogate/tmp` folder to comply with the bayesmm framework defaults. Do not manually move these folders, as the validation scripts are mapped to read directly from them.)*
