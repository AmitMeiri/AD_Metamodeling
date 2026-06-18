# Bayesian Metamodeling of Alzheimer's Disease Progression

This repository contains the codebase and verification framework for the Alzheimer's Disease (AD) Metamodeling project. 

*   **Student**: Amit Meiri
*   **Advisors & Mentors**: Prof. Shahar Arzy and Dr. Barak Raveh
*   **Institution**: The Hebrew University of Jerusalem (HUJI)

---

## 1. Project Overview

This project implements a **Bayesian Metamodeling** framework to integrate separate biological and clinical progression models of Alzheimer's Disease. Rather than trying to build a single massive model from scratch, metamodeling allows us to probabilistically link independent, pre-existing models:
1.  **Temporal Disease Decline Model (AD-ODE)**: Models the temporal dynamics, kinetic rates, and clinical decline stages of biomarkers.
2.  **Spatial Progression Model (SuStaIn)**: Models the spatiotemporal progression and subtype sequences of tau accumulation across the brain.

The **`Coupling`** directory acts as the central heart of this project. It defines the joint probabilistic constraints (coupling links) that mathematically bind these models together, allowing them to share information, resolve conflicts, and collapse prediction uncertainty during joint MCMC sampling.

> [!CAUTION]
> **Baseline Sampling Engine Limitation (Post-Draw Approximation)**:
> In the framework's baseline sampler (`bayesian_metamodeling/meta/sampling.py`), coupling is **not** resolved as a true joint posterior. Instead, the engine draws independent random samples from the source priors, runs them forward through the transform, and then **completely overwrites the target variables** with the transformed value plus noise.
>
> Consequently, the target variable's own prior distributions and uncoupled surrogate predictions are **entirely ignored** during sampling. It functions as a forward simulation mapping rather than true joint probabilistic constraint resolution.

---

## 2. Project Directory Tree

Below is the conceptual layout of the project, highlighting the core components of each module:

```
AD_Metamodeling/
├── README.md                           # This root README file
├── requirements.txt                    # Full developer dependencies for base models
├── requirements_coupling_only.txt      # Lightweight dependencies for running the coupling
├── .gitignore                          # Ignored files configuration
│
├── ode_2023/                           # AD-ODE Temporal Model
│   ├── README.md                       # ODE model documentation & directionality guide
│   ├── ad_ode_demo_posterior.pkl       # Base model trained MCMC parameter matrix
│   ├── reproduce_figures.py            # Generates original paper figures from data
│   ├── test_indices.py                 # Utility to extract ODE parameters
│   └── Surrogate/                      # Neural network surrogate folder
│       ├── walkthrough.md              # Technical breakdown of NPE Surrogate
│       ├── ad_ode_surrogate.json       # Autoregressive flow architecture config
│       ├── build_training_data.py      # Extracts base model data for surrogate
│       ├── preprocess_surrogate.py     # Surrogate preprocessing & training runner
│       ├── validate_ode_subject.py     # Surrogate performance evaluation
│       └── tmp/surrogate_artifacts/    # Trained surrogate weights & payload
│
├── SuStaIn_2021/                       # SuStaIn Spatial Model
│   ├── README_SuStaIn.md               # SuStaIn model documentation & stage guide
│   ├── run_simulation.py               # Generates synthetic cross-sectional scans
│   ├── walkthrough_base_model.md       # Technical explanation of pySuStaIn MCMC
│   ├── sim_zscore_output/              # Simulation results, PVD plots, & pickle files
│   └── Surrogate_SuStaIn/              # Neural network surrogate folder
│       ├── walkthrough_surrogate_model.md # Technical breakdown of NPE Surrogate
│       ├── sustain_surrogate.json      # Autoregressive flow architecture config
│       ├── build_training_data.py      # Extracts base model data for surrogate
│       ├── preprocess_surrogate.py     # Surrogate training runner
│       ├── validate_test_subject.py    # Test subject validation script
│       └── tmp/surrogate_artifacts/    # Trained surrogate weights & payload
│
├── Coupling/                           # The Metamodeling Integration (Heart of Project)
│   ├── README.md                       # Structural design & math formula guide
│   ├── build_coupling.py               # Compiles joint coupling JSON schema
│   ├── evaluate_coupling_patient.py    # Main evaluation script for testing patients
│   ├── Coupling_Test.md                # Test specification detailing patient archetypes
│   ├── patient_coupling_evaluation.png # Uncoupled vs Coupled posterior comparison plot
│   └── coupling_specs/                 # Output folder for compiled specifications
│
└── Library_Changes/                    # External Metamodeling Library Customizations
    ├── external_library_changes.md     # Detailed unified diffs & explanation report
    └── src/                            # Copies of the modified Python source files
```

---

## 3. Installation & Running Guide (Environment Setup)

To run the coupling simulation and evaluation smoothly, you need to set up a Python environment. 

> [!IMPORTANT]
> **Environment Golden Rule:** You must install BOTH the project's dependencies AND the external metamodeling library into the **exact same single Python environment**. Do NOT create separate environments for them. The environment can be named anything you like.

Follow these simple steps from your terminal/command prompt:

### Step 1: Create and Activate a Single Environment
Create a clean Python 3.10+ virtual environment (using Conda, PyCharm, or Python's `venv`) and activate it.

### Step 2: Install Dependencies (Choose Your Track)
While your single environment is active, install the libraries based on your goal:

**Track A: I just want to run the coupling test & surrogates (Lightweight)**
If you are the end-user and just want to run the pre-trained metamodel forecasts, you don't need heavy compilers. Run:
```bash
pip install -r requirements_coupling_only.txt
```

**Track B: I am a developer and want to run the original base models (Heavy)**
If you want to re-run the heavy base ODE or SuStaIn MCMC models from scratch, run the full list:
```bash
pip install -r requirements.txt
```

### Step 3: Install the External Metamodeling Library
Now, install the custom `bayesian_metamodeling` framework directly into that **same** environment. Navigate to wherever you downloaded the external framework and install it in "editable" mode (`-e .`):
```bash
cd /path/to/metamodeler_codex_scaffold_docs-develop
pip install -e .
```
*   **CLI Interface**: Once installed, the library exposes the `bayesmm` CLI tool.
    *   **Compile a Model**:
        ```bash
        bayesmm meta build <path_to_spec_json>
        ```
    *   **Sample a Model**:
        ```bash
        bayesmm meta sample <path_to_spec_json> --draws 500 --tune 100 --chains 2
        ```


### 3.3 Dynamic Path Configuration (Safety Guide)
When running the evaluation scripts, they expect the `bayesian_metamodeling` framework to be installed in your environment (as done via `pip install -e .` in step 3.2). 

If you cloned the external library to a custom location and did not install it via pip, you must ensure your `PYTHONPATH` environment variable includes the `src/` directory of the cloned framework before running the scripts, for example:

**On Linux/macOS:**
```bash
export PYTHONPATH="/path/to/metamodeler_codex_scaffold_docs-develop/src:$PYTHONPATH"
```
**On Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "C:\path\to\metamodeler_codex_scaffold_docs-develop\src;" + $env:PYTHONPATH
```

---

## 4. External Library Customizations

Because the standard `bayesian_metamodeling` library did not originally support complex spatiotemporal dynamics or dynamic condition mapping, we introduced custom modifications directly to its core:

1.  **Multiple-Source & Multiple-Target Support**: Enabled linking multiple variables inside mathematical transforms and mapping to multiple target noise nodes.
2.  **`directional_potential` Link Type**: Added a new soft coupling constraint that applies directional rewards/penalties to PyMC log-probability nodes (functioning like a statistical "wind" to guide parameters).
3.  **Project-Specific Transforms**: Integrated custom mathematical mapping functions:
    *   `sustain_to_ode_stage`: Maps SuStaIn's $0\text{--}21$ spatial stage to the ODE's $0\text{--}2$ clinical stage using subtype-conditioned logistic curves.
    *   `clinical_subtype_scorer`: Calculates propensity scores from genetics, cognitive decline, and biological rate, outputting soft subtype priors via a temperature-scaled Softmax.
    *   `velocity_modifier_score`: Computes dynamic dot-product velocity weights.

### Provided Files & Declarations
*   All modifications are explicitly declared and mapped out in: [external_library_changes.md](AD_Metamodeling/Library_Changes/external_library_changes.md)
*   For ease of deployment and direct reference, the fully updated Python source files containing these changes are provided under: [Library_Changes/src/](AD_Metamodeling/Library_Changes/src)

---

## 5. Quick Start: Running a Patient Evaluation

To run the joint MCMC coupling simulation for our test subjects and compare their uncoupled vs. coupled posteriors:

1.  Navigate to the repository folder.
2.  Run the evaluation script:
    ```bash
    python C:/Project/AD_Metamodeling/Coupling/evaluate_coupling_patient.py
    ```
3.  The script will:
    *   Draw uncoupled samples from both surrogate models.
    *   Inject uncoupled statistics as priors into the spec template.
    *   Call `bayesmm` to build and sample the coupled model.
    *   Generate a detailed comparison plot at: [patient_coupling_evaluation.png](AD_Metamodeling/Coupling/patient_coupling_evaluation.png)
