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
├── requirements.txt                    # Project-wide dependencies
│
├── ode_2023/                           # AD-ODE Temporal Model
│   ├── README.md                       # ODE model documentation & directionality guide
│   ├── run_ode_surrogate.py            # Code to run ODE simulations & generate data
│   ├── walkthough_base_model.md        # Technical breakdown of ODE kinetics
│   └── Surrogate/                      # Neural network surrogate folder
│       ├── surrogate_config.json       # Autoregressive flow architecture
│       ├── preprocess_surrogate.py     # Surrogate preprocessing & training runner
│       ├── validate_surrogate.py       # Surrogate performance evaluation
│       └── tmp/surrogate_artifacts/    # Trained surrogate weights & payload
│
├── SuStaIn_2021/                       # SuStaIn Spatial Model
│   ├── README_SuStaIn.md               # SuStaIn model documentation & stage guide
│   ├── run_simulation.py               # Generates synthetic cross-sectional scans
│   ├── walkthrough_base_model.md       # Technical explanation of pySuStaIn MCMC
│   ├── sim_zscore_output/              # Simulation results, PVD plots, & pickle files
│   └── Surrogate_SuStaIn/              # Neural network surrogate folder
│       ├── sustain_surrogate.json      # autogressive flow surrogate layout
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

## 3. Requirements & System Dependencies

To run the coupling simulation and evaluation, the following environments and libraries must be installed:

### 3.1 Python Environment
*   **Python Version**: 3.10+ (Recommended: 3.10 or 3.11)
*   **Key Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `pymc`, `pytensor`, `sbi`, `pydantic` (V2).
*   *For a complete list of packages, install the root requirements file*:
    ```bash
    pip install -r requirements.txt
    ```

### 3.2 The Metamodeling Library (`bayesian_metamodeling`)
This project runs on top of the external framework library developed by Dr. Barak Raveh:
*   **Official Package Name**: `bayesian_metamodeling` (scaffold-docs framework)
*   **Installation**: The library must be installed in your environment in editable mode:
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
When running evaluation scripts directly via Python, ensure your `PYTHONPATH` includes the source directory of the framework so the code can run safely. 

The evaluation script [evaluate_coupling_patient.py](file:///C:/Project/AD_Metamodeling/Coupling/evaluate_coupling_patient.py) handles this automatically by dynamically prepending the repository path at startup:
```python
sys.path.insert(0, "C:/Project/metamodeler_codex_scaffold_docs-develop/src")
```
*Note: If you move the project or external library folder, update this path or ensure the environment variable `PYTHONPATH` is set.*

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
