# Coupling Verification Test

This document outlines the testing methodology and patient archetypes used to verify the mathematical behavior of the Bayesian metamodel coupling between the **AD-ODE Temporal Model** and the **SuStaIn Spatial Model**.

---

## 1. Coupling Architecture & Specification

The coupling is built programmatically using the script `build_coupling.py`. This script compiles the joint probabilistic specification and configures three distinct coupling links:
1. **Tau Dynamics Coupling**: Binds the ODE's continuous tau accumulation velocity ($v_{11}$) to SuStaIn's spatial subtype classification probabilities.
2. **Disease Stage Coupling**: Connects SuStaIn's spatial stage progression (`expected_stage` on a $0\text{--}21$ scale) to the ODE's clinical stage progression ($0.0\text{--}2.0$ scale, representing CN, MCI, and Dementia) using a subtype-conditioned logistic mapping.
3. **Subtype Prior Re-weighting**: Re-weights SuStaIn's subtype classification probabilities based on the patient's APOE4 carrier status and the ODE's calculated biological rate of tau progression.

The resulting compiled configuration is exported as a single JSON file: `ode_sustain_coupling.json`.

---

## 2. Verification Methodology

To test that the coupling forces appropriate mathematical compromise and mutual reinforcement, we use the script **`evaluate_coupling_patient.py`**. 

The verification procedure is as follows:
1. **Define Patients**: We construct synthetic patient profiles with specific conflict or alignment between their spatial scan (SuStaIn inputs) and temporal history (ODE inputs).
2. **Run Uncoupled Models**: We run the ODE and SuStaIn surrogate models independently, observing their probability distributions.
3. **Run Coupled Model**: We run the joint model, enabling the coupling terms to resolve the interaction between temporal and spatial markers.
4. **Compare Outputs**: We plot the uncoupled vs. coupled probability density curves for key parameters:
   * `tau_self_dynamic` (ODE biological velocity)
   * `memory_result_yr5` (ADAS-Cog memory score forecast)
   * `clinical_stage_yr5` (ODE Clinical Stage representing CN, MCI, Dementia)
   * `prob_subtype_0`, `prob_subtype_1`, `prob_subtype_2` (SuStaIn subtype classifications)
   * `expected_stage` (SuStaIn spatial stage severity)

---

## 3. Patient Test Profiles

We define two distinct patient profiles representing **Conflict (Subject 1)** and **Amplification (Subject 2)**.

### Subject 1: Fast-Progression Limbic Contradiction (Conflict Resolving)
This subject creates a direct contradiction between spatial presentation and temporal/genetic characteristics:
* **Inputs**:
  * **ODE (Fast Temporal Velocity)**: Age 60, APOE4 negative.
    * `amyloid`: baseline = `-0.5`, $2\text{-year} = -0.2$ (Healthy, low pathology).
    * `tau`: baseline = `0.0`, $2\text{-year} = 50.0$ (Very low absolute values, but a massively steep *rate* of growth).
  * **SuStaIn (Spatial Limbic Pattern)**:
    * `zscores` = `[2.81, 1.58, 3.47, 0.8, 2.2, 0.4, 1.5]` (Medial temporal/limbic regions [0, 1, 2] are high, and neocortical/non-limbic regions [3, 4, 5, 6] are kept low to isolate the Classic Limbic subtype).
* **Expected Uncoupled Behavior**:
  * **SuStaIn** predicts a high probability of Subtype 1 (Classic Limbic) due to the spatial signature.
  * **ODE** predicts a high `tau_self_dynamic` velocity due to the steep $0 \to 50$ growth, but forecasts a healthy clinical stage because amyloid is healthy and absolute tau is small.
* **Expected Coupled Behavior (The Shift)**:
  * The fast biological tau progression rate and APOE4-negative status strongly contradict the typical Limbic subtype profile (which biologically progresses slowly and is heavily associated with APOE4-positive status).
  * As a result, the coupling **reduces SuStaIn's confidence in the Limbic Subtype**, shifting probability back toward the Atypical subtypes.
  * Simultaneously, the ODE's clinical forecast is dragged worse by SuStaIn's spatial stage, correcting the ODE's naive "completely healthy" prediction.

### Subject 2: Neocortical Explosion (Mutual Alignment & Amplification)
This subject represents a catastrophic, perfectly aligned progression profile where spatial and temporal markers mutually reinforce each other:
* **Inputs**:
  * **ODE (Temporal Explosion)**: Age 65, APOE4 positive.
    * `amyloid`: baseline = `-0.98`, $2\text{-year} = 0.58$ (Increasing amyloid trajectory satisfying model constraints).
    * `tau`: baseline = `0.28`, $2\text{-year} = 260.0$ (High baseline tau, moderate 2-year velocity).
  * **SuStaIn (Atypical Spatial Saturation)**:
    * `zscores` = `[2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5]` (Maxed-out tau leaning heavily into the neocortex, strongly indicating Subtype 0 "Atypical").
* **Expected Uncoupled Behavior**:
  * **SuStaIn** predicts a near-100% probability of Subtype 0 (Atypical) at a late stage.
  * **ODE** predicts a clinical stage of `1.36` (representing mainly MCI, and only partially Dementia).
* **Expected Coupled Behavior (The Amplification)**:
  * Subtype 0 carries the maximum biological velocity modifier ($+1.0$).
  * The coupling takes the ODE's already extreme `tau_self_dynamic` and amplifies it further due to the high certainty of Subtype 0.
  * The variance of all parameters collapses into narrow, razor-sharp peaks, representing absolute certainty that the patient is on a rapid, catastrophic neurodegenerative trajectory.

---

## 4. Verification Results & Comparison Plot

The verification results and comparative plots are saved and accessible at:
* **Project Plot**: [patient_coupling_evaluation.png]

