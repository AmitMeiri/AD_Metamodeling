# Surrogate Model Walkthrough: Training and Sampling

This document explains how we built the specific surrogate model for this project, how to train it, and how to sample from it.

---

## 0. What is this Surrogate Model?

The **Surrogate Model** is a fast, neural network-based approximation of the full, computationally heavy **Ordinary Differential Equation (ODE) Biomarker Model**. 

The original ODE model uses Hamiltonian Monte Carlo (HMC) to simulate the continuous trajectories of Alzheimer's biomarkers over time based on complex matrix exponentials. While highly accurate, running full ODE simulations for individual patients is extremely slow. 
The goal of this surrogate is to "learn" the mathematical behavior of the ODE model. By training on thousands of pre-simulated ODE trajectories, the surrogate can instantly predict the underlying biological dynamics and future clinical outcomes for a new patient, effectively replacing the slow ODE solver in a clinical or real-time setting.

---

## 1. How Our Specific Model Was Built

Before training the surrogate, we had to define exactly what the neural network should learn to predict. The goal was to map a patient's baseline characteristics and early biomarker forecasts to their underlying biological dynamics and clinical outcomes.

This architecture is defined inside the **`ad_ode_surrogate.json`** configuration file.

### The Chosen Variables
We configured the surrogate with the following architecture:

**Inputs (The "Conditioning" Variables):**
What information do we feed the model to make a prediction?
1. `age_baseline`: The patient's age at the time of the first (baseline) scan.
2. `apoe4_status`: Genetic risk factor (0 = non-carrier, 1 = carrier).
3. `amyloid_baseline`: The patient's current Amyloid load at baseline.
4. `tau_baseline`: The patient's current Tau load at baseline.
5. `amyloid_2yr`: The Amyloid load 2 years from baseline.
6. `tau_2yr`: The Tau load 2 years from baseline.

> [!NOTE]
> **Clinical Utility:** While we use the slow ODE model to *simulate* these 2-year values during the training phase, the ultimate goal is to use **real patient scans** (taken 2 years apart) in a clinical setting. This means doctors won't need to simulate anything—they just input the real scan values, and the surrogate instantly predicts the underlying biological dynamics and 5-year clinical outcome.

**Outputs (The Target Parameters):**
What is the model trying to predict? 

> **Probabilistic Output:** The surrogate model does *not* output a single deterministic value for these parameters. Instead, it outputs a **full probability distribution** (thousands of samples) for each parameter. This allows us to calculate confidence intervals, probabilities, and measure clinical uncertainty.

1. `amyloid_self_dynamic`: The underlying Amyloid accumulation rate (a general scalar from the $v_{00}$ matrix index, independent of time steps).
2. `tau_self_dynamic`: The underlying Tau accumulation rate (a general scalar from the $v_{11}$ matrix index, independent of time steps).
3. `amyloid_drive_tau`: The underlying interaction effect of Amyloid driving Tau (a general scalar from the $v_{10}$ matrix index, independent of time steps).
4. `memory_result_baseline`: The patient's continuous memory decline score at baseline.
5. `clinical_stage_baseline`: The patient's continuous expected clinical stage score at baseline.
6. `memory_result_yr5`: A specific 5-year forecast of the patient's continuous memory decline score.
7. `clinical_stage_yr5`: A specific 5-year forecast of the patient's continuous expected clinical stage score [0.0 - 2.0].

### Preparing the Data
The script **`build_training_data.py`** is responsible for creating the dataset. It opens the raw, massive `ad_ode_demo_posterior.pkl` file, simulates 4,000 patients, extracts exactly the 13 variables listed above using the matrix indices, and saves them into a clean JSON file (`ad_ode_training_data.json`).

---

## 2. Training the Surrogate

### How to Train
Once the data and JSON configuration are ready, you can train a new surrogate model from scratch by running:
```bash
python preprocess_surrogate.py
```

### What Happens Under the Hood
When you run `preprocess_surrogate.py`, the script programmatically triggers the external `bayesian_metamodeling` library using a subprocess call:
```python
cmd = ["bayesmm", "surrogate", "fit", "ad_ode_surrogate.json"]
subprocess.run(cmd, ...)
```
This tells the private `bayesmm` CLI to look at our `ad_ode_surrogate.json` configuration, locate the 4,000-patient dataset we built, and begin fitting the model.

### The Theory: How NPE Training Works
Simulation-Based Inference (SBI) using **Neural Posterior Estimation (NPE)** is a technique used to replace incredibly slow mathematical simulations (like Hamiltonian Monte Carlo on differential equations). 

During training, we are not just fitting a simple curve. We are training a **Normalizing Flow Neural Network** (specifically a Masked Autoregressive Flow). 
The network looks at the 4 inputs and learns to predict the complex, multi-dimensional probability distribution of the 5 outputs. 

Instead of waiting hours for an MCMC algorithm to explore the probability space, the neural network learns the *exact mathematical shape* of that posterior distribution. Once trained, the weights of this neural network are saved as an artifact in the `tmp/surrogate_artifacts/` directory.

---

## 3. Using the Trained Surrogate (Sampling)

### How to Sample
You can see sampling in action by running either of the validation scripts:
```bash
python validate_ode_subject.py
python plot_surrogate_figures.py
```

### What Happens Under the Hood
In both scripts, we first load the trained network into memory using a function from the external metamodeling library:
```python
from bayesian_metamodeling.surrogates.backends import load_backend_model
model = load_backend_model(payload_path)
```
Once the model is loaded, we can ask it for predictions by passing a dictionary of patient inputs into the `sample` function:
```python
x_input = {
    'age_baseline': np.array([65]),
    'apoe4_status': np.array([0]),
    'amyloid_baseline': np.array([0.5]),
    'tau_baseline': np.array([1.1]),
    'amyloid_2yr': np.array([0.7]),
    'tau_2yr': np.array([1.4])
}
# Draw 4,000 probabilistic samples instantly
samples_surr = model.sample(x_input, n=4000, seed=42)[0]
```

### The Theory: What is the Saved Model and How Does Sampling Work?
A common misconception is that a surrogate model is just a giant lookup table of saved probabilities. **This is not the case.** 

The saved surrogate model is the actual **trained weights of the Normalizing Flow neural network**. 
*(Note: The `model.sample()` function is a method provided directly by the external `bayesian_metamodeling` library's backend model class).*

When you pass `x_input` into the `model.sample()` function:
1. The neural network processes your specific patient inputs.
2. It evaluates its layers to output the exact parameters (means, variances, covariances) that define a custom probability distribution tailored specifically to that patient.
3. The function then mathematically draws `n=4000` random samples from that newly generated distribution.

This acts as a dynamic function that instantly returns a full, mathematically rigorous probability landscape for any arbitrary combination of patient inputs.
