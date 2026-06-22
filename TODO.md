# Future Assignments

## 1. Implement Real Coupling Mechanism
We need to implement the mathematical link between the ODE and SuStaIn models. There are two primary architectural methods to achieve this:

**Method 1: Full Simulation-Based Inference (SBI) - The "Coupled Prior" Approach**
*   **Concept:** Bake the coupling directly into the neural network. We generate millions of synthetic patients by drawing parameters from both models, applying the coupling constraints (e.g., Gaussian link), and training a massive *joint* Normalizing Flow (MAF) on the coupled dataset.
    *   *Note on Soft Coupling:* To preserve the delicate "tug-of-war" ($\sigma$) without resorting to a harsh binary cutoff, we either use the constraint score as a **Sample Weight** during neural network training, or we use a fast **MCMC on the Prior** to generate the parameter pairs before running the simulators.
*   **Pros:** Inference for a new patient is instant (a single neural network forward pass).
*   **Cons:** Requires an massive upfront simulation and training phase. We must train a new, much larger network to learn the joint distribution $P(\theta_A, \theta_B \mid y_A, y_B)$.

**Method 2: MCMC over Independent Surrogates (Post-Hoc Correction) - *Current Path***
*   **Concept:** Build the logic in `sampling.py`. We use `metamodeler` to train independent MAF neural networks for the ODE and SuStaIn models. During inference for a new patient, we run MCMC (NUTS) to sample from the joint distribution defined by: $\text{MAF}_A(x_A) + \text{MAF}_B(x_B) + \text{LinkPenalty}(x_A, x_B)$.
*   **Pros:** No need to train a massive joint neural network. We reuse the independently trained surrogates and apply the mathematical constraint algebraically at inference time.
*   **Cons:** Requires running MCMC during patient inference, which takes a few minutes compared to a fraction of a second.

## 2. Hybrid Surrogate Training (Add Forward Pass Samples)
Augment the surrogate training dataset by generating more `(theta, x)` pairs. Instead of relying solely on the outputs of the MCMC (our current training samples), we need to randomly sample `theta` from a prior distribution and perform true forward simulations to generate `x`.

**Why is this good?**
Currently, our surrogate acts as a lightning-fast "MCMC Cloner." Because it only trained on existing patient data, it might fail or hallucinate if presented with an edge-case patient whose data falls far outside the original dataset. By randomly sampling from a broad prior and running the math forward, we teach the neural network the global "rules of physics" of the simulator. This guarantees the surrogate will perfectly replicate the mathematical model under *any* scenario, vastly improving its generalization and robustness.

**How to implement it:**
*   **For the ODE Model:** 
    Define a broad Uniform prior for your dynamic parameters (e.g., `tau_self_dynamic`, `amyloid_drive_tau`). Write a loop to sample thousands of random parameters from this prior, plug them into the base ODE mathematical solver (e.g., `scipy.integrate.odeint`) to simulate the resulting biomarker trajectories (`x`), and append these new `(theta, x)` pairs to `ad_ode_training_data.json`.
*   **For the SuStaIn Model:** 
    Do NOT use `generate_data_Zscore_sustain()` from `simfuncs.py`, as that only generates toy data. Instead, build a forward simulator based on the actual *learned* model outputs:
    1. Sample a full sequence matrix from `samples_sequence` in the trained `.pickle` file (this injects the model's structural MCMC uncertainty into the dataset).
    2. Randomly assign a single, discrete ground-truth `Subtype` and `Stage` ($\theta$).
    3. Look up the expected Z-score state for each brain region in the sampled matrix for that specific subtype/stage.
    4. Generate Z-scores for each region and add significant statistical measurement noise ($x$). 
    When the neural network trains on these noisy pairs, the overlapping data distributions will mathematically force the network to output perfectly calibrated mixed posterior probabilities for ambiguous cases. Append these pairs to `sustain_training_data.json`.
