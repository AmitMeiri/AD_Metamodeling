### 3. Joint Inference
To perform simultaneous, coherent reasoning across distinct biological scales, the framework compiles the individual surrogates (`SurrogateLikelihoodFactorIR`) and cross-model coupling constraints (`CouplingFactorIR`) into a unified Bayesian factor graph. 
#### 3.1. Probabilistic Factor Graph Formulation
Let $\boldsymbol{\theta}_1 \in \Theta_1$ denote the latent parameter space of the spatial subtype/stage model (SuStaIn: subtype probabilities $\mathbf{P} = [P_1, P_2, P_3, P_4]^T$ and continuous stage $S_1$), and let $\boldsymbol{\theta}_2 \in \Theta_2$ denote the latent parameter space of the temporal dynamics model (ODE: longitudinal tau progression velocity $v_\tau$, amyloid burden, and clinical cognitive stage $C_2$). Given observed multimodal patient data $\mathbf{x} = \{\mathbf{x}_1, \mathbf{x}_2\}$, the joint posterior over the full parameter space $\boldsymbol{\Theta} = (\boldsymbol{\theta}_1, \boldsymbol{\theta}_2)$ is factorized as:
$$
p(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2 \mid \mathbf{x}_1, \mathbf{x}_2) \propto p(\boldsymbol{\theta}_1) \, p(\boldsymbol{\theta}_2) \cdot \tilde{p}(\boldsymbol{\theta}_1 \mid \mathbf{x}_1) \, \tilde{p}(\boldsymbol{\theta}_2 \mid \mathbf{x}_2) \cdot \prod_{k=1}^K \Psi_k(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2)
$$
where:
1. **Priors $p(\boldsymbol{\theta}_1), p(\boldsymbol{\theta}_2)$**: Define baseline bounds and non-informative/informative distributions over the parameters of each sub-model.
2. **Surrogate Likelihood / Posterior Factors $\tilde{p}(\boldsymbol{\theta}_m \mid \mathbf{x}_m)$**: Neural Posterior Estimators (amortized neural network density estimators, NPE) trained independently on forward simulations of SuStaIn and the ODE model. These capture the uncoupled, data-driven marginal evidence from each imaging/clinical modality.
3. **Coupling Potential Factors $\Psi_k(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2)$**: Differentiable energetic penalties and directional alignment constraints that bridge the two sub-models:
   * **Gaussian Link Potentials**: Enforce concordance between overlapping physical or clinical concepts (e.g., mapping spatial SuStaIn stage $S_1$ via subtype-weighted logistic transformation $g(\mathbf{P}, S_1)$ to the ODE clinical stage $C_2$):
     $$
     \log \Psi_{\text{Gaussian}}(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2) = -\frac{1}{2\sigma_k^2} \left\| g(\boldsymbol{\theta}_1) - h(\boldsymbol{\theta}_2) \right\|^2
     $$
   * **Directional Potentials**: Inject asymmetric kinetic dependencies (e.g., accelerating ODE tau velocity $v_\tau$ proportional to the aggregate aggressiveness score $S(\mathbf{P}) = \sum_{i=1}^4 P_i W_i$ of the assigned subtype):
     $$
     \log \Psi_{\text{Directional}}(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2) = \frac{1}{\sigma_{\text{dir}}} \Big( v_\tau \cdot S(\mathbf{P}) \Big)
     $$
#### 3.2. Joint MCMC Sampling Scheme
Inference over the compiled Intermediate Representation (IR) is executed via a global Markov Chain Monte Carlo (MCMC) sampler (Metropolis-Hastings / NUTS compiled through the PyMC/NumPyro backend). 
At each iteration $t$, the sampler proposes a transition in the joint latent space:
$$
\boldsymbol{\Theta}^* = (\boldsymbol{\theta}_1^*, \boldsymbol{\theta}_2^*) \sim q(\boldsymbol{\Theta}^* \mid \boldsymbol{\Theta}^{(t-1)})
$$
The target joint log-posterior density $\log \mathcal{L}(\boldsymbol{\Theta}^*)$ is evaluated by summing the uncoupled log-densities from the neural surrogates and the active coupling energy terms:
$$
\log \mathcal{L}(\boldsymbol{\Theta}^*) = \sum_{m=1}^2 \log \tilde{p}(\boldsymbol{\theta}_m^* \mid \mathbf{x}_m) + \sum_{k=1}^K \log \Psi_k(\boldsymbol{\theta}_1^*, \boldsymbol{\theta}_2^*)
$$
The proposal is accepted with probability:
$$
\alpha(\boldsymbol{\Theta}^{(t-1)}, \boldsymbol{\Theta}^*) = \min\left(1, \; \frac{\mathcal{L}(\boldsymbol{\Theta}^*) \, q(\boldsymbol{\Theta}^{(t-1)} \mid \boldsymbol{\Theta}^*)}{\mathcal{L}(\boldsymbol{\Theta}^{(t-1)}) \, q(\boldsymbol{\Theta}^* \mid \boldsymbol{\Theta}^{(t-1)})}\right)
$$
This joint evaluation forces the sampler to reject parameter combinations that fit one modality in isolation but violate cross-modal biological consistency, thereby shrinking the posterior uncertainty of both models and producing mutually reconciled patient trajectories.