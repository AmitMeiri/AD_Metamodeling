import json
import sys
from pathlib import Path

# Add the root directory of the bayesian metamodeling framework to the Python path 
# so we can import the command line interface (CLI) runner.
sys.path.insert(0, "C:/Project/metamodeler_codex_scaffold_docs-develop/src")
from bayesian_metamodeling.tutorial import run_mm_cli

# Helper to find the latest trained surrogate payload dynamically
def get_latest_model_path(base_dir):
    base = Path(base_dir)
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No trained surrogate model found in {base_dir}")
    subdirs.sort(key=lambda d: (d / 'artifact.json').stat().st_mtime if (d / 'artifact.json').exists() else 0)
    return subdirs[-1] / 'artifact.json'

ode_payload_path = get_latest_model_path("C:/Project/AD_Metamodeling/ode_2023/Surrogate/tmp/surrogate_artifacts")
sustain_payload_path = get_latest_model_path("C:/Project/AD_Metamodeling/SuStaIn_2021/Surrogate_SuStaIn/tmp/surrogate_artifacts")

# Define the integration timeframes (dt) in years to simulate disease progression.
# We will generate a separate coupled model for each dt.
dts = [5.0, 2.0, 10.0, 1.0, 0.5]
out_dir = Path("C:/Project/AD_Metamodeling/Coupling/coupling_specs")
out_dir.mkdir(parents=True, exist_ok=True)

# Define the base JSON specification for the Bayesian Metamodel.
# This schema dictates the architecture of the joint probability space.
spec = {
  "schema_version": "1.0",
  "name": "ode_sustain_coupling",
  "ppl_backend": "pymc", # Use PyMC as the Probabilistic Programming Language backend
  "surrogate_refs": [
    # Paths to the pre-trained neural networks (Neural Posterior Estimators).
    # These act as the frozen structural blocks of our joint model.
    str(ode_payload_path).replace('\\', '/'),
    str(sustain_payload_path).replace('\\', '/')
  ],
  "variables": [
    # Explicitly declare all input and output variables present across BOTH surrogates.
    # The compiler needs this to allocate PyMC tensor variables in the background graph.
    { "name": "age_baseline", "type": "scalar" },
    { "name": "apoe4_status", "type": "scalar" },
    { "name": "amyloid_baseline", "type": "scalar" },
    { "name": "tau_baseline", "type": "scalar" },
    { "name": "amyloid_2yr", "type": "scalar" },
    { "name": "tau_2yr", "type": "scalar" },
    { "name": "tau_self_dynamic", "type": "scalar" },
    { "name": "amyloid_self_dynamic", "type": "scalar" },
    { "name": "amyloid_drive_tau", "type": "scalar" },
    { "name": "memory_result_yr5", "type": "scalar" },
    { "name": "clinical_stage_yr5", "type": "scalar" },
    { "name": "region_0_zscore", "type": "scalar" },
    { "name": "region_1_zscore", "type": "scalar" },
    { "name": "region_2_zscore", "type": "scalar" },
    { "name": "region_3_zscore", "type": "scalar" },
    { "name": "region_4_zscore", "type": "scalar" },
    { "name": "region_5_zscore", "type": "scalar" },
    { "name": "region_6_zscore", "type": "scalar" },
    { "name": "prob_subtype_0", "type": "scalar" },
    { "name": "prob_subtype_1", "type": "scalar" },
    { "name": "prob_subtype_2", "type": "scalar" },
    { "name": "expected_stage", "type": "scalar" },
    { "name": "suStIn_global_tau", "type": "scalar" }
  ],
  "priors": [
    # Define the starting parameter distributions for the independent input variables.
    # We use a standard Normal distribution (loc = mean, scale = standard deviation).
    # During sampling, MCMC will explore the space starting from these distributions.
    { "variable": "age_baseline", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "apoe4_status", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "amyloid_baseline", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "tau_baseline", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "amyloid_2yr", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "tau_2yr", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "region_0_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "region_1_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "region_2_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "region_3_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "region_4_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "region_5_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
    { "variable": "region_6_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } }
  ],
  # Define the probabilistic "soft constraints" that link the two surrogates.
  # A gaussian_link forces Target ~ Normal(Transform(Source), sigma).
  # If the target and the transformed source diverge, MCMC applies a statistical penalty.
  "couplings": [
    {
      # Subtype-Conditioned Stage Mapping
      # Maps the SuStaIn continuous spatial event score (expected_stage 0-21)
      # to the ODE clinical severity scale (clinical_stage 0.0-2.0) [scale for CN,MCI,AD]
      # Different subtypes decline at different spatial burdens.
      "kind": "gaussian_link",
      "source": "expected_stage,prob_subtype_0,prob_subtype_1,prob_subtype_2",
      "target": "clinical_stage_yr5",
      "transform": { "kind": "sustain_to_ode_stage" },
      "sigma": 0.2
    },
    {
      "kind": "deterministic", # Direct equality constraint, no Gaussian penalty width
      "source": "region_0_zscore,region_1_zscore,region_2_zscore,region_3_zscore,region_4_zscore,region_5_zscore,region_6_zscore",
      "target": "suStIn_global_tau",
      "transform": { "kind": "sum" }
    },
    {
      # Subtype-Driven Velocity Potential
      # Directionally pushes the ODE's intrinsic velocity based on the spatial pathway.
      # S = (P_0 * 1.0) + (P_1 * -0.3) + (P_2 * 0.5)
      # Log-Prob Bonus = sigma * tau_self_dynamic * S
      "kind": "directional_potential",
      "source": "prob_subtype_0,prob_subtype_1,prob_subtype_2",
      "target": "tau_self_dynamic",
      "transform": { "kind": "velocity_modifier_score", "weights": [1.0, -0.3, 0.5] },
      "sigma": 0.5
    },
    {
      # Clinical Subtype Prior
      # Evaluates APOE4 status, tau velocity, and memory impairment to generate a Softmax prior.
      # Biologically anchors the SuStaIn subtype probabilities to the patient's temporal clinical severity.
      "kind": "gaussian_link",
      "source": "apoe4_status,tau_self_dynamic,tau_baseline,memory_result_yr5",
      "target": "prob_subtype_0,prob_subtype_1,prob_subtype_2",
      "transform": { "kind": "clinical_subtype_scorer", "beta": 1.0 },
      "sigma": 0.25
    }
  ]
}

# Write the specific configuration to disk
spec_path = out_dir / "metamodel_coupling.json"
spec_path.write_text(json.dumps(spec, indent=2))

# We use forward slashes for the CLI call to avoid escape character issues in Windows
spec_path_str = str(spec_path).replace('\\', '/')

print("\n--- Running build ---")
run_mm_cli('meta', 'build', spec_path_str)

print("\n--- Running sample ---")
run_mm_cli('meta', 'sample', spec_path_str, '--draws', '10', '--tune', '5', '--chains', '2', '--seed', '123')

print("\nCoupling model successfully generated, built, and sampled.")
