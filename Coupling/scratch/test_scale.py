import os
import json
from pathlib import Path
import sys

sys.path.insert(0, "C:/Project/metamodeler_codex_scaffold_docs-develop/src")
from bayesian_metamodeling.tutorial import run_mm_cli

def main():
    spec = {
      "schema_version": "1.0",
      "name": "test_affine_scale",
      "ppl_backend": "pymc",
      "surrogate_refs": [
        "C:/Project/ode_2023/ad_ode_surrogate.json",
        "C:/Project/Simulating_SuStaIn/Surrogate_SuStaIn/sustain_surrogate.json"
      ],
      "variables": [
        { "name": "age_baseline", "type": "scalar" },
        { "name": "apoe4_status", "type": "scalar" },
        { "name": "amyloid_5yr", "type": "scalar" },
        { "name": "tau_5yr", "type": "scalar" },
        { "name": "tau_self_dynamic", "type": "scalar" },
        { "name": "amyloid_self_dynamic", "type": "scalar" },
        { "name": "amyloid_drive_tau", "type": "scalar" },
        { "name": "memory_cognitive_test_result", "type": "scalar" },
        { "name": "clinical_stage", "type": "scalar" },
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
        { "variable": "age_baseline", "distribution": { "kind": "normal", "loc": 60.0, "scale": 1.0 } },
        { "variable": "apoe4_status", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "amyloid_5yr", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "tau_5yr", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_0_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_1_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_2_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_3_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_4_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_5_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } },
        { "variable": "region_6_zscore", "distribution": { "kind": "normal", "loc": 0.0, "scale": 1.0 } }
      ],
      "couplings": [
        {
          "kind": "deterministic",
          "source": "region_0_zscore,region_1_zscore,region_2_zscore,region_3_zscore,region_4_zscore,region_5_zscore,region_6_zscore",
          "target": "suStIn_global_tau",
          "transform": { "kind": "sum" }
        },
        {
          "kind": "gaussian_link",
          "source": "tau_5yr,tau_self_dynamic",
          "target": "suStIn_global_tau",
          "transform": { "kind": "integration", "dt": 2.0, "alpha": 0.02, "beta": 0.0 },
          "sigma": 0.15
        }
      ]
    }
    
    spec_path = Path("C:/Project/Coupling/temp_test_spec.json")
    spec_path.write_text(json.dumps(spec, indent=2))
    
    print("Building model...")
    run_mm_cli('meta', 'build', str(spec_path).replace('\\', '/'))
    print("Done building model!")

if __name__ == '__main__':
    main()
