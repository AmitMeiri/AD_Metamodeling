# External Metamodeling Library Modifications

> [!IMPORTANT]
> **Declaration of Modifications & Provision of Files**:
> This document explicitly declares all modifications made to the external **`bayesian_metamodeling`** framework library. For convenience and ease of deployment, the fully updated and modified Python source files containing these additions are provided directly in the [Library_Changes/src](file:///C:/Project/AD_Metamodeling/Library_Changes/src) directory.

This document details the modifications made to the external framework library `bayesian_metamodeling` stored in `C:\Project\metamodeler_codex_scaffold_docs-develop` compared to the original, pristine version in `C:\Project\OG_metamodeler_codex\metamodeler_codex_scaffold_docs-develop`.

## Summary of Changes

- **New Files**: 0
- **Deleted Files**: 0
- **Modified Files**: 6

## Detailed Code Changes

### [MODIFY] `src\bayesian_metamodeling\meta\builder.py`

```diff
--- original/src\bayesian_metamodeling\meta\builder.py
+++ modified/src\bayesian_metamodeling\meta\builder.py
@@ -68,8 +68,8 @@
     for ref in spec.surrogate_refs:
         surrogate_id, artifact = _resolve_surrogate_ref(ref)
         var_lists = artifact.get("variable_lists", {})
-        inputs = list(var_lists.get("inputs", []))
-        outputs = list(var_lists.get("outputs", []))
+        inputs = list(var_lists.get("inputs", artifact.get("inputs", [])))
+        outputs = list(var_lists.get("outputs", artifact.get("outputs", [])))
         if not inputs or not outputs:
             raise ValueError(f"Surrogate artifact missing variable lists: {ref}")
```

### [MODIFY] `src\bayesian_metamodeling\meta\compiler.py`

```diff
--- original/src\bayesian_metamodeling\meta\compiler.py
+++ modified/src\bayesian_metamodeling\meta\compiler.py
@@ -39,8 +39,20 @@
                     var = scale**2
                     total += -0.5 * (np.log(2 * np.pi * var) + ((x - loc) ** 2) / var)
             elif isinstance(factor, CouplingFactorIR):
-                source = float(values[factor.source])
-                target = float(values[factor.target])
+                if "," in factor.source:
+                    sources = [float(values[s.strip()]) for s in factor.source.split(",")]
+                    source = sources[0]
+                else:
+                    sources = [float(values[factor.source])]
+                    source = sources[0]
+                    
+                if "," in factor.target:
+                    targets = np.array([float(values[t.strip()]) for t in factor.target.split(",")])
+                    target = targets[0]
+                else:
+                    targets = np.array([float(values[factor.target])])
+                    target = targets[0]
+
                 relation = factor.transform.get("kind", "identity")
                 if relation == "identity":
                     transformed = source
@@ -48,20 +60,73 @@
                     alpha = float(factor.transform.get("alpha", 1.0))
                     beta = float(factor.transform.get("beta", 0.0))
                     transformed = alpha * source + beta
+                elif relation == "sum":
+                    transformed = sum(sources)
+                elif relation == "integration":
+                    dt = float(factor.transform.get("dt", 1.0))
+                    alpha = float(factor.transform.get("alpha", 1.0))
+                    beta = float(factor.transform.get("beta", 0.0))
+                    transformed = alpha * (sources[0] + sources[1] * dt) + beta
+                elif relation == "subtype_conditioned_stage":
+                    x = sources[0]
+                    p0 = sources[1]
+                    p1 = sources[2]
+                    p2 = sources[3]
+                    
+                    c_limbic = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.5))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 1.2))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.8)))
+                    c_atyp = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.2))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 0.6))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.5)))
+                    
+                    transformed = (p1 * c_limbic) + ((p0 + p2) * c_atyp)
+                elif relation == "sustain_to_ode_stage":
+                    x = sources[0]
+                    p0 = sources[1]
+                    p1 = sources[2]
+                    p2 = sources[3]
+                    
+                    # DISCLAIMER: The midpoint anchors (10.0 for Limbic, 15.0 for Atypical) are highly 
+                    # dependent on the total number of SuStaIn stages and regions (here assumed 21).
+                    # If the number of stages changes, these anchors MUST be recalibrated.
+                    c_limbic = 2.0 / (1.0 + np.exp(-0.4 * (x - 10.0)))
+                    c_atyp = 2.0 / (1.0 + np.exp(-0.4 * (x - 15.0)))
+                    
+                    transformed = (p1 * c_limbic) + ((p0 + p2) * c_atyp)
+                elif relation == "clinical_subtype_scorer":
+                    beta_val = float(factor.transform.get("beta", 1.0))
+                    apoe4 = sources[0]
+                    vel = sources[1]
+                    burden = sources[2]
+                    mem = sources[3]
+                    
+                    vel_norm = 1.0 - np.exp(-vel)
+                    mem_norm = np.clip(mem, 0.0, 1.0)
+                    
+                    score_limbic = apoe4 + (1.0 - vel_norm) + mem_norm
+                    score_neo = (1.0 - apoe4) + vel_norm + (1.0 - mem_norm)
+                    
+                    raw_scores = np.array([score_neo, score_limbic, score_neo])
+                    exp_scores = np.exp(beta_val * raw_scores)
+                    transformed = exp_scores / np.sum(exp_scores)
+                elif relation == "velocity_modifier_score":
+                    # Score = sum(P_i * W_i)
+                    weights = factor.transform.get("weights", [])
+                    score = sum(s * w for s, w in zip(sources, weights))
+                    transformed = score
                 else:
                     transformed = source
 
                 if factor.coupling_type == "deterministic_transform":
-                    total += (
-                        _DETERMINISTIC_PENALTY
-                        if abs(target - transformed) > _DETERMINISTIC_TOL
-                        else 0.0
-                    )
+                    if not np.allclose(targets, transformed, atol=_DETERMINISTIC_TOL):
+                        total += _DETERMINISTIC_PENALTY
+                elif factor.coupling_type == "directional_potential":
+                    # Directional Potential: Add (sigma * Target * Transformed_Score) to the log-probability
+                    # This pushes the sampler to maximize the reward naturally.
+                    sigma = float(factor.sigma or 1.0)
+                    total += sigma * np.sum(targets * transformed)
                 else:
                     sigma = float(factor.sigma or DEFAULT_COUPLING_SIGMA)
                     var = sigma**2
-                    residual = target - transformed
-                    total += -0.5 * (np.log(2 * np.pi * var) + (residual**2) / var)
+                    residual = targets - transformed
+                    total += np.sum(-0.5 * (np.log(2 * np.pi * var) + (residual**2) / var))
             elif isinstance(factor, SurrogateLikelihoodFactorIR):
                 surrogate = surrogates.get(factor.surrogate_ref)
                 if surrogate is None:
```

### [MODIFY] `src\bayesian_metamodeling\meta\ir.py`

```diff
--- original/src\bayesian_metamodeling\meta\ir.py
+++ modified/src\bayesian_metamodeling\meta\ir.py
@@ -29,7 +29,7 @@
     model_config = ConfigDict(extra="forbid")
 
     kind: Literal["coupling"] = "coupling"
-    coupling_type: Literal["equality_soft", "gaussian_link", "deterministic_transform"]
+    coupling_type: Literal["equality_soft", "gaussian_link", "deterministic_transform", "directional_potential"]
     source: str
     target: str
     transform: dict[str, Any] = Field(default_factory=lambda: {"kind": "identity"})
```

### [MODIFY] `src\bayesian_metamodeling\meta\sampling.py`

```diff
--- original/src\bayesian_metamodeling\meta\sampling.py
+++ modified/src\bayesian_metamodeling\meta\sampling.py
@@ -79,23 +79,96 @@
     for factor in ir.factors:
         if not isinstance(factor, CouplingFactorIR):
             continue
-        source = samples[factor.source]
+        if "," in factor.source:
+            sources = [samples[s.strip()] for s in factor.source.split(",")]
+            source = sources[0]
+        else:
+            sources = [samples[factor.source]]
+            source = sources[0]
+            
         if factor.transform.get("kind") == "affine":
             alpha = float(factor.transform.get("alpha", 1.0))
             beta = float(factor.transform.get("beta", 0.0))
             transformed = alpha * source + beta
+        elif factor.transform.get("kind") == "sum":
+            transformed = sum(sources)
+        elif factor.transform.get("kind") == "integration":
+            dt = float(factor.transform.get("dt", 1.0))
+            alpha = float(factor.transform.get("alpha", 1.0))
+            beta = float(factor.transform.get("beta", 0.0))
+            transformed = alpha * (sources[0] + sources[1] * dt) + beta
+        elif factor.transform.get("kind") == "subtype_conditioned_stage":
+            x = sources[0]
+            p0 = sources[1]
+            p1 = sources[2]
+            p2 = sources[3]
+            
+            c_limbic = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.5))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 1.2))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.8)))
+            c_atyp = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.2))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 0.6))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.5)))
+            
+            transformed = (p1 * c_limbic) + ((p0 + p2) * c_atyp)
+        elif factor.transform.get("kind") == "sustain_to_ode_stage":
+            x = sources[0]
+            p0 = sources[1]
+            p1 = sources[2]
+            p2 = sources[3]
+            
+            # DISCLAIMER: The midpoint anchors (10.0 for Limbic, 15.0 for Atypical) are highly 
+            # dependent on the total number of SuStaIn stages and regions (here assumed 21).
+            # If the number of stages changes, these anchors MUST be recalibrated.
+            c_limbic = 2.0 / (1.0 + np.exp(-0.4 * (x - 10.0)))
+            c_atyp = 2.0 / (1.0 + np.exp(-0.4 * (x - 15.0)))
+            
+            transformed = (p1 * c_limbic) + ((p0 + p2) * c_atyp)
+        elif factor.transform.get("kind") == "clinical_subtype_scorer":
+            beta_val = float(factor.transform.get("beta", 1.0))
+            apoe4 = sources[0]
+            vel = sources[1]
+            burden = sources[2]
+            mem = sources[3]
+            
+            vel_norm = 1.0 - np.exp(-vel)
+            mem_norm = np.clip(mem, 0.0, 1.0)
+            
+            score_limbic = apoe4 + (1.0 - vel_norm) + mem_norm
+            score_neo = (1.0 - apoe4) + vel_norm + (1.0 - mem_norm)
+            
+            raw_scores = np.stack([score_neo, score_limbic, score_neo], axis=0)
+            exp_scores = np.exp(beta_val * raw_scores)
+            transformed = exp_scores / np.sum(exp_scores, axis=0)
+        elif factor.transform.get("kind") == "velocity_modifier_score":
+            weights = factor.transform.get("weights", [])
+            # Dynamic calculation of Score = sum(P_i * W_i)
+            # Sources are expected to be an array/list of probabilities
+            transformed = sum(s * w for s, w in zip(sources, weights))
         else:
             transformed = source
 
+        target_names = [t.strip() for t in factor.target.split(",")] if "," in factor.target else [factor.target]
+
         if factor.coupling_type == "deterministic_transform":
-            samples[factor.target] = transformed
+            for i, target_name in enumerate(target_names):
+                if len(target_names) > 1 and isinstance(transformed, np.ndarray) and transformed.shape[0] == len(target_names):
+                    samples[target_name] = transformed[i]
+                else:
+                    samples[target_name] = transformed
+        elif factor.coupling_type == "directional_potential":
+            # For sampling approximation, directional potential applies a heuristic shift
+            sigma = float(factor.sigma or 1.0)
+            for i, target_name in enumerate(target_names):
+                shift = sigma * transformed * 0.05
+                samples[target_name] += shift
         else:
             sigma = float(factor.sigma or DEFAULT_COUPLING_SIGMA)
             _NUMPYRO_NOISE_SCALE_FACTOR = 1.05
             noise_scale = sigma if backend == "pymc" else sigma * _NUMPYRO_NOISE_SCALE_FACTOR
-            samples[factor.target] = transformed + rng.normal(
-                0.0, noise_scale, size=(chains, draws)
-            )
+            
+            for i, target_name in enumerate(target_names):
+                noise = rng.normal(0.0, noise_scale, size=(chains, draws))
+                if len(target_names) > 1 and isinstance(transformed, np.ndarray) and transformed.shape[0] == len(target_names):
+                    samples[target_name] = transformed[i] + noise
+                else:
+                    samples[target_name] = transformed + noise
 
     sample_id = uuid4().hex
     out_dir = Path("tmp/metamodel_samples") / sample_id
```

### [MODIFY] `src\bayesian_metamodeling\spec\metamodel.py`

```diff
--- original/src\bayesian_metamodeling\spec\metamodel.py
+++ modified/src\bayesian_metamodeling\spec\metamodel.py
@@ -20,7 +20,7 @@
 class MetamodelCouplingSpec(BaseModel):
     model_config = ConfigDict(extra="forbid")
 
-    kind: Literal["gaussian_link", "equality_soft", "deterministic"]
+    kind: Literal["gaussian_link", "equality_soft", "deterministic", "directional_potential"]
     source: str = Field(min_length=1)
     target: str = Field(min_length=1)
     transform: dict[str, Any] = Field(default_factory=lambda: {"kind": "identity"})
```

### [MODIFY] `src\bayesian_metamodeling\surrogates\backends.py`

```diff
--- original/src\bayesian_metamodeling\surrogates\backends.py
+++ modified/src\bayesian_metamodeling\surrogates\backends.py
@@ -446,7 +446,15 @@
 def _require_pymc():
     try:
         with _optional_backend_import_context():
-            import pymc as pm  # type: ignore[import-not-found]
+            import warnings
+
+            with warnings.catch_warnings():
+                warnings.filterwarnings(
+                    "ignore",
+                    category=RuntimeWarning,
+                    module=r".*threadpoolctl",
+                )
+                import pymc as pm  # type: ignore[import-not-found]
     except ModuleNotFoundError as exc:
         raise RuntimeError(
             "Backend 'pymc_gp' requires 'pymc'. "
@@ -698,16 +706,22 @@
             sigma = pm.HalfNormal("sigma", sigma=1.0, shape=(d,))
             pm.Normal("obs", mu=mu, sigma=sigma[None, :], observed=y)
 
-        idata = pm.sample(
-            draws=draws,
-            tune=tune,
-            chains=chains,
-            cores=1,
-            random_seed=seed,
-            target_accept=target_accept,
-            progressbar=False,
-            compute_convergence_checks=False,
-        )
+        with warnings.catch_warnings():
+            warnings.filterwarnings(
+                "ignore",
+                category=RuntimeWarning,
+                module=r".*threadpoolctl",
+            )
+            idata = pm.sample(
+                draws=draws,
+                tune=tune,
+                chains=chains,
+                cores=1,
+                random_seed=seed,
+                target_accept=target_accept,
+                progressbar=False,
+                compute_convergence_checks=False,
+            )
 
     posterior_weights = np.asarray(idata.posterior["beta"], dtype=float).reshape(-1, n_features, d)
     posterior_bias = np.asarray(idata.posterior["intercept"], dtype=float).reshape(-1, d)
```
