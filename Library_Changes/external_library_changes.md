# External Metamodeling Library Modifications

> [!IMPORTANT]
> **Declaration of Modifications & Provision of Files**:
> This document explicitly declares all modifications made to the external **`bayesian_metamodeling`** framework library. For convenience and ease of deployment, the fully updated and modified Python source files containing these additions are provided directly in the [Library_Changes/src](file:///C:/Project/AD_Metamodeling/Library_Changes/src) directory.

This document details the modifications made to the external framework library `bayesian_metamodeling` stored in `C:\Project\metamodeler_codex_scaffold_docs-develop` compared to the original, pristine version of the same library files.

## Summary of Changes

- **New Files**: 0
- **Deleted Files**: 0
- **Modified Files**: 6

## Detailed Code Changes

### [MODIFY] `src\bayesian_metamodeling\meta\builder.py`

```diff
--- original/meta/builder.py
+++ modified/meta/builder.py
@@ -68,8 +68,10 @@
     for ref in spec.surrogate_refs:
         surrogate_id, artifact = _resolve_surrogate_ref(ref)
         var_lists = artifact.get("variable_lists", {})
-        inputs = list(var_lists.get("inputs", []))
-        outputs = list(var_lists.get("outputs", []))
+        # AD_Metamodeling Customization: Support fallback inputs/outputs directly from
+        # the artifact level if the newer "variable_lists" key is absent.
+        inputs = list(var_lists.get("inputs", artifact.get("inputs", [])))
+        outputs = list(var_lists.get("outputs", artifact.get("outputs", [])))
         if not inputs or not outputs:
             raise ValueError(f"Surrogate artifact missing variable lists: {ref}")
 
```

### [MODIFY] `src\bayesian_metamodeling\meta\compiler.py`

```diff
```

### [MODIFY] `src\bayesian_metamodeling\meta\ir.py`

```diff
--- original/meta/ir.py
+++ modified/meta/ir.py
@@ -29,7 +29,9 @@
     model_config = ConfigDict(extra="forbid")
 
     kind: Literal["coupling"] = "coupling"
-    coupling_type: Literal["equality_soft", "gaussian_link", "deterministic_transform"]
+    # AD_Metamodeling Customization: Expand the allowed coupling type literals to support
+    # the custom soft "directional_potential" link used to reward parameter trends.
+    coupling_type: Literal["equality_soft", "gaussian_link", "deterministic_transform", "directional_potential"]
     source: str
     target: str
     transform: dict[str, Any] = Field(default_factory=lambda: {"kind": "identity"})
```

### [MODIFY] `src\bayesian_metamodeling\meta\sampling.py`

```diff
```

### [MODIFY] `src\bayesian_metamodeling\spec\metamodel.py`

```diff
--- original/spec/metamodel.py
+++ modified/spec/metamodel.py
@@ -20,7 +20,9 @@
 class MetamodelCouplingSpec(BaseModel):
     model_config = ConfigDict(extra="forbid")
 
-    kind: Literal["gaussian_link", "equality_soft", "deterministic"]
+    # AD_Metamodeling Customization: Expand allowed spec coupling kinds to include the 
+    # custom "directional_potential" soft potential link.
+    kind: Literal["gaussian_link", "equality_soft", "deterministic", "directional_potential"]
     source: str = Field(min_length=1)
     target: str = Field(min_length=1)
     transform: dict[str, Any] = Field(default_factory=lambda: {"kind": "identity"})
```

### [MODIFY] `src\bayesian_metamodeling\surrogates\backends\__init__.py`

```diff
--- original/surrogates/backends/__init__.py
+++ modified/surrogates/backends/__init__.py
@@ -149,7 +149,17 @@
 def _require_pymc():
     try:
         with _optional_backend_import_context():
-            import pymc as pm  # type: ignore[import-not-found]
+            # AD_Metamodeling Customization: Catch and suppress threadpoolctl RuntimeWarnings 
+            # that can clutter the output when importing PyMC/PyTensor.
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
@@ -379,16 +389,24 @@
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
+        # AD_Metamodeling Customization: Suppress threadpoolctl warnings during sampling 
+        # to ensure clean logs during MCMC execution.
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
