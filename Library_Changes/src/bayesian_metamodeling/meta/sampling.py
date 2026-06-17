"""Metamodel sampling and artifact persistence."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from bayesian_metamodeling.meta.compiler import compile_metamodel
from bayesian_metamodeling.meta.ir import (
    DEFAULT_COUPLING_SIGMA,
    CouplingFactorIR,
    MetamodelIR,
    PriorFactorIR,
)
from bayesian_metamodeling.spec import MetaModelSpec
from bayesian_metamodeling.storage._filelock import locked_registry

META_SAMPLE_REGISTRY_PATH = Path("tmp/metamodel_samples_registry.json")


def list_meta_samples() -> list[dict[str, str]]:
    if not META_SAMPLE_REGISTRY_PATH.exists():
        return []
    registry = json.loads(META_SAMPLE_REGISTRY_PATH.read_text(encoding="utf-8"))
    return [
        {"sample_id": sid, "backend": payload.get("backend", "")}
        for sid, payload in sorted(registry.items())
    ]


def _prior_params(ir: MetamodelIR) -> dict[str, tuple[float, float]]:
    params = {var.name: (0.0, 1.0) for var in ir.variables}
    for factor in ir.factors:
        if isinstance(factor, PriorFactorIR):
            kind = factor.distribution.get("kind", "normal")
            if kind == "normal":
                params[factor.variable] = (
                    float(factor.distribution.get("loc", 0.0)),
                    float(factor.distribution.get("scale", 1.0)),
                )
    return params


def _sample_core(
    *,
    backend: str,
    ir: MetamodelIR,
    spec_payload: dict,
    dataset_digest: str,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> dict[str, str]:
    """Generate samples from a metamodel IR using prior draws and coupling transforms.

    Note: coupling constraints are applied as a post-draw transform approximation.
    This means soft couplings inject noise *after* the prior draw rather than being
    jointly sampled. The approximation is exact for deterministic transforms but
    introduces a bias for soft/Gaussian links compared to full MCMC sampling.
    """
    compiled = compile_metamodel(ir, backend=backend)
    priors = _prior_params(ir)

    rng = np.random.default_rng(seed)
    sample_vars = sorted(var.name for var in ir.variables)
    samples = {
        name: rng.normal(loc=priors[name][0], scale=priors[name][1], size=(chains, draws))
        for name in sample_vars
    }

    # Apply coupling constraints as a simple post-draw transform baseline.
    for factor in ir.factors:
        if not isinstance(factor, CouplingFactorIR):
            continue
        if "," in factor.source:
            sources = [samples[s.strip()] for s in factor.source.split(",")]
            source = sources[0]
        else:
            sources = [samples[factor.source]]
            source = sources[0]
            
        if factor.transform.get("kind") == "affine":
            alpha = float(factor.transform.get("alpha", 1.0))
            beta = float(factor.transform.get("beta", 0.0))
            transformed = alpha * source + beta
        elif factor.transform.get("kind") == "sum":
            transformed = sum(sources)
        elif factor.transform.get("kind") == "integration":
            dt = float(factor.transform.get("dt", 1.0))
            alpha = float(factor.transform.get("alpha", 1.0))
            beta = float(factor.transform.get("beta", 0.0))
            transformed = alpha * (sources[0] + sources[1] * dt) + beta
        elif factor.transform.get("kind") == "subtype_conditioned_stage":
            x = sources[0]
            p0 = sources[1]
            p1 = sources[2]
            p2 = sources[3]
            
            c_limbic = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.5))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 1.2))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.8)))
            c_atyp = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.2))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 0.6))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.5)))
            
            transformed = (p1 * c_limbic) + ((p0 + p2) * c_atyp)
        elif factor.transform.get("kind") == "sustain_to_ode_stage":
            x = sources[0]
            p0 = sources[1]
            p1 = sources[2]
            p2 = sources[3]
            
            # DISCLAIMER: The midpoint anchors (10.0 for Limbic, 15.0 for Atypical) are highly 
            # dependent on the total number of SuStaIn stages and regions (here assumed 21).
            # If the number of stages changes, these anchors MUST be recalibrated.
            c_limbic = 2.0 / (1.0 + np.exp(-0.4 * (x - 10.0)))
            c_atyp = 2.0 / (1.0 + np.exp(-0.4 * (x - 15.0)))
            
            transformed = (p1 * c_limbic) + ((p0 + p2) * c_atyp)
        elif factor.transform.get("kind") == "clinical_subtype_scorer":
            beta_val = float(factor.transform.get("beta", 1.0))
            apoe4 = sources[0]
            vel = sources[1]
            burden = sources[2]
            mem = sources[3]
            
            vel_norm = 1.0 - np.exp(-vel)
            mem_norm = np.clip(mem, 0.0, 1.0)
            
            score_limbic = apoe4 + (1.0 - vel_norm) + mem_norm
            score_neo = (1.0 - apoe4) + vel_norm + (1.0 - mem_norm)
            
            raw_scores = np.stack([score_neo, score_limbic, score_neo], axis=0)
            exp_scores = np.exp(beta_val * raw_scores)
            transformed = exp_scores / np.sum(exp_scores, axis=0)
        elif factor.transform.get("kind") == "velocity_modifier_score":
            weights = factor.transform.get("weights", [])
            # Dynamic calculation of Score = sum(P_i * W_i)
            # Sources are expected to be an array/list of probabilities
            transformed = sum(s * w for s, w in zip(sources, weights))
        else:
            transformed = source

        target_names = [t.strip() for t in factor.target.split(",")] if "," in factor.target else [factor.target]

        if factor.coupling_type == "deterministic_transform":
            for i, target_name in enumerate(target_names):
                if len(target_names) > 1 and isinstance(transformed, np.ndarray) and transformed.shape[0] == len(target_names):
                    samples[target_name] = transformed[i]
                else:
                    samples[target_name] = transformed
        elif factor.coupling_type == "directional_potential":
            # For sampling approximation, directional potential applies a heuristic shift
            sigma = float(factor.sigma or 1.0)
            for i, target_name in enumerate(target_names):
                shift = sigma * transformed * 0.05
                samples[target_name] += shift
        else:
            sigma = float(factor.sigma or DEFAULT_COUPLING_SIGMA)
            _NUMPYRO_NOISE_SCALE_FACTOR = 1.05
            noise_scale = sigma if backend == "pymc" else sigma * _NUMPYRO_NOISE_SCALE_FACTOR
            
            for i, target_name in enumerate(target_names):
                noise = rng.normal(0.0, noise_scale, size=(chains, draws))
                if len(target_names) > 1 and isinstance(transformed, np.ndarray) and transformed.shape[0] == len(target_names):
                    samples[target_name] = transformed[i] + noise
                else:
                    samples[target_name] = transformed + noise

    sample_id = uuid4().hex
    out_dir = Path("tmp/metamodel_samples") / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_payload = {
        "dims": {"chain": chains, "draw": draws},
        "variables": {name: values.tolist() for name, values in samples.items()},
    }
    dataset_path = out_dir / "samples_dataset.json"
    dataset_path.write_text(json.dumps(dataset_payload, indent=2, sort_keys=True))

    inference_payload = {
        "backend": backend,
        "name": ir.name,
        "draws": draws,
        "chains": chains,
        "seed": seed,
        "variables": sample_vars,
        "created_at": datetime.now(UTC).isoformat(),
    }
    inference_path = out_dir / "inference_data.json"
    inference_path.write_text(json.dumps(inference_payload, indent=2, sort_keys=True))

    spec_digest = hashlib.sha256(
        json.dumps(spec_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset_digest_hash = hashlib.sha256(dataset_digest.encode("utf-8")).hexdigest()
    registry_entry = {
        "sample_id": sample_id,
        "ir_name": ir.name,
        "backend": backend,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "seed": seed,
        "spec_digest": spec_digest,
        "dataset_digest": dataset_digest_hash,
        "dependency_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "inference_data_path": str(inference_path),
        "samples_dataset_path": str(dataset_path),
        "created_at": datetime.now(UTC).isoformat(),
    }

    with locked_registry(META_SAMPLE_REGISTRY_PATH):
        registry = {}
        if META_SAMPLE_REGISTRY_PATH.exists():
            registry = json.loads(META_SAMPLE_REGISTRY_PATH.read_text(encoding="utf-8"))
        registry[sample_id] = registry_entry
        META_SAMPLE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_SAMPLE_REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True))

    # Ensure compiled backend is executable.
    probe = {name: float(samples[name][0, 0]) for name in sample_vars}
    _ = compiled.evaluate_log_prob(probe, surrogates={})

    return {
        "sample_id": sample_id,
        "inference_data_path": str(inference_path),
        "samples_dataset_path": str(dataset_path),
    }


def sample_metamodel(
    *,
    spec: MetaModelSpec,
    ir: MetamodelIR,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> dict[str, str]:
    spec_payload = spec.model_dump(mode="json")
    dataset_digest = json.dumps(spec.surrogate_refs, sort_keys=True)

    if spec.ppl_backend == "pymc":
        return _sample_core(
            backend="pymc",
            ir=ir,
            spec_payload=spec_payload,
            dataset_digest=dataset_digest,
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
        )
    if spec.ppl_backend == "numpyro":
        return _sample_core(
            backend="numpyro",
            ir=ir,
            spec_payload=spec_payload,
            dataset_digest=dataset_digest,
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
        )
    raise ValueError(f"Unsupported ppl_backend: {spec.ppl_backend}")
