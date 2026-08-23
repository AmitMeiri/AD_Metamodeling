"""Compiler boundary for metamodel IR backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bayesian_metamodeling.meta.ir import (
    DEFAULT_COUPLING_SIGMA,
    CouplingFactorIR,
    MetamodelIR,
    PriorFactorIR,
    SurrogateLikelihoodFactorIR,
)
from bayesian_metamodeling.surrogates import SurrogateModel

_DETERMINISTIC_PENALTY = -1e6
_DETERMINISTIC_TOL = 1e-9


@dataclass
class CompiledMetaModel:
    backend: str
    ir: MetamodelIR

    def evaluate_log_prob(
        self, values: dict[str, float], surrogates: dict[str, SurrogateModel] | None = None
    ) -> float:
        surrogates = surrogates or {}
        total = 0.0
        for factor in self.ir.factors:
            if isinstance(factor, PriorFactorIR):
                dist_kind = factor.distribution.get("kind", "normal")
                if dist_kind == "normal":
                    loc = float(factor.distribution.get("loc", 0.0))
                    scale = float(factor.distribution.get("scale", 1.0))
                    x = float(values[factor.variable])
                    var = scale**2
                    total += -0.5 * (np.log(2 * np.pi * var) + ((x - loc) ** 2) / var)
            elif isinstance(factor, CouplingFactorIR):
                # AD_Metamodeling Customization: Support multi-variable source inputs (comma-separated).
                # Resolves to a list of floats (sources), while keeping a single 'source' variable (sources[0])
                # for backwards compatibility with single-input relations.
                if "," in factor.source:
                    sources = [float(values[s.strip()]) for s in factor.source.split(",")]
                    source = sources[0]
                else:
                    sources = [float(values[factor.source])]
                    source = sources[0]
                    
                # AD_Metamodeling Customization: Support multi-variable targets (comma-separated).
                # Target evaluation is represented as a numpy array for vector operations.
                if "," in factor.target:
                    targets = np.array([float(values[t.strip()]) for t in factor.target.split(",")])
                    target = targets[0]
                else:
                    targets = np.array([float(values[factor.target])])
                    target = targets[0]

                relation = factor.transform.get("kind", "identity")
                if relation == "identity":
                    transformed = source
                elif relation == "affine":
                    alpha = float(factor.transform.get("alpha", 1.0))
                    beta = float(factor.transform.get("beta", 0.0))
                    transformed = alpha * source + beta
                elif relation == "sum":
                    # AD_Metamodeling Customization: Sum transform (e.g. sum regional Z-scores into global burden).
                    transformed = sum(sources)
                elif relation == "integration":
                    # AD_Metamodeling Customization: Integration transform (Euler step calculation for biomarker kinetics).
                    dt = float(factor.transform.get("dt", 1.0))
                    alpha = float(factor.transform.get("alpha", 1.0))
                    beta = float(factor.transform.get("beta", 0.0))
                    transformed = alpha * (sources[0] + sources[1] * dt) + beta
                elif relation == "subtype_conditioned_stage":
                    # AD_Metamodeling Customization: Maps SuStaIn stage to ODE clinical stage using 
                    # subtype-conditioned logistics.
                    x = sources[0]
                    p0 = sources[1]
                    p1 = sources[2]
                    p2 = sources[3]
                    
                    c_limbic = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.5))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 1.2))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.8)))
                    c_atyp = 5.0 / (1.0 + np.exp(-10.0 * (x - 0.2))) + 10.0 / (1.0 + np.exp(-10.0 * (x - 0.6))) + 6.0 / (1.0 + np.exp(-10.0 * (x - 1.5)))
                    
                    transformed = (p1 * c_limbic) + ((p0 + p2) * c_atyp)
                elif relation == "sustain_to_ode_stage":
                    # AD_Metamodeling Customization: Maps pySuStaIn 0-21 stage to ODE 0-2 stage.
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
                elif relation == "clinical_subtype_scorer":
                    # AD_Metamodeling Customization: Computes patient propensity scores to membership of 
                    # neo/limbic subtypes using APOE4, rates, and memory.
                    beta_val = float(factor.transform.get("beta", 1.0))
                    apoe4 = sources[0]
                    vel = sources[1]
                    burden = sources[2]
                    mem = sources[3]
                    
                    vel_norm = 1.0 - np.exp(-vel)
                    mem_norm = np.clip(mem, 0.0, 1.0)
                    
                    score_limbic = apoe4 + (1.0 - vel_norm) + mem_norm
                    score_neo = (1.0 - apoe4) + vel_norm + (1.0 - mem_norm)
                    
                    raw_scores = np.array([score_neo, score_limbic, score_neo])
                    exp_scores = np.exp(beta_val * raw_scores)
                    transformed = exp_scores / np.sum(exp_scores)
                elif relation == "velocity_modifier_score":
                    # AD_Metamodeling Customization: Computes linear dot product Score = sum(P_i * W_i) 
                    # of subtype weights to scale biomarker progression rates.
                    weights = factor.transform.get("weights", [])
                    score = sum(s * w for s, w in zip(sources, weights))
                    transformed = score
                else:
                    transformed = source

                if factor.coupling_type == "deterministic_transform":
                    # AD_Metamodeling Customization: Use np.allclose to compare array-valued targets.
                    if not np.allclose(targets, transformed, atol=_DETERMINISTIC_TOL):
                        total += _DETERMINISTIC_PENALTY
                elif factor.coupling_type == "directional_potential":
                    # AD_Metamodeling Customization: Add (1/sigma * Target * Transformed_Score) to log-prob.
                    # This works like a soft directional wind to guide sampling parameters.
                    sigma = float(factor.sigma or 1.0)
                    total += (1.0 / sigma) * np.sum(targets * transformed)
                else:
                    # AD_Metamodeling Customization: Evaluate Gaussian link residual over multi-variable output arrays.
                    sigma = float(factor.sigma or DEFAULT_COUPLING_SIGMA)
                    var = sigma**2
                    residual = targets - transformed
                    total += np.sum(-0.5 * (np.log(2 * np.pi * var) + (residual**2) / var))
            elif isinstance(factor, SurrogateLikelihoodFactorIR):
                surrogate = surrogates.get(factor.surrogate_ref)
                if surrogate is None:
                    # Keep a neutral contribution when surrogate payloads
                    # are not loaded in a sampling-only flow.
                    continue
                inputs = {name: np.array([values[name]], dtype=float) for name in factor.inputs}
                outputs = {name: np.array([values[name]], dtype=float) for name in factor.outputs}
                total += float(np.asarray(surrogate.log_prob(inputs, outputs)).reshape(-1)[0])
        return float(total)


def compile_metamodel(ir: MetamodelIR, backend: str = "pymc") -> CompiledMetaModel:
    if backend in {"pymc", "numpyro"}:
        return CompiledMetaModel(backend=backend, ir=ir)
    raise ValueError(f"Unsupported backend: {backend}")
