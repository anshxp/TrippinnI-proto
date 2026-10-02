"""Combined robust-covariance and autoencoder evidence."""
from __future__ import annotations
import numpy as np
from rule_engine.learners.numeric.robust_covariance import anomaly_scores as covariance_scores
from rule_engine.learners.numeric.autoencoder import reconstruction_errors

def ensemble_reconstruction_scores(matrix: np.ndarray) -> np.ndarray:
    X=np.asarray(matrix,dtype=float)
    if not len(X): return np.array([])
    scores=[covariance_scores(X),reconstruction_errors(X)]
    normalized=[]
    for v in scores:
        if not np.any(v): normalized.append(np.zeros_like(v)); continue
        lo,hi=np.nanpercentile(v,5),np.nanpercentile(v,95)
        normalized.append(np.zeros_like(v) if hi<=lo else np.clip((v-lo)/(hi-lo),0,1))
    return np.mean(normalized,axis=0)
