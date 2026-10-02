"""Isolation Forest numeric anomaly learner."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest

def anomaly_scores(values: np.ndarray, random_state: int = 42) -> np.ndarray:
    X=np.asarray(values,dtype=float).reshape(-1,1)
    model=IsolationForest(n_estimators=200,contamination="auto",random_state=random_state,n_jobs=-1)
    model.fit(X)
    return -model.score_samples(X)
