"""Robust covariance learner for multivariate numeric data."""
from __future__ import annotations
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

def anomaly_scores(matrix: np.ndarray, random_state: int=42) -> np.ndarray:
    X=np.asarray(matrix,dtype=float)
    if X.ndim!=2 or len(X)<30 or X.shape[1]<2: return np.zeros(len(X))
    model=make_pipeline(RobustScaler(),EllipticEnvelope(contamination=0.05,random_state=random_state))
    model.fit(X)
    return -model.decision_function(X).ravel()
