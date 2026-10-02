"""Local Outlier Factor numeric anomaly learner."""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def anomaly_scores(values: np.ndarray) -> np.ndarray:
    X=np.asarray(values,dtype=float).reshape(-1,1)
    if len(X)<10: return np.zeros(len(X))
    n_neighbors=min(35,max(5,len(X)//10),len(X)-1)
    model=LocalOutlierFactor(n_neighbors=n_neighbors,contamination="auto")
    model.fit_predict(X)
    return -model.negative_outlier_factor_
