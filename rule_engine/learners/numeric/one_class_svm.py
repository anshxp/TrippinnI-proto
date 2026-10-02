"""One-Class SVM numeric anomaly learner."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

def anomaly_scores(values: np.ndarray) -> np.ndarray:
    X=np.asarray(values,dtype=float).reshape(-1,1)
    if len(X)<30: return np.zeros(len(X))
    model=make_pipeline(RobustScaler(),OneClassSVM(kernel="rbf",gamma="scale",nu=0.05))
    model.fit(X)
    return -model.decision_function(X).ravel()
