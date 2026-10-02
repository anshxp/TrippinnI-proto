"""Conditional numeric anomaly learner."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

def residual_scores(matrix: np.ndarray,random_state: int=42) -> np.ndarray:
    X=np.asarray(matrix,dtype=float)
    if X.ndim!=2 or X.shape[1]<2 or len(X)<40: return np.zeros(len(X))
    residuals=np.zeros(len(X)); cv=KFold(n_splits=min(5,len(X)),shuffle=True,random_state=random_state)
    for target in range(X.shape[1]):
        predictors=np.delete(X,target,axis=1)
        model=make_pipeline(RobustScaler(),RandomForestRegressor(n_estimators=100,random_state=random_state,n_jobs=-1))
        pred=cross_val_predict(model,predictors,X[:,target],cv=cv,n_jobs=1)
        residuals+=np.abs(X[:,target]-pred)
    return residuals/X.shape[1]
