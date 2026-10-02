"""Robust temporal distribution evidence."""
from __future__ import annotations
import numpy as np

def directional_support(delta_seconds: np.ndarray) -> dict[str,float]:
    v=np.asarray(delta_seconds,dtype=float); v=v[np.isfinite(v)]
    if not len(v): return {"positive":0.0,"negative":0.0,"zero":0.0}
    return {"positive":float(np.mean(v>=0)),"negative":float(np.mean(v<=0)),"zero":float(np.mean(v==0))}
