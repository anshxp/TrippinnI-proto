"""Functional-dependency discovery."""
from __future__ import annotations
import pandas as pd

def dependency_score(df: pd.DataFrame, child: str, parent: str) -> float:
    pair=df[[child,parent]].dropna()
    if pair.empty: return 0.0
    return float((pair.groupby(child,dropna=False)[parent].nunique()<=1).mean())
