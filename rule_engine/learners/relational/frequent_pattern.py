"""Frequent-pattern rarity learner."""
from __future__ import annotations
from collections import Counter
import pandas as pd

def pattern_scores(df: pd.DataFrame, columns: list[str], min_support: float=0.01) -> list[float]:
    if df.empty or not columns: return [0.0]*len(df)
    tokens=[tuple(f"{c}={row[c]}" for c in columns if pd.notna(row[c])) for _,row in df[columns].iterrows()]
    counts=Counter(tokens); threshold=max(1,int(len(df)*min_support))
    return [0.0 if counts[t]>=threshold else 1.0 for t in tokens]
