"""Lightweight pairwise association-rule discovery."""
from __future__ import annotations
from itertools import combinations
from typing import Any
import pandas as pd

def discover_rules(df: pd.DataFrame, columns: list[str], min_support: float=0.05, min_confidence: float=0.95, max_columns: int=8) -> list[dict[str,Any]]:
    columns=columns[:max_columns]
    if len(columns)<2 or df.empty: return []
    rules=[]; n=len(df)
    for left,right in combinations(columns,2):
        pair=df[[left,right]].dropna()
        if pair.empty: continue
        lc=pair[left].value_counts(); rc=pair[right].value_counts()
        for (a,b),count in pair.groupby([left,right],dropna=False).size().items():
            support=float(count/n)
            if support<min_support: continue
            cab=float(count/lc[a]); cba=float(count/rc[b])
            if cab>=min_confidence: rules.append({"antecedent":{left:a},"consequent":{right:b},"support":support,"confidence":cab})
            if cba>=min_confidence: rules.append({"antecedent":{right:b},"consequent":{left:a},"support":support,"confidence":cba})
    return rules
