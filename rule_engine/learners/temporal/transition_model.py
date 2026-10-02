"""Empirical categorical event-transition model."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import pandas as pd

def learn_transitions(df: pd.DataFrame,event_column: str,time_column: str) -> dict[tuple[Any,Any],float]:
    work=df[[event_column,time_column]].copy()
    work[time_column]=pd.to_datetime(work[time_column],errors="coerce",format="mixed")
    work=work.dropna().sort_values(time_column)
    transitions=defaultdict(int); totals=defaultdict(int); previous=None
    for event in work[event_column]:
        if previous is not None:
            transitions[(previous,event)]+=1; totals[previous]+=1
        previous=event
    return {k:transitions[k]/totals[k] for k in transitions if totals[k]}
