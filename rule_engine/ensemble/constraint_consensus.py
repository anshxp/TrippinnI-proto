"""Consensus across independent constraint learners."""
from __future__ import annotations
from typing import Iterable

def consensus(scores: Iterable[float],threshold: float=0.5)->dict[str,float|bool]:
    values=[min(1.0,max(0.0,float(x))) for x in scores]
    if not values: return {"score":0.0,"agreement":0.0,"flag":False}
    return {"score":sum(values)/len(values),"agreement":sum(x>=threshold for x in values)/len(values),"flag":sum(x>=threshold for x in values)/len(values)>=0.5}
