"""Module 3 remediation orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from preprocessing.datatype import DatatypeRemediator
from preprocessing.duplicates import DuplicateRemediator
from preprocessing.encoding import EncodingRemediator
from preprocessing.missing import MissingValueRemediator
from preprocessing.normalization import NormalizationRemediator
from preprocessing.outliers import OutlierRemediator
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog

@dataclass(slots=True)
class RemediationResult:
    dataset: dict[str,pd.DataFrame]
    logs: dict[str,TransformationLog]=field(default_factory=dict)
    summary: dict[str,Any]=field(default_factory=dict)
    def to_dict(self):
        return {"summary":self.summary,"logs":{t:l.to_dict() for t,l in self.logs.items()}}

class RemediationExecutor:
    def __init__(self, policy: RemediationPolicy|None=None):
        self.policy=policy or RemediationPolicy()
        self.missing=MissingValueRemediator()
        self.duplicates=DuplicateRemediator()
        self.datatype=DatatypeRemediator()
        self.normalization=NormalizationRemediator()
        self.outliers=OutlierRemediator()
        self.encoding=EncodingRemediator()

    def run(self,dataset,profiles,quality_results=None):
        if not self.policy.enabled:
            return RemediationResult({k:v.copy() for k,v in dataset.items()},summary={"enabled":False,"transformations":0})
        refined,logs,tables={},{},{}
        for table,frame in dataset.items():
            if not isinstance(frame,pd.DataFrame):
                continue
            log=TransformationLog(); profile=profiles.get(table,{}) or {}
            quality=(quality_results or {}).get(table)
            issues=getattr(quality,"issues",[]) if quality else []
            current=frame.copy(deep=True)
            before_rows,before_cells=len(current),current.size
            current=self.normalization.apply(table,current,profile,log,self.policy)
            current=self.datatype.apply(table,current,profile,log,self.policy)
            current=self.missing.apply(table,current,profile,log,self.policy)
            current=self.duplicates.apply(table,current,profile,log,self.policy)
            current=self.outliers.apply(table,current,issues,log,self.policy)
            current=self.encoding.apply(table,current,profile,log,self.policy)
            refined[table]=current; logs[table]=log
            tables[table]={"rows_before":before_rows,"rows_after":len(current),"cells_before":before_cells,"cells_after":current.size,"transformations":log.count}
        return RemediationResult(refined,logs,{"enabled":True,"tables":tables,"transformations":sum(l.count for l in logs.values()),"scope":"bounded in-memory remediation dataset"})
