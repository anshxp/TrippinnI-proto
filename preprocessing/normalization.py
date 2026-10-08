"""Conservative normalization helpers."""
from __future__ import annotations
import pandas as pd
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog, TransformationRecord

class NormalizationRemediator:
    name="NormalizationRemediator"
    def apply(self,table,df,profile,log,policy):
        if not policy.enabled: return df
        result=df.copy()
        if policy.normalize_whitespace:
            for column in result.columns:
                if not pd.api.types.is_object_dtype(result[column]): continue
                old=result[column]
                new=old.map(lambda v:v.strip() if isinstance(v,str) else v)
                changed=old.notna() & new.notna() & old.ne(new)
                for idx in result.index[changed]:
                    log.add(TransformationRecord(table=table,row_index=int(idx),column=str(column),
                        action="normalize_text",reason="leading_or_trailing_whitespace",
                        original_value=old.at[idx],new_value=new.at[idx],method="strip",detector="profile"))
                result[column]=new
        return result
