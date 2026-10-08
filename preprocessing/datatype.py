"""Loss-minimizing datatype normalization driven by profiler metadata."""
from __future__ import annotations
import pandas as pd
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog, TransformationRecord

class DatatypeRemediator:
    name="DatatypeRemediator"
    def apply(self,table,df,profile,log,policy):
        if not policy.enabled or not policy.allows("datatype_normalization"): return df
        columns=profile.get("columns",{}) if isinstance(profile,dict) else {}
        result=df.copy()
        for column,meta in columns.items():
            if column not in result.columns or not isinstance(meta,dict): continue
            expected=str(meta.get("expected_dtype") or meta.get("dtype") or "").lower()
            converted=self._convert(result[column],expected) if expected else None
            if converted is None: continue
            for idx in result.index:
                old,new=result.at[idx,column],converted.at[idx]
                if pd.isna(old) and pd.isna(new): continue
                try: changed=old!=new
                except (TypeError,ValueError): changed=True
                if changed:
                    log.add(TransformationRecord(table=table,row_index=int(idx),column=str(column),
                        action="normalize_dtype",reason="profile_conformance",original_value=old,
                        new_value=new,method=expected,detector="DatatypeDetector"))
            result[column]=converted
        return result

    @staticmethod
    def _convert(series,expected):
        try:
            if "datetime" in expected or "date" in expected:
                c=pd.to_datetime(series,errors="coerce")
                return None if (series.notna() & c.isna()).any() else c
            if "bool" in expected:
                n=series.astype("string").str.strip().str.lower()
                c=n.map({"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False})
                return None if (series.notna() & c.isna()).any() else c
            if "int" in expected or "integer" in expected:
                n=pd.to_numeric(series,errors="coerce")
                if (series.notna() & n.isna()).any() or n.dropna().mod(1).ne(0).any(): return None
                return n.astype("Int64")
            if any(x in expected for x in ("float","double","numeric")):
                n=pd.to_numeric(series,errors="coerce")
                return None if (series.notna() & n.isna()).any() else n
        except (TypeError,ValueError,OverflowError):
            return None
        return None
