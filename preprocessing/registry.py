""""Allow-listed, conservative dataframe actions for the prototype."""
from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.ensemble import IsolationForest
from preprocessing.agent import ActionSpec


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    columns = {}
    for name in df.columns:
        s = df[name]
        item = {"dtype": str(s.dtype), "rows": len(s), "missing_count": int(s.isna().sum()), "missing_fraction": float(s.isna().mean()) if len(s) else 0.0, "distinct_non_null": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            item["numeric_summary"] = {"min": float(s.min()), "q1": float(s.quantile(.25)), "median": float(s.median()), "q3": float(s.quantile(.75)), "max": float(s.max())}
        columns[str(name)] = item
    return {"row_count": len(df), "column_count": len(df.columns), "columns": columns, "duplicate_row_candidates": int(df.duplicated().sum())}


def _column(df, params):
    c = params.get("column")
    if not isinstance(c, str) or c not in df.columns: raise ValueError("parameters.column must be an existing column")
    return c


def _result(df, summary, changed=0, archived=0, details=None):
    return {"status":"committed", "summary":summary, "changed_records":int(changed), "archived_records":int(archived), "dataset":df, "details":details or {}}


def missingness(df, **kwargs):
    return _result(df.copy(deep=True), "Missingness profiled; no values changed.", details=profile_dataframe(df)["columns"])


def duplicates(df, **kwargs):
    count = int(df.duplicated(keep=False).sum())
    return _result(df.copy(deep=True), f"Flag-only exact duplicate scan: {count} rows in duplicate groups; no rows removed.", count, details={"rows_in_duplicate_groups":count})


def iqr_flags(df, *, parameters, **kwargs):
    c = _column(df, parameters)
    if not pd.api.types.is_numeric_dtype(df[c]): raise ValueError("IQR flags require a numeric column")
    s=df[c]; q1=s.quantile(.25); q3=s.quantile(.75); spread=q3-q1
    flags=((s < q1-1.5*spread) | (s > q3+1.5*spread)).fillna(False) if pd.notna(spread) and spread != 0 else pd.Series(False,index=df.index)
    out=df.copy(deep=True); name=f"{c}__iqr_outlier_flag"; out[name]=flags.astype(bool)
    return _result(out, f"Added {name}; no clinical values changed.", int(flags.sum()), details={"column":c,"flag_column":name,"flagged_rows":int(flags.sum())})


def isolation_forest(df, *, parameters, **kwargs):
    cols=parameters.get("columns")
    if not isinstance(cols,list) or not cols or any(c not in df.columns for c in cols): raise ValueError("parameters.columns must list existing columns")
    frame=df[cols].apply(pd.to_numeric,errors="coerce"); valid=frame.notna().all(axis=1)
    if int(valid.sum())<20: raise ValueError("Isolation Forest requires at least 20 complete rows")
    contamination=float(parameters.get("contamination",.02))
    if not .001 <= contamination <= .1: raise ValueError("contamination must be between .001 and .1")
    model=IsolationForest(n_estimators=100,contamination=contamination,random_state=42,n_jobs=1)
    flags=pd.Series(False,index=df.index); flags.loc[valid]=model.fit_predict(frame.loc[valid]) == -1
    out=df.copy(deep=True); out["__isolation_forest_anomaly_flag"]=flags.astype(bool)
    return _result(out,"Added anomaly flags only; no records or values removed.",int(flags.sum()),details={"columns":cols,"flagged_rows":int(flags.sum())})


def median_imputation(df, *, parameters, archive_dir, run_id, action_id, **kwargs):
    c=_column(df,parameters)
    if not pd.api.types.is_numeric_dtype(df[c]): raise ValueError("Median imputation only accepts numeric columns")
    missing=df[c].isna(); count=int(missing.sum())
    if count == 0: return _result(df.copy(deep=True),"No missing values; no change.")
    if df[c].notna().sum()==0: raise ValueError("Cannot impute an all-missing column")
    value=float(df[c].median()); root=Path(archive_dir)/"removed_or_replaced"; root.mkdir(parents=True,exist_ok=True)
    path=root/f"{run_id}_changes.jsonl"
    with path.open("a",encoding="utf-8") as f:
        for pos in range(len(df)):
            if bool(missing.iloc[pos]):
                f.write(json.dumps({"run_id":run_id,"action_id":action_id,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"row_position":pos,"index_label":str(df.index[pos]),"column":c,"original_value":None,"reason":"median imputation","fill_value":value})+"\\n")
    out=df.copy(deep=True); out.loc[missing,c]=value
    return _result(out,f"Median-imputed {count} cells in {c}; original missing cells archived.",count,count,{"column":c,"fill_value":value})


def build_action_registry():
    specs=[
      ActionSpec("report_missingness","rule","Summarize missingness; no mutation."),
      ActionSpec("report_exact_duplicates","rule","Report exact duplicate groups; never delete rows."),
      ActionSpec("flag_iqr_outliers","rule","Add IQR outlier flags for one numeric column."),
      ActionSpec("isolation_forest_flags","ml","Add Isolation Forest anomaly flags."),
      ActionSpec("median_imputation","ml","Median-impute a numeric column after approval; archive original missing cells.",risk="review",requires_approval=True),
    ]
    return specs,{"report_missingness":missingness,"report_exact_duplicates":duplicates,"flag_iqr_outliers":iqr_flags,"isolation_forest_flags":isolation_forest,"median_imputation":median_imputation}
