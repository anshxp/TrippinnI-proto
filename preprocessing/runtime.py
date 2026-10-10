""""Opt-in bounded runner for one DataFrame or chunk."""
from __future__ import annotations
import json, time, uuid
from pathlib import Path
import pandas as pd
from preprocessing.agent import OllamaPlanner
from preprocessing.registry import build_action_registry, profile_dataframe


def run_dataframe(df: pd.DataFrame, *, file_name: str, output_dir: str | Path, model: str = "qwen3:4b", max_iterations: int = 5, allow_imputation: bool = False) -> dict:
    """Profile -> LLM selects one registered tool -> execute -> re-profile.

    Pass a bounded chunk/table. This function does not read whole source files.
    """
    if max_iterations < 1: raise ValueError("max_iterations must be positive")
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    safe_name=Path(file_name).stem.replace(" ","_")
    archive=root/"archive"; archive.mkdir(parents=True,exist_ok=True)
    specs,handlers=build_action_registry()
    if not allow_imputation: specs=[s for s in specs if s.name!="median_imputation"]
    spec_map={s.name:s for s in specs}
    planner=OllamaPlanner(model=model)
    working=df.copy(deep=True)
    profile=profile_dataframe(working)
    history=[]; reports=[]; run_id=uuid.uuid4().hex
    for iteration in range(1,max_iterations+1):
        decision=planner.choose_action(profile,specs,history)
        name=decision.get("action")
        if decision.get("stop") is True or name=="stop": break
        if name not in spec_map: raise ValueError(f"LLM selected unregistered action: {name!r}")
        if not isinstance(decision.get("parameters",{}),dict): raise ValueError("LLM parameters must be an object")
        if any(x["action"]==name and x["status"]=="committed" for x in history):
            history.append({"iteration":iteration,"action":name,"status":"repeated_action_blocked"}); break
        started=time.time(); action_id=uuid.uuid4().hex; before=profile
        result=handlers[name](working,parameters=decision.get("parameters",{}),archive_dir=archive,run_id=run_id,action_id=action_id)
        if result.get("status")!="committed":
            reports.append({"action":name,"status":result.get("status","failed"),"error":result.get("error"),"before_profile":before})
            break
        # Handlers return a new DataFrame. The original caller's DataFrame stays unchanged.
        working=result.get("dataset",working)
        profile=profile_dataframe(working)
        record={"run_id":run_id,"action_id":action_id,"iteration":iteration,"action":name,"module":spec_map[name].module,"status":"committed","rationale":str(decision.get("rationale","")),"parameters":decision.get("parameters",{}),"summary":result.get("summary",""),"changed_records":result.get("changed_records",0),"archived_records":result.get("archived_records",0),"details":result.get("details",{}),"before_profile":before,"after_profile":profile,"elapsed_seconds":round(time.time()-started,3)}
        reports.append(record); history.append({"iteration":iteration,"action":name,"status":"committed","summary":record["summary"]})
        with (archive/f"{run_id}_action_log.jsonl").open("a",encoding="utf-8") as stream: stream.write(json.dumps(record,default=str)+"\n")
    output_path=root/f"{safe_name}_processed.csv"; tmp=output_path.with_suffix(".tmp")
    working.to_csv(tmp,index=False); tmp.replace(output_path)
    report_path=root/f"{safe_name}_preprocessing_report.json"
    report={"run_id":run_id,"file_name":file_name,"iterations":len(reports),"model":model,"final_profile":profile,"actions":reports,"output_file":str(output_path),"archive_dir":str(archive)}
    report_path.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    report["report_file"]=str(report_path)
    return report
