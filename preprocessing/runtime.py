""""Opt-in runner for a single file/table working copy."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from preprocessing.agent import IterativePreprocessingAgent, OllamaPlanner
from preprocessing.registry import build_action_registry, profile_dataframe


def run_dataframe(df: pd.DataFrame, *, file_name: str, output_dir: str | Path, model: str = "qwen3:4b", max_iterations: int = 5, allow_imputation: bool = False) -> dict:
    """Run the agent on a bounded DataFrame copy and write output/report files.

    This entry point deliberately does not read entire EHR files. Callers should
    pass a bounded chunk or an explicitly sized working table.
    """
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    safe_name=Path(file_name).stem.replace(" ","_")
    working=df.copy(deep=True)
    specs,handlers=build_action_registry()
    if allow_imputation:
        specs=[type(s)(s.name,s.module,s.description,s.risk,False) if s.name=="median_imputation" else s for s in specs]
    agent=IterativePreprocessingAgent(OllamaPlanner(model=model),specs,handlers,profile_dataframe,root/"archive",max_iterations=max_iterations)
    result=agent.run(working)
    final_df=result.pop("final_dataset",working)
    output_path=root/f"{safe_name}_processed.csv"
    temp_path=output_path.with_suffix(".tmp")
    final_df.to_csv(temp_path,index=False)
    temp_path.replace(output_path)
    result["file_name"]=file_name
    result["output_file"]=str(output_path)
    report_path=root/f"{safe_name}_preprocessing_report.json"
    report_path.write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    result["report_file"]=str(report_path)
    return result
