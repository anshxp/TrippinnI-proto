"""Report-driven static visualizations for profiling and quality results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

import config


class ReportVisualizer:
    """Generate charts without reloading source tables into memory."""

    def __init__(self, output_root: Path | None = None):
        self.reports = config.OUTPUT_ROOT / "reports"
        self.output = output_root or self.reports / "visualizations"
        self.output.mkdir(parents=True, exist_ok=True)

    def generate(self) -> dict[str, str]:
        profiles = self._load(self.reports / "profiling", "_profile.json")
        quality = self._load(self.reports / "quality", ".json")
        if not profiles:
            raise FileNotFoundError("No profiling reports found.")
        summary = self._summary(profiles, quality)
        summary_path = self.output / "dashboard_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        generated = {"dashboard_summary": str(summary_path)}
        charts = {
            "table_scale": self._table_scale(profiles),
            "missingness_by_table": self._missingness_by_table(profiles),
            "top_missing_columns": self._top_missing_columns(profiles),
            "quality_score_by_table": self._quality_scores(quality),
            "issues_by_detector": self._detectors(quality),
            "issues_by_severity": self._severities(quality),
            "semantic_types": self._semantic_types(profiles),
        }
        for name, data in charts.items():
            if not data:
                continue
            path = self.output / f"{name}.png"
            self._bar(path, data, self._title(name), self._ylabel(name))
            generated[name] = str(path)
        return generated

    @staticmethod
    def _load(directory: Path, suffix: str) -> dict[str, dict[str, Any]]:
        result = {}
        if not directory.exists():
            return result
        for path in sorted(directory.glob(f"*{suffix}")):
            if path.name.startswith("."):
                continue
            try:
                result[path.stem] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        return result

    def _summary(self, profiles, quality):
        tables = []
        semantic = {}
        for key, profile in profiles.items():
            d = profile.get("dataset", {})
            table = d.get("table_name") or key.removesuffix("_profile")
            q = quality.get(table) or quality.get(f"{table}_quality") or {}
            tables.append({
                "table": table,
                "rows": d.get("rows", 0),
                "columns": d.get("columns", 0),
                "missing_percentage": d.get("missing_percentage", 0.0),
                "duplicate_rows": d.get("duplicate_rows", 0),
                "memory_mb": profile.get("memory", {}).get("total_memory_mb", 0.0),
                "quality_score": q.get("quality_score"),
                "total_issues": q.get("total_issues", 0),
                "detection_sample_rows": (q.get("summary") or {}).get("evaluation", {}).get("sample_rows"),
                "detection_sample_fraction": (q.get("summary") or {}).get("evaluation", {}).get("sample_fraction_of_profile"),
                "detector_summary": q.get("detector_summary", {}),
                "severity_summary": q.get("severity_summary", {}),
            })
            for column in profile.get("columns", {}).values():
                t = column.get("semantic_type") or "unknown"
                semantic[t] = semantic.get(t, 0) + 1
        return {
            "report_version": 2,
            "quality_scope": {
                "profiling_scope": "deterministic first prefix",
                "detection_scope": "bounded reservoir sample",
                "interpretation": "issue counts are unique findings; missingness is scored by affected cells",
            },
            "table_count": len(tables),
            "total_rows_profiled": sum(int(t["rows"] or 0) for t in tables),
            "total_columns": sum(int(t["columns"] or 0) for t in tables),
            "semantic_type_counts": semantic,
            "tables": sorted(tables, key=lambda x: x["table"]),
        }

    def _table_scale(self, profiles):
        return self._profile_values(profiles, "rows")

    def _missingness_by_table(self, profiles):
        return self._profile_values(profiles, "missing_percentage")

    def _top_missing_columns(self, profiles):
        values = []
        for key, p in profiles.items():
            table = p.get("dataset", {}).get("table_name") or key.removesuffix("_profile")
            for c in p.get("columns", {}).values():
                values.append((f"{table}.{c.get('name', 'unknown')}",
                               float(c.get("null_percentage", 0) or 0)))
        return sorted(values, key=lambda x: x[1], reverse=True)[:20]

    def _quality_scores(self, quality):
        values = []
        for key, q in quality.items():
            score = q.get("quality_score")
            if score is not None:
                values.append((key.removesuffix("_quality"), float(score)))
        return sorted(values, key=lambda x: x[1])

    def _detectors(self, quality):
        counts = {}
        for q in quality.values():
            for name, count in (q.get("detector_summary") or {}).items():
                counts[name] = counts.get(name, 0) + int(count or 0)
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    def _severities(self, quality):
        counts = {}
        for q in quality.values():
            for name, count in (q.get("severity_summary") or {}).items():
                counts[name] = counts.get(name, 0) + int(count or 0)
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    def _semantic_types(self, profiles):
        counts = {}
        for p in profiles.values():
            for c in p.get("columns", {}).values():
                name = c.get("semantic_type") or "unknown"
                counts[name] = counts.get(name, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _profile_values(profiles, field):
        values = []
        for key, p in profiles.items():
            d = p.get("dataset", {})
            table = d.get("table_name") or key.removesuffix("_profile")
            values.append((table, float(d.get(field, 0) or 0)))
        return sorted(values, key=lambda x: x[1], reverse=(field != "rows"))

    @staticmethod
    def _title(name):
        return name.replace("_", " ").title()

    @staticmethod
    def _ylabel(name):
        if name == "table_scale":
            return "Rows"
        if "missing" in name:
            return "Missing cells (%)"
        if "quality" in name:
            return "Quality score"
        if "semantic" in name:
            return "Column count"
        return "Issue count"

    @staticmethod
    def _bar(path, data, title, ylabel):
        if not data:
            return
        labels = [x[0] for x in data]
        values = [x[1] for x in data]
        fig, ax = plt.subplots(figsize=(11, max(4.5, min(14, 0.34 * len(labels) + 2.5))))
        ax.barh(range(len(labels)), values)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel)
        ax.set_title(title)
        if title == "Table Scale" and any(v > 0 for v in values):
            ax.set_xscale("log")
        if "Quality" in title:
            ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
