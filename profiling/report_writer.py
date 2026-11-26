import json
from pathlib import Path

import config


class ReportWriter:
    """Persist profile JSON reports to a configurable directory."""

    def __init__(self, output_dir=None):
        self.output = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else config.OUTPUT_ROOT / "reports" / "profiling"
        )
        self.output.mkdir(parents=True, exist_ok=True)

    def save(self, table_name, profile, relative_path=None):
        destination = self.output
        if relative_path is not None:
            destination = destination / Path(relative_path)
        destination.mkdir(parents=True, exist_ok=True)

        file = destination / f"{table_name}_profile.json"
        with file.open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4, default=str)
        print(f"Saved: {file}")
        return file
