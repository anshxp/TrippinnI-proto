import json
from pathlib import Path

import config


class ReportWriter:

    def __init__(self):

        self.output = config.OUTPUT_ROOT / "reports" / "profiling"

        self.output.mkdir(
            parents=True,
            exist_ok=True
        )

    ########################################################

    def save(self, table_name, profile):

        file = self.output / f"{table_name}_profile.json"

        with open(file, "w") as f:

            json.dump(
                profile,
                f,
                indent=4,
                default=str
            )

        print(f"Saved: {file}")