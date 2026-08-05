import pandas as pd


class KeyDetector:

    """
    Detects candidate primary keys and foreign keys.
    """

    FK_KEYWORDS = {

        "patient",

        "encounter",

        "organization",

        "provider",

        "payer",

        "device"

    }

    ###############################################################

    def detect(self, dataframe: pd.DataFrame):

        return {

            "primary_keys": self.primary_keys(dataframe),

            "foreign_keys": self.foreign_keys(dataframe)

        }

    ###############################################################

    def primary_keys(self, dataframe):

        candidates = []

        for column in dataframe.columns:

            if dataframe[column].is_unique:

                candidates.append(column)

        return candidates

    ###############################################################

    def foreign_keys(self, dataframe):

        candidates = []

        for column in dataframe.columns:

            name = column.lower()

            for keyword in self.FK_KEYWORDS:

                if keyword in name:

                    candidates.append(column)

                    break

        return candidates

    def start_streaming(self) -> dict:
        return {"columns": [], "seen_values": {}, "not_unique": set()}

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        state["columns"] = list(dataframe.columns)
        for column in dataframe.columns:
            if column in state["not_unique"]:
                continue
            seen = state["seen_values"].setdefault(column, set())
            hashes = pd.util.hash_pandas_object(dataframe[column], index=False)
            for value_hash in hashes:
                if value_hash in seen:
                    state["not_unique"].add(column)
                    # No further values need to be retained once uniqueness
                    # has been disproven for this column.
                    state["seen_values"].pop(column, None)
                    break
                seen.add(value_hash)

    def finalize_streaming(self, state: dict) -> dict:
        primary_keys = [
            column for column in state["columns"]
            if column not in state["not_unique"]
        ]
        foreign_keys = []
        for column in state["columns"]:
            if any(keyword in column.lower() for keyword in self.FK_KEYWORDS):
                foreign_keys.append(column)
        return {"primary_keys": primary_keys, "foreign_keys": foreign_keys}

    def detect_chunks(self, chunks) -> dict:
        state = self.start_streaming()
        for chunk in chunks:
            self.update_streaming(state, chunk)
        return self.finalize_streaming(state)
