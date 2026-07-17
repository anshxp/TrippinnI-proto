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