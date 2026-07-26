"""
Response Parser.
"""

import json


class ResponseParser:

    @staticmethod
    def parse(
        response: str,
    ) -> dict:

        try:
            return json.loads(response)

        except Exception:

            return {
                "raw_response": response
            }