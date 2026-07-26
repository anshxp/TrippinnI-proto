"""
LLM Explainer.

Generates natural language explanations for detected data quality issues.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ml.llm.llm_factory import create_llm
from ml.llm.prompt_builder import PromptBuilder
from ml.llm.response_parser import ResponseParser
from models.issue import Issue


class LLMExplainer:
    """
    Generates explanations for detected issues using the configured LLM.
    """

    def __init__(self):

        self.llm = create_llm()
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

    def explain_issue(
        self,
        issue: Issue,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single issue.
        """

        prompt = self.prompt_builder.build(issue)

        response = self.llm.generate(prompt)

        return self.response_parser.parse(response)

    def explain_issues(
        self,
        issues: List[Issue],
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple issues.
        """

        explanations = []

        for issue in issues:
            explanations.append(
                self.explain_issue(issue)
            )

        return explanations