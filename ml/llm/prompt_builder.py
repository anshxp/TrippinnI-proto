"""
Prompt Builder.
"""

from models.issue import Issue

from ml.llm.templates import (
    SYSTEM_PROMPT,
    ISSUE_TEMPLATE,
)


class PromptBuilder:

    @staticmethod
    def build_issue_prompt(
        issue: Issue,
    ) -> str:

        issue_prompt = ISSUE_TEMPLATE.format(
            issue_type=issue.issue_type,
            table=issue.table,
            column=issue.column,
            original_value=issue.original_value,
            expected_value=issue.expected_value,
            severity=issue.severity,
            confidence=issue.confidence,
        )

        return (
            SYSTEM_PROMPT
            + "\n\n"
            + issue_prompt
        )