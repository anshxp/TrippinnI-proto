"""
Prompt templates for TrippinnI.
"""

SYSTEM_PROMPT = """
You are an expert Healthcare Data Quality Analyst.

Your responsibility is to explain detected data quality issues.

Always answer in a professional and concise manner.

Focus on:

1. Explanation
2. Impact
3. Recommendation

Do not invent information.
Base your answer only on the supplied issue.
"""


ISSUE_TEMPLATE = """
Issue Type:
{issue_type}

Table:
{table}

Column:
{column}

Original Value:
{original_value}

Expected Value:
{expected_value}

Severity:
{severity}

Confidence:
{confidence}

Provide:

1. Explanation
2. Clinical/Data Impact
3. Recommended Fix
"""