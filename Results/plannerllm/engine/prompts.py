"""Prompt templates used by the executor and the agents.

Kept in one place so a PROMPT_REWRITE directive (which swaps an operator's
instruction text) and the report appendix both refer to the same strings.
"""

SYSTEM_EXEC = ("You are a careful data-processing component inside a database pipeline. "
               "Follow the instruction exactly and do not add commentary.")

GENERATE = """{instruction}

Question: {question}

Context:
{context}

Answer:"""

MAP = """{instruction}

Question: {question}

Text:
{text}

Relevant text (or NONE):"""

SQL_GENERATE = """{instruction}

Database schema (table(columns)):
{schema}
{examples}
Question: {question}

Return only the SQL query."""

SQL_REPAIR = """The SQLite query below failed or returned no rows.

Schema:
{schema}

Question: {question}
Query: {sql}
Problem: {problem}

Return only a corrected SQLite query."""

# ----------------------------------------------------------------- agents
SYSTEM_AGENT = ("You are an expert query optimizer for LLM-powered data pipelines. "
                "You reply with strict JSON only.")

CRITERIA = """A pipeline performs this task: {task}
The final operator is of kind '{kind}'.
List at most {k} concrete criteria a reviewer should use to judge one output of this
pipeline (accuracy, completeness, faithfulness to the provided context, format...).
Reply as JSON: {{"criteria": [{{"name": "...", "description": "..."}}]}}"""

RATE = """You are validating the output of a data pipeline.
Task: {task}

Question: {question}
Context the pipeline used (truncated):
{context}

Output to judge:
{output}

Score the output from 1 (very bad) to 5 (excellent) on each criterion:
{criteria}
Reply as JSON: {{"scores": {{"<criterion name>": <1-5>, ...}}}}"""

RATE_BATCH = """You are validating the outputs of a data pipeline.
Task: {task}

Criteria:
{criteria}

Score each of the {n} items below from 1 (very bad) to 5 (excellent) on every criterion. Judge only
from the question and the context shown; do not reward length.

{items}

Reply as JSON: {{"items": [{{"id": 1, "scores": {{"<criterion name>": <1-5>, ...}}}}, ...]}}"""

PAIRWISE = """Two versions of a data pipeline answered the same question.
Task: {task}
Question: {question}
Context (truncated):
{context}

Output A:
{a}

Output B:
{b}

Which output better satisfies the task (accuracy, completeness, faithfulness)?
Reply as JSON: {{"winner": "A" or "B", "reason": "<one sentence>"}}"""

PROPOSE = """Task of the pipeline: {task}
Current pipeline (operators in order): {pipeline}
Data statistics: {stats}

Available rewrite directives (name: what it does):
{catalog}

Pick the directives most likely to improve accuracy per dollar for this pipeline.
Reply as JSON: {{"directives": [{{"name": "<DIRECTIVE>", "why": "<short reason>"}}]}}"""

PROMPT_REWRITE = """The following instruction is used by the '{kind}' operator of a pipeline whose task is:
{task}

Current instruction:
{prompt}

Rewrite the instruction so a small 8B model follows it more reliably: be specific
about what to output, the format, and to rely only on the given context.
Reply as JSON: {{"prompt": "<new instruction>"}}"""
