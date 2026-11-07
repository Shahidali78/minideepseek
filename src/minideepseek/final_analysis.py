"""Conservative, observable error labels for frozen held-out outputs."""

import re
from collections import Counter

from .core import parse_final

CATEGORIES = (
    "arithmetic_error",
    "multi_step_reasoning_error",
    "incorrect_operation_selection",
    "answer_formatting_failure",
    "failed_answer_extraction",
    "truncated_completion",
    "unsupported_answer_pattern",
    "unclassified_numeric_error",
)


def error_label(response, outcome, truncated, question):
    """One operational label per incorrect completion; no hidden-reasoning claims.

    Truncation has precedence, even if a partial response contains a wrong answer.
    Arithmetic error means a parsed wrong final value on a simple arithmetic task.
    It does not diagnose which internal calculation failed. Unobserved reasoning
    and operation-selection causes are deliberately left unassigned.
    """
    if outcome == "correct":
        return "correct"
    if truncated:
        return "truncated_completion"
    if outcome == "parse_failure":
        if "\\boxed" in response or "<answer>" in response:
            return "unsupported_answer_pattern"
        if re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", response):
            return "answer_formatting_failure"
        return "failed_answer_extraction"
    if parse_final(response) is None:
        raise ValueError("numeric-error outcome without parsed answer")
    if re.fullmatch(r"Calculate -?\d+ [+*/-] -?\d+\.", question):
        return "arithmetic_error"
    return "unclassified_numeric_error"


def error_summary(records):
    all_counts = Counter()
    first_counts = Counter()
    for row in records:
        for i, (text, outcome, truncated) in enumerate(
            zip(row["responses"], row["outcomes"], row["truncated"], strict=True)
        ):
            label = error_label(text, outcome, truncated, row["question"])
            all_counts[label] += 1
            if i == 0:
                first_counts[label] += 1
    labels = ("correct", *CATEGORIES)
    return {
        "all_samples": {k: all_counts[k] for k in labels},
        "first_samples": {k: first_counts[k] for k in labels},
        "interpretation": "Operational output labels; zero causal categories mean not established, not proven absent. Truncation and parser metrics are also reported independently.",
    }
