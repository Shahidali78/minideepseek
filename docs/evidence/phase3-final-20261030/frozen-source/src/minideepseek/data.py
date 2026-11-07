"""Data validation independent of model downloads."""

import hashlib
import json
from pathlib import Path

from .core import correct, make_prompt, numeric


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    rows = [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if x.strip()
    ]
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    for r in rows:
        if not r.get("question", "").strip():
            raise ValueError("empty question")
        reference = numeric(r["answer"])
        verified = verified_synthetic_answer(r["question"])
        if verified is not None and reference != verified:
            raise ValueError("mathematically incorrect synthetic reference")
        if r.get("solution") and not correct(r["solution"], r["answer"]):
            raise ValueError(f"invalid solution: {r.get('id')}")
    return rows


def key(question):
    return " ".join(question.casefold().split())


def audit(directory):
    report, sets = {}, {}
    for split in ["train", "val", "test"]:
        path = Path(directory) / f"{split}.jsonl"
        rows = load_rows(path)
        questions = [key(r["question"]) for r in rows]
        sets[split] = set(questions)
        report[split] = {
            "rows": len(rows),
            "unique_questions": len(sets[split]),
            "duplicates": len(rows) - len(sets[split]),
            "sha256": digest(path),
        }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"contamination between {a} and {b}: {len(overlap)}")
    return report


def training_rows(directory):
    audit(directory)
    seen, out = set(), []
    for row in load_rows(Path(directory) / "train.jsonl"):
        k = key(row["question"])
        if k not in seen:
            seen.add(k)
            out.append(row)
    return out


def grpo_records(rows):
    return [{"prompt": make_prompt(r["question"]), "answer": r["answer"]} for r in rows]


def sft_records(rows):
    # Final answers are verified targets. No invented reasoning traces.
    return [
        {
            "prompt": make_prompt(r["question"]),
            "completion": [
                {"role": "assistant", "content": "\\boxed{" + r["answer"] + "}"}
            ],
        }
        for r in rows
    ]


def verified_synthetic_answer(question):
    """Independently recompute known generator templates; unknown sources return None."""
    import re
    from fractions import Fraction

    match = re.fullmatch(
        r"Calculate (-?\d+) ([+*/-]) (-?\d+)\.(?: Give an exact fraction\.)?", question
    )
    if match:
        a, op, b = match.groups()
        a, b = int(a), int(b)
        if op == "+":
            return Fraction(a + b)
        if op == "-":
            return Fraction(a - b)
        if op == "*":
            return Fraction(a * b)
        return Fraction(a, b)
    m = re.fullmatch(r"What is (\d+) percent of (\d+)\?", question)
    if m:
        return Fraction(int(m[1]) * int(m[2]), 100)
    m = re.fullmatch(r"Add the fractions (\d+)/(\d+) and 1/2\.", question)
    if m:
        return Fraction(int(m[1]), int(m[2])) + Fraction(1, 2)
    m = re.fullmatch(
        r"A box has (\d+) balls\. Add (\d+) balls, then give away 3\. How many remain\?",
        question,
    )
    if m:
        return Fraction(int(m[1]) + int(m[2]) - 3)
    m = re.fullmatch(r"Solve for x: 2\*x \+ (\d+) = (\d+)\.", question)
    if m:
        return Fraction(int(m[2]) - int(m[1]), 2)
    m = re.fullmatch(
        r"What number comes next: (\d+), (\d+), (\d+), (\d+)\? Continue the constant difference\.",
        question,
    )
    if m:
        a, b, c, d = map(int, m.groups())
        if b - a != c - b or c - b != d - c:
            raise ValueError("inconsistent sequence")
        return Fraction(d + (d - c))
    return None
