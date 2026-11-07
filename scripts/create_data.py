"""Verified synthetic additions; existing datasets are never overwritten."""

import argparse
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minideepseek.data import audit, key, load_rows
from minideepseek.runtime import save_json


def build(split, low, high, count, seed, excluded=None):
    rng = random.Random(seed)
    seen = set(excluded or [])
    records = []
    for attempt in range(count * 100):
        a, b = rng.randint(low, high), rng.randint(low, high)
        kind = attempt % 9
        q, answer = [
            (f"Calculate {a} + {b}.", a + b),
            (f"Calculate {a} - {b}.", a - b),
            (f"Calculate {a} * {b}.", a * b),
            (f"Calculate {a} / {b}. Give an exact fraction.", Fraction(a, b)),
            (f"What is {a} percent of {b}?", Fraction(a * b, 100)),
            (f"Add the fractions {a}/{b} and 1/2.", Fraction(a, b) + Fraction(1, 2)),
            (
                f"A box has {a} balls. Add {b} balls, then give away 3. How many remain?",
                a + b - 3,
            ),
            (f"Solve for x: 2*x + {a} = {a + 2 * b}.", b),
            (
                f"What number comes next: {a}, {a + b}, {a + 2 * b}, {a + 3 * b}? Continue the constant difference.",
                a + 4 * b,
            ),
        ][kind]
        if key(q) in seen:
            continue
        seen.add(key(q))
        records.append(
            {
                "id": f"{split}-{len(records):05}",
                "question": q,
                "answer": str(answer),
                "category": kind,
            }
        )
        if len(records) == count:
            return records
    raise ValueError("not enough unique questions")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/expanded-v1")
    p.add_argument("--gsm8k", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(
            "Choose a new dataset directory; preserving existing files"
        )
    reserved_path = Path("data/test.jsonl")
    if reserved_path.exists():
        reserved = load_rows(reserved_path)
    else:
        # Recreate the starter's public synthetic protocol only when no reserved
        # file exists. Existing local test data always takes precedence.
        rng = random.Random(3)
        reserved = []
        for i in range(240):
            x, y = rng.randint(71, 110), rng.randint(71, 110)
            op = ["+", "-", "*"][i % 3]
            value = {"+": x + y, "-": x - y, "*": x * y}[op]
            reserved.append(
                {
                    "id": f"test-{i:05d}",
                    "question": f"Calculate {x} {op} {y}.",
                    "answer": str(value),
                }
            )
    excluded = {key(r["question"]) for r in reserved}
    if a.gsm8k:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
        rows = []
        for i, r in enumerate(ds):
            q = r["question"]
            if key(q) not in excluded:
                excluded.add(key(q))
                rows.append(
                    {
                        "id": f"gsm8k-train-{i}",
                        "question": q,
                        "answer": r["answer"].rsplit("####", 1)[1].strip(),
                    }
                )
        splits = {"train": rows[200:], "val": rows[:200], "test": reserved}
    else:
        train = build("train", 1, 80, 1200, a.seed, excluded)
        val = build(
            "val",
            1,
            80,
            150,
            a.seed + 1,
            excluded | {key(r["question"]) for r in train},
        )
        splits = {"train": train, "val": val, "test": reserved}
    out.mkdir(parents=True)
    for split, rows in splits.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    save_json(
        out / "manifest.json",
        {
            "seed": a.seed,
            "source": "gsm8k-train" if a.gsm8k else "synthetic-v2",
            "audit": audit(out),
        },
    )
    print(json.dumps(audit(out), indent=2))


if __name__ == "__main__":
    main()
